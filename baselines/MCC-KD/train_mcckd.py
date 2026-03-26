"""
MCC-KD 训练脚本 - DeepMath 版本
实现 Multi-CoT Consistent Distillation 和 Overall Objective

与 MCC-KD 官方实现对齐：
1. 数据格式: instruction, output_a, output_b, indices_a, indices_b
2. 损失计算: L_total = L_ce + alpha * L_kl
   - L_ce: 对 output_a 和 output_b 分别计算交叉熵损失，取平均
   - L_kl: 对 answer 区域的 logits 计算双向 KL 散度
"""

import os
import sys
import json
import logging
import argparse
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import deepspeed

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """模型参数"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "预训练模型路径或 HuggingFace 模型名称"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "PyTorch 数据类型", "choices": ["float16", "bfloat16", "float32"]}
    )


@dataclass
class DataArguments:
    """数据参数"""
    train_file: str = field(
        default="/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_mcckd.jsonl",
        metadata={"help": "训练数据路径"}
    )
    max_length: int = field(
        default=8192,
        metadata={"help": "最大序列长度"}
    )
    preprocessing_num_workers: int = field(
        default=16,
        metadata={"help": "预处理并行数"}
    )


@dataclass
class MCCKDArguments:
    """MCC-KD 特定参数 - 与官方实现对齐"""
    alpha: float = field(
        default=0.1,
        metadata={"help": "KL 损失的权重系数 (对应 MCC-KD 官方实现的 alpha)"}
    )
    kl_temperature: float = field(
        default=1.0,
        metadata={"help": "KL 散度的温度系数 T"}
    )
    kl_eps: float = field(
        default=7e-5,
        metadata={"help": "KL 计算中的数值稳定性常数 eps"}
    )
    rationales_per_question: int = field(
        default=2,
        metadata={"help": "每个问题的推理路径数量"}
    )
    consistency_weight: float = field(
        default=1.0,
        metadata={"help": "一致性损失的权重系数"}
    )
    use_consistency_loss: bool = field(
        default=True,
        metadata={"help": "是否使用一致性损失"}
    )


@dataclass
class LoraArguments:
    """LoRA 参数"""
    use_lora: bool = field(
        default=True,
        metadata={"help": "是否使用 LoRA"}
    )
    lora_r: int = field(
        default=128,
        metadata={"help": "LoRA 秩"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha 参数"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "LoRA 目标模块"}
    )


def powmax(tensor, exponent=1, dim=-1, eps=7e-5):
    """
    类似于 softmax，执行幂次归一化（与 MCC-KD 官方实现一致）
    """
    numerator = torch.pow(tensor, exponent=exponent)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / (denominator + eps)


class KLDivLoss(nn.Module):
    """
    KL 散度损失 - 与 MCC-KD 官方实现 src/criterion.py 中的 KLDivLoss 对齐
    """
    def __init__(self, eps=7e-5):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor = None,
        T: float = 1.0
    ):
        """
        计算 KL-Divergence 损失
        
        Args:
            logits: 学生模型的 logits [batch, seq_len, vocab_size]
            targets: 目标 logits (来自另一个 rationale) [batch, seq_len, vocab_size]
            masks: 掩码，标识有效位置 [batch, seq_len]
            T: 温度参数
        
        Returns:
            标量损失值
        """
        # 展平为二维 [batch*seq_len, vocab_size]
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, targets.size(-1)).to(logits)
        
        # 计算概率分布（使用温度 T 进行软化）
        estimates = torch.softmax(logits.float(), dim=-1).type_as(logits)
        targets = torch.softmax(targets.float() / T, dim=-1).type_as(targets)
        
        # 使用 powmax 进行平滑归一化（与 MCC-KD 官方实现一致）
        estimates = powmax(estimates + self.eps)
        targets = powmax(targets + self.eps)
        
        # 计算 KL 散度: KL(target || estimate) = sum(target * (log(target) - log(estimate)))
        loss = targets * (torch.log(targets) - torch.log(estimates))
        loss = torch.sum(loss, dim=-1)
        
        # 应用 mask
        if masks is not None:
            masks = masks.view(-1).to(logits.device)
            loss = torch.masked_select(loss, masks.bool())
        
        return torch.mean(loss)


class MCCDKDataset(Dataset):
    """
    MCC-KD 数据集 - 与官方实现格式对齐
    
    数据格式:
    {
        "instruction": "问题文本",
        "output_a": "<think>\nthought1\n</think>\n\nsolution1",
        "output_b": "<think>\nthought2\n</think>\n\nsolution2",
        "indices_a": [[start1, end1]],  # answer 在 output_a 中的字符位置
        "indices_b": [[start2, end2]]   # answer 在 output_b 中的字符位置
    }
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 8192):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 读取数据
        self.raw_data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc="Loading dataset", total=len(lines), disable=len(lines) < 1000):
            line = line.strip()
            if line:
                self.raw_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.raw_data)} training samples (each with 2 rationales)")
    
    def __len__(self):
        return len(self.raw_data)
    
    def _tokenize_with_indices(self, instruction: str, output: str, indices: List[List[int]]) -> Dict:
        """
        对 instruction 和 output 进行 tokenize，并计算 indices 对应的 token 位置
        
        Args:
            instruction: 问题文本
            output: 格式化后的输出（包含 think 和 solution）
            indices: [[start_char, end_char]] answer 的字符位置
        
        Returns:
            包含 input_ids, attention_mask, labels, answer_token_indices 的字典
        """
        # 使用 chat template 构建完整文本
        messages = [
            {"role": "system", "content": "You are a helpful assistant that solves math problems step by step."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        
        # 构建 prompt（不包含 assistant 回复）
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 构建完整文本
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        full_tokens = self.tokenizer(full_text, add_special_tokens=False)
        
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        
        # 构建 labels：prompt 部分设为 -100，只计算 output 部分的 loss
        prompt_len = len(prompt_tokens["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        
        # 计算 answer 在 token 序列中的位置
        # indices 是字符级别的 [[start_char, end_char]]
        answer_token_indices = []
        if indices and len(indices) > 0 and indices[0][0] >= 0:
            char_start, char_end = indices[0]
            
            # 计算 output 在 full_text 中的起始字符位置（即 prompt 之后）
            output_char_start = len(prompt)
            
            # 计算 answer 在 full_text 中的绝对字符位置
            answer_char_start = output_char_start + char_start
            answer_char_end = output_char_start + char_end
            
            # 将字符位置映射到 token 位置
            # 方法：对 full_text 的前缀进行编码，找到 answer 的起始和结束 token
            prefix_before_answer = full_text[:answer_char_start]
            answer_with_suffix = full_text[answer_char_start:answer_char_end]
            
            prefix_tokens = self.tokenizer(prefix_before_answer, add_special_tokens=False)["input_ids"]
            answer_tokens = self.tokenizer(answer_with_suffix, add_special_tokens=False)["input_ids"]
            
            token_start = len(prefix_tokens)
            token_end = token_start + len(answer_tokens)
            
            answer_token_indices = [[token_start, token_end]]
        else:
            # 如果没有 indices，使用整个 output 部分
            answer_token_indices = [[prompt_len, len(input_ids)]]
        
        # 截断到 max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
            # 同步截断 answer_indices，确保不越界
            truncated_indices = []
            for start, end in answer_token_indices:
                if start < self.max_length:
                    truncated_indices.append([start, min(end, self.max_length)])
            answer_token_indices = truncated_indices if truncated_indices else [[prompt_len, self.max_length]]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "answer_indices": answer_token_indices,  # token 级别的位置
        }
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.raw_data[idx]
        
        instruction = item.get("instruction", "")
        output_a = item.get("output_a", "")
        output_b = item.get("output_b", "")
        indices_a = item.get("indices_a")
        indices_b = item.get("indices_b")
        
        # DEBUG: 打印第一个样本的原始数据信息
        if idx == 0:
            logger.info(f"[DEBUG] Dataset sample 0 - Raw data keys: {list(item.keys())}")
            logger.info(f"[DEBUG] Dataset sample 0 - indices_a: {indices_a}, indices_b: {indices_b}")
        
        # 分别 tokenize output_a 和 output_b
        tokens_a = self._tokenize_with_indices(instruction, output_a, indices_a)
        tokens_b = self._tokenize_with_indices(instruction, output_b, indices_b)
        
        result = {
            "input_ids_a": torch.tensor(tokens_a["input_ids"], dtype=torch.long),
            "attention_mask_a": torch.tensor(tokens_a["attention_mask"], dtype=torch.long),
            "labels_a": torch.tensor(tokens_a["labels"], dtype=torch.long),
            "answer_indices_a": tokens_a["answer_indices"],
            
            "input_ids_b": torch.tensor(tokens_b["input_ids"], dtype=torch.long),
            "attention_mask_b": torch.tensor(tokens_b["attention_mask"], dtype=torch.long),
            "labels_b": torch.tensor(tokens_b["labels"], dtype=torch.long),
            "answer_indices_b": tokens_b["answer_indices"],
        }
        
        # DEBUG: 打印返回结果的信息
        if idx == 0:
            logger.info(f"[DEBUG] Dataset sample 0 - Returned keys: {list(result.keys())}")
            logger.info(f"[DEBUG] Dataset sample 0 - answer_indices_a type: {type(result['answer_indices_a'])}, value: {result['answer_indices_a']}")
        
        return result


class MCCKDTrainer(Trainer):
    """
    MCC-KD 训练器 - 与官方实现 ParallelSolverMccDistillTrainer 对齐
    
    损失函数:
    L_total = L_ce + alpha * L_kl
    
    其中:
    - L_ce: (CE_loss_a + CE_loss_b) / 2
    - L_kl: 对 answer 区域的 logits 计算双向 KL 散度
    """
    
    def __init__(self, mcckd_args: MCCKDArguments, **kwargs):
        super().__init__(**kwargs)
        self.mcckd_args = mcckd_args
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion_kl = KLDivLoss(eps=mcckd_args.kl_eps)
    
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, any]]) -> Dict[str, Union[torch.Tensor, any]]:
        """
        重写 _prepare_inputs 以确保 answer_indices 不会丢失
        
        父类方法会尝试将所有输入转换为 tensor，但 answer_indices 是嵌套列表，
        需要特殊处理以避免丢失。
        """
        # 保存 answer_indices（它们不是 tensor，不应该被转换或移动到设备）
        answer_indices_a = inputs.pop("answer_indices_a", None)
        answer_indices_b = inputs.pop("answer_indices_b", None)
        
        # 调用父类方法处理其他 tensor 输入
        inputs = super()._prepare_inputs(inputs)
        
        # 将 answer_indices 重新添加回 inputs
        if answer_indices_a is not None:
            inputs["answer_indices_a"] = answer_indices_a
        if answer_indices_b is not None:
            inputs["answer_indices_b"] = answer_indices_b
        
        return inputs
    
    def compute_kl_for_mcc(
        self,
        indices_a: List[List[List[int]]],
        indices_b: List[List[List[int]]],
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        T: float
    ) -> torch.Tensor:
        """
        计算 MCC-KD 的 KL 损失 - 与官方实现 compute_kl_for_mcc 方法对齐
        
        根据 indices 提取每个 sample answer 区域的 logits，计算双向 KL 散度
        
        Args:
            indices_a: answer 在 output_a 中的 token 位置，结构为 [[[start1, end1]], [[start2, end2]], ...]
            indices_b: answer 在 output_b 中的 token 位置，结构为 [[[start1, end1]], [[start2, end2]], ...]
            logits_a: output_a 的 logits [batch, seq_len, vocab_size]
            logits_b: output_b 的 logits [batch, seq_len, vocab_size]
            T: 温度参数
        
        Returns:
            KL 损失标量（batch 内所有 sample 的平均）
        """
        batch_size = logits_a.size(0)
        total_kl_loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            # 获取当前 sample 的索引（假设每个 sample 只有一个 answer 区域）
            sample_indices_a = indices_a[i] if i < len(indices_a) else []
            sample_indices_b = indices_b[i] if i < len(indices_b) else []
            
            # 提取 [start, end]，如果不存在则使用整个序列
            if sample_indices_a and len(sample_indices_a) > 0:
                start_a, end_a = sample_indices_a[0]
            else:
                start_a, end_a = 0, logits_a.size(1)
            
            if sample_indices_b and len(sample_indices_b) > 0:
                start_b, end_b = sample_indices_b[0]
            else:
                start_b, end_b = 0, logits_b.size(1)
            
            # 提取当前 sample answer 区域的 logits
            # logits_a[i]: [seq_len, vocab_size]
            sample_logits_a = logits_a[i, start_a:end_a, :]  # [answer_len_a, vocab_size]
            sample_logits_b = logits_b[i, start_b:end_b, :]  # [answer_len_b, vocab_size]
            
            # 对齐序列长度（取较短的长度）
            min_len = min(sample_logits_a.size(0), sample_logits_b.size(0))
            if min_len == 0:
                continue  # 跳过空的 answer 区域
            
            sample_logits_a = sample_logits_a[:min_len, :]
            sample_logits_b = sample_logits_b[:min_len, :]
            
            # 添加 batch 维度用于 KL 计算 [1, min_len, vocab_size]
            sample_logits_a = sample_logits_a.unsqueeze(0)
            sample_logits_b = sample_logits_b.unsqueeze(0)
            
            # 创建 mask
            masks = torch.ones(1, min_len, device=sample_logits_a.device)
            
            # 计算双向 KL 散度
            p_loss = self.criterion_kl.forward(sample_logits_a, sample_logits_b, masks=masks, T=T)
            q_loss = self.criterion_kl.forward(sample_logits_b, sample_logits_a, masks=masks, T=T)
            
            sample_kl_loss = (p_loss + q_loss) * 0.5
            total_kl_loss += sample_kl_loss
            valid_samples += 1
        
        # 返回 batch 内所有 sample 的平均 KL loss
        if valid_samples > 0:
            return total_kl_loss / valid_samples
        else:
            # 如果没有有效样本，返回 0
            return torch.tensor(0.0, device=logits_a.device, requires_grad=True)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        计算 MCC-KD 的总损失 - 与官方实现 ParallelSolverMccDistillTrainer.distill 对齐
        
        使用拼接 batch 的方案，只进行一次前向传播，避免 DeepSpeed ZeRO 的
        "parameter has already been reduced" 错误。
        
        L_total = L_ce + alpha * L_kl
        
        其中:
        - L_ce = (CE_loss_a + CE_loss_b) / 2
        - L_kl = alpha * bidirectional_KL(answer_logits_a, answer_logits_b)
        """
        # DEBUG: 检查 inputs 的内容
        if self.state.global_step == 0:
            logger.info(f"[DEBUG] compute_loss - inputs keys: {list(inputs.keys())}")
            for k, v in inputs.items():
                if hasattr(v, 'shape'):
                    logger.info(f"[DEBUG] compute_loss - {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                else:
                    logger.info(f"[DEBUG] compute_loss - {k}: type={type(v)}, value={v}")
        
        # 提取 input_a 和 input_b
        input_ids_a = inputs["input_ids_a"]
        attention_mask_a = inputs["attention_mask_a"]
        labels_a = inputs["labels_a"]
        answer_indices_a = inputs.get("answer_indices_a", None)
        
        input_ids_b = inputs["input_ids_b"]
        attention_mask_b = inputs["attention_mask_b"]
        labels_b = inputs["labels_b"]
        answer_indices_b = inputs.get("answer_indices_b", None)
        
        # 防御性检查：确保 answer_indices 存在
        if answer_indices_a is None or answer_indices_b is None:
            logger.error(f"[ERROR] answer_indices is None! inputs keys: {list(inputs.keys())}")
            # 如果丢失，使用默认的全序列范围
            seq_len_a = input_ids_a.size(1)
            seq_len_b = input_ids_b.size(1)
            answer_indices_a = [[[0, seq_len_a]] * input_ids_a.size(0)]
            answer_indices_b = [[[0, seq_len_b]] * input_ids_b.size(0)]
        
        # ========== 拼接 batch：将两个输入合并为一个大 batch ==========
        # 这样可以只进行一次前向传播，避免 DeepSpeed ZeRO 的梯度重复 reduce 问题
        batch_size = input_ids_a.size(0)
        
        # 获取 pad_token_id（用于 padding input_ids）
        pad_token_id = getattr(model.config, 'pad_token_id', 0)
        
        # 处理不同长度：将 a 和 b padding 到相同长度
        seq_len_a = input_ids_a.size(1)
        seq_len_b = input_ids_b.size(1)
        max_seq_len = max(seq_len_a, seq_len_b)
        
        if seq_len_a < max_seq_len:
            # Padding input_ids_a
            pad_len = max_seq_len - seq_len_a
            input_ids_a = torch.nn.functional.pad(input_ids_a, (0, pad_len), value=pad_token_id)
            attention_mask_a = torch.nn.functional.pad(attention_mask_a, (0, pad_len), value=0)
            labels_a = torch.nn.functional.pad(labels_a, (0, pad_len), value=-100)
            
        if seq_len_b < max_seq_len:
            # Padding input_ids_b
            pad_len = max_seq_len - seq_len_b
            input_ids_b = torch.nn.functional.pad(input_ids_b, (0, pad_len), value=pad_token_id)
            attention_mask_b = torch.nn.functional.pad(attention_mask_b, (0, pad_len), value=0)
            labels_b = torch.nn.functional.pad(labels_b, (0, pad_len), value=-100)
        
        # 拼接成一个大 batch [2*batch_size, max_seq_len]
        combined_input_ids = torch.cat([input_ids_a, input_ids_b], dim=0)
        combined_attention_mask = torch.cat([attention_mask_a, attention_mask_b], dim=0)
        combined_labels = torch.cat([labels_a, labels_b], dim=0)
        
        # ========== 只有一次前向传播！==========
        outputs = model(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask
        )
        combined_logits = outputs.logits  # [2*batch_size, max_seq_len, vocab_size]
        
        # 拆分 logits
        logits_a = combined_logits[:batch_size]  # [batch_size, max_seq_len, vocab_size]
        logits_b = combined_logits[batch_size:]  # [batch_size, max_seq_len, vocab_size]
        
        # 截取回原始长度（去除 padding 部分）用于计算 CE loss
        if seq_len_a < max_seq_len:
            logits_a_for_ce = logits_a[:, :seq_len_a, :]
            labels_a_for_ce = labels_a[:, :seq_len_a]
        else:
            logits_a_for_ce = logits_a
            labels_a_for_ce = labels_a
            
        if seq_len_b < max_seq_len:
            logits_b_for_ce = logits_b[:, :seq_len_b, :]
            labels_b_for_ce = labels_b[:, :seq_len_b]
        else:
            logits_b_for_ce = logits_b
            labels_b_for_ce = labels_b
        
        # 计算 CE Loss A
        shift_logits_a = logits_a_for_ce[..., :-1, :].contiguous()
        shift_labels_a = labels_a_for_ce[..., 1:].contiguous()
        ce_loss_a = self.criterion_ce(
            shift_logits_a.view(-1, shift_logits_a.size(-1)),
            shift_labels_a.view(-1)
        )
        
        # 计算 CE Loss B
        shift_logits_b = logits_b_for_ce[..., :-1, :].contiguous()
        shift_labels_b = labels_b_for_ce[..., 1:].contiguous()
        ce_loss_b = self.criterion_ce(
            shift_logits_b.view(-1, shift_logits_b.size(-1)),
            shift_labels_b.view(-1)
        )
        
        # 平均 CE Loss
        ce_loss = (ce_loss_a + ce_loss_b) * 0.5
        
        # ========== 计算 KL Loss ==========
        # 注意：计算 KL 时使用原始的 logits（包含 padding 部分也没关系，因为 answer_indices 会指定有效区域）
        kl_loss = self.compute_kl_for_mcc(
            indices_a=answer_indices_a,
            indices_b=answer_indices_b,
            logits_a=logits_a,
            logits_b=logits_b,
            T=self.mcckd_args.kl_temperature
        )
        
        # 加权 KL Loss
        kl_loss_weighted = self.mcckd_args.alpha * kl_loss
        
        # ========== 总损失 ==========
        total_loss = ce_loss + kl_loss_weighted
        
        # 记录各个损失组件（用于监控）
        if self.state.global_step % 100 == 0:
            self.log({
                "train/ce_loss": ce_loss.item(),
                "train/kl_loss": kl_loss.item(),
                "train/kl_loss_weighted": kl_loss_weighted.item(),
                "train/total_loss": total_loss.item(),
            })
        
        if return_outputs:
            return total_loss, {"logits_a": logits_a, "logits_b": logits_b}
        return total_loss


def mcckd_data_collator(features, tokenizer, max_length):
    """
    MCC-KD 数据整理器 - 处理成对的 output_a 和 output_b
    """
    
    # 分离 answer_indices（不需要转换为 tensor）
    answer_indices_a = [f.pop("answer_indices_a") for f in features]
    answer_indices_b = [f.pop("answer_indices_b") for f in features]
    
    # 分别收集 output_a 和 output_b 的字段
    features_a = []
    features_b = []
    
    for f in features:
        features_a.append({
            "input_ids": f["input_ids_a"],
            "attention_mask": f["attention_mask_a"],
            "labels": f["labels_a"],
        })
        features_b.append({
            "input_ids": f["input_ids_b"],
            "attention_mask": f["attention_mask_b"],
            "labels": f["labels_b"],
        })
    
    # 使用 DataCollatorForSeq2Seq 分别处理
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding='longest',
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    
    batch_a = collator(features_a)
    batch_b = collator(features_b)
    
    # 合并为统一的 batch
    batch = {
        "input_ids_a": batch_a["input_ids"],
        "attention_mask_a": batch_a["attention_mask"],
        "labels_a": batch_a["labels"],
        "answer_indices_a": answer_indices_a,
        
        "input_ids_b": batch_b["input_ids"],
        "attention_mask_b": batch_b["attention_mask"],
        "labels_b": batch_b["labels"],
        "answer_indices_b": answer_indices_b,
    }
    
    # DEBUG: 验证 batch 内容
    #logger.info(f"[DEBUG] Collator output batch keys: {list(batch.keys())}")
    #logger.info(f"[DEBUG] answer_indices_a type: {type(batch['answer_indices_a'])}, len: {len(batch['answer_indices_a'])}")
    #logger.info(f"[DEBUG] answer_indices_a[0]: {batch['answer_indices_a'][0] if len(batch['answer_indices_a']) > 0 else 'N/A'}")
    
    return batch


def main():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, MCCKDArguments, LoraArguments, TrainingArguments))
    
    # 检查是否有 --config 参数
    config_file = None
    config_file_type = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
            if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                config_file_type = "yaml"
            elif config_file.endswith(".json"):
                config_file_type = "json"
            sys.argv = sys.argv[:i] + sys.argv[i+2:]
            break
    
    if config_file and config_file_type == "yaml":
        import yaml
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        args_list = []
        for key, value in config_dict.items():
            if isinstance(value, bool):  
                args_list.extend([f"--{key}", str(value).lower()])
            elif isinstance(value, list):
                args_list.append(f"--{key}")
                args_list.extend([str(v) for v in value])
            else:
                args_list.extend([f"--{key}", str(value)])
        model_args, data_args, mcckd_args, lora_args, training_args = \
            parser.parse_args_into_dataclasses(args=args_list)
    elif config_file and config_file_type == "json":
        model_args, data_args, mcckd_args, lora_args, training_args = \
            parser.parse_json_file(json_file=os.path.abspath(config_file))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, mcckd_args, lora_args, training_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, mcckd_args, lora_args, training_args = \
            parser.parse_args_into_dataclasses()
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 初始化
    with tqdm(total=4, desc="Initialization", position=0) as pbar:
        # 加载 tokenizer
        logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            padding_side="right"
        )
        pbar.update(1)
        
        # 设置 pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 加载模型
        logger.info(f"Loading model from {model_args.model_name_or_path}")
        torch_dtype = getattr(torch, model_args.torch_dtype)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
            use_cache=False,
        )
        pbar.update(1)
    
        # 应用 LoRA
        if lora_args.use_lora:
            logger.info("Applying LoRA")
            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        pbar.update(1)
    
        # 加载数据集
        logger.info(f"Loading dataset from {data_args.train_file}")
        train_dataset = MCCDKDataset(
            data_path=data_args.train_file,
            tokenizer=tokenizer,
            max_length=data_args.max_length
        )
        pbar.update(1)
        
        # DEBUG: 检查数据集中的样本
        #logger.info(f"[DEBUG] Dataset size: {len(train_dataset)}")
        #if len(train_dataset) > 0:
        #    sample_0 = train_dataset[0]
        #    logger.info(f"[DEBUG] Sample 0 from dataset keys: {list(sample_0.keys())}")
        #    for k, v in sample_0.items():
        #        if hasattr(v, 'shape'):
        #            logger.info(f"[DEBUG] Sample 0 - {k}: shape={v.shape}, dtype={v.dtype}")
        #        else:
        #            logger.info(f"[DEBUG] Sample 0 - {k}: type={type(v)}, value={v[:100] if isinstance(v, list) and len(v) > 0 else v}")

        # 创建 Trainer
        logger.info("Creating trainer")
        
        # 创建带有 tokenizer 和 max_length 的 data_collator
        from functools import partial
        collator_fn = partial(mcckd_data_collator, tokenizer=tokenizer, max_length=data_args.max_length)
        #collator_fn = partial(mcckd_data_collator, tokenizer=tokenizer)
        
        trainer = MCCKDTrainer(
            mcckd_args=mcckd_args,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator_fn,
            tokenizer=tokenizer,
        )
        
        # DEBUG: 测试数据加载器
        #logger.info("[DEBUG] Testing dataloader...")
        #train_dataloader = trainer.get_train_dataloader()
        #try:
        #    # 尝试获取第一个 batch
        #    first_batch = next(iter(train_dataloader))
        #    logger.info(f"[DEBUG] First batch keys: {list(first_batch.keys())}")
        #    for k, v in first_batch.items():
        #        if hasattr(v, 'shape'):
        #            logger.info(f"[DEBUG] First batch - {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
        #        else:
        #            logger.info(f"[DEBUG] First batch - {k}: type={type(v)}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")
        #            if isinstance(v, list) and len(v) > 0:
        #                logger.info(f"[DEBUG] First batch - {k}[0]: {v[0]}")
        #except Exception as e:
        #    logger.error(f"[DEBUG] Error getting first batch: {e}")
        #    import traceback
        #    logger.error(traceback.format_exc())
    
    # 训练
    logger.info("Starting training")
    
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    else:
        # 确保输出目录存在，避免 get_last_checkpoint 报错
        os.makedirs(training_args.output_dir, exist_ok=True)
        if last_checkpoint := get_last_checkpoint(training_args.output_dir):
            checkpoint = last_checkpoint
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # 保存模型
    trainer.save_model()
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
