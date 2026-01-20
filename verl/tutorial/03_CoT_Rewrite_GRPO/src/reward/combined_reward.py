"""
组合奖励函数

R = R_cog + lambda * R_logic

这个文件定义了 Verl 需要的奖励函数接口
"""

from typing import Any
import torch
import numpy as np

from verl import DataProto
from .cognitive_reward import get_cognitive_reward_computer
from .logic_reward import get_fsf_analyzer, compute_logic_reward


def compute_cot_rewriter_reward(
    data: DataProto,
    tokenizer: Any,
    student_model_path: str = "./models/Qwen2.5-3B-Instruct",
    lambda_logic: float = 0.5,
    use_fsf_reward: bool = True,
    fsf_judge_type: str = "api",
    fsf_api_url: str = "http://localhost:8000/v1/chat/completions",
    **kwargs,
) -> dict[str, Any]:
    """
    CoT Rewriter 组合奖励函数

    Verl 调用的入口函数

    Args:
        data: Verl DataProto
        tokenizer: 分词器
        student_model_path: Student Model 路径
        lambda_logic: 逻辑奖励权重
        use_fsf_reward: 是否使用 FSF 奖励
        fsf_judge_type: FSF Judge 类型
        fsf_api_url: FSF API 地址

    Returns:
        {"reward_tensor": Tensor, "reward_extra_info": dict}
    """
    batch_size = data.batch.batch_size[0]

    # 解码文本
    prompts = tokenizer.batch_decode(
        data.batch["prompts"],
        skip_special_tokens=True
    )

    rewrite_cots = tokenizer.batch_decode(
        data.batch["responses"],
        skip_special_tokens=True
    )

    # 获取额外信息
    answers = []
    original_cots = []
    questions = []

    for i in range(batch_size):
        reward_info = data.non_tensor_batch.get("reward_model", {})
        if isinstance(reward_info, list):
            reward_info = reward_info[i] if i < len(reward_info) else {}
        answer = reward_info.get("ground_truth", "")
        answers.append(answer)

        orig_cot = data.non_tensor_batch.get("original_cot", [""])[i] \
            if "original_cot" in data.non_tensor_batch else ""
        original_cots.append(orig_cot)
        questions.append(prompts[i])

    # 计算认知奖励
    cognitive_computer = get_cognitive_reward_computer(student_model_path)
    cognitive_rewards = cognitive_computer(
        questions=questions,
        rewrite_cots=rewrite_cots,
        answers=answers,
    )

    # 计算逻辑奖励 (可选)
    if use_fsf_reward and len(original_cots[0]) > 0:
        fsf_analyzer = get_fsf_analyzer(
            judge_type=fsf_judge_type,
            api_url=fsf_api_url,
        )
        logic_rewards = compute_logic_reward(
            original_cots=original_cots,
            rewrite_cots=rewrite_cots,
            fsf_analyzer=fsf_analyzer,
        )
        logic_rewards = torch.tensor(logic_rewards, device=cognitive_rewards.device)
    else:
        logic_rewards = torch.zeros(batch_size, device=cognitive_rewards.device)

    # 组合奖励
    total_rewards = cognitive_rewards + lambda_logic * logic_rewards

    # 处理异常值
    total_rewards = torch.nan_to_num(total_rewards, nan=0.0)
    total_rewards = total_rewards.clamp(-100, 100)

    # 转换为 token-level 奖励 (Outcome Reward)
    response_length = data.batch["responses"].shape[1]
    token_level_rewards = torch.zeros(
        batch_size, response_length,
        device=cognitive_rewards.device,
        dtype=cognitive_rewards.dtype
    )

    response_mask = data.batch.get("response_mask", None)
    if response_mask is not None:
        seq_lengths = response_mask.sum(dim=1).long() - 1
        seq_lengths = seq_lengths.clamp(min=0, max=response_length - 1)
    else:
        seq_lengths = torch.full((batch_size,), response_length - 1, dtype=torch.long)

    for i in range(batch_size):
        token_level_rewards[i, seq_lengths[i]] = total_rewards[i]

    return {
        "reward_tensor": token_level_rewards,
        "reward_extra_info": {
            "cognitive_rewards": cognitive_rewards.cpu().tolist(),
            "logic_rewards": logic_rewards.cpu().tolist(),
            "total_rewards": total_rewards.cpu().tolist(),
        }
    }


def compute_cognitive_reward_only(
    data: DataProto,
    tokenizer: Any,
    student_model_path: str = "./models/Qwen2.5-3B-Instruct",
    **kwargs,
) -> dict[str, Any]:
    """只使用认知奖励的简化版本"""
    return compute_cot_rewriter_reward(
        data=data,
        tokenizer=tokenizer,
        student_model_path=student_model_path,
        use_fsf_reward=False,
        lambda_logic=0.0,
        **kwargs,
    )
