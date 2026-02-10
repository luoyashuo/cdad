#!/usr/bin/env python3
"""
测试脚本：验证 Qwen3-4B 模型能否被 vLLM 自动加载和推理
使用正确的聊天模板格式，显示完整输出，并提供去掉 think 标签的选项
"""

import os
import sys
import re

# 禁用 V1 引擎，使用更稳定的 V0 引擎
os.environ["VLLM_USE_V1"] = "0"

import torch
from transformers import AutoTokenizer

MODEL_PATH = "/data/luoys/modelscope_cache/models/Qwen/Qwen3-4B"


def remove_think_tags(text):
    """去掉 <think>...</think> 标签及其内容"""
    # 使用正则表达式去掉 think 标签及其中内容
    # 匹配 <think> 开头，然后是任意字符（非贪婪），然后是 </think>
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 去掉多余的空白
    cleaned = cleaned.strip()
    return cleaned


def extract_final_answer(text):
    """提取 </think> 之后的内容（最终答案）"""
    match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def check_model_files():
    """检查模型文件是否存在"""
    print("=" * 60)
    print("步骤 1: 检查模型文件")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
        return False
    
    required_files = ["config.json", "tokenizer.json"]
    for fname in required_files:
        fpath = os.path.join(MODEL_PATH, fname)
        if os.path.exists(fpath):
            print(f"✓ 找到 {fname}")
        else:
            print(f"❌ 缺少 {fname}")
            return False
    
    # 检查权重文件
    safetensors_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')]
    if safetensors_files:
        print(f"✓ 找到 {len(safetensors_files)} 个 safetensors 权重文件")
    else:
        print("❌ 未找到权重文件")
        return False
    
    print("✓ 模型文件检查通过\n")
    return True


def test_vllm_import():
    """测试 vLLM 是否能正常导入"""
    print("=" * 60)
    print("步骤 2: 检查 vLLM 导入")
    print("=" * 60)
    
    try:
        import vllm
        print(f"✓ vLLM 版本: {vllm.__version__}")
        print(f"  - V1 引擎: {os.environ.get('VLLM_USE_V1', '默认')}")
        
        from vllm import LLM, SamplingParams
        print("✓ 成功导入 LLM 和 SamplingParams")
        print("✓ vLLM 导入检查通过\n")
        return True
    except ImportError as e:
        print(f"❌ vLLM 导入失败: {e}")
        return False


def test_model_loading():
    """测试模型加载"""
    print("=" * 60)
    print("步骤 3: 测试模型加载")
    print("=" * 60)
    
    try:
        from vllm import LLM
        
        print(f"正在加载模型: {MODEL_PATH}")
        print("这可能需要一些时间，请耐心等待...")
        
        # 使用 V0 引擎配置
        llm = LLM(
            model=MODEL_PATH,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=4096,
            enforce_eager=True,
        )
        
        print(f"✓ 模型加载成功!")
        print(f"  - 模型类型: {type(llm.llm_engine.model_executor.driver_worker.model_runner.model).__name__}")
        print(f"✓ 模型加载检查通过\n")
        return llm
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_inference_without_think(llm):
    """测试推理 - 去掉 think 标签"""
    print("=" * 60)
    print("步骤 4: 测试推理 (去掉 <think> 标签)")
    print("=" * 60)
    
    try:
        from vllm import SamplingParams
        
        # 加载 tokenizer 以使用聊天模板
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"✓ Tokenizer 加载成功")
        
        # 测试对话
        test_cases = [
            {
                "name": "自我介绍",
                "messages": [{"role": "user", "content": "你好，请介绍一下自己。"}]
            },
            {
                "name": "数学问题",
                "messages": [{"role": "user", "content": "1 + 1 = "}]
            },
            {
                "name": "知识问答",
                "messages": [{"role": "user", "content": "法国的首都是什么？"}]
            },
        ]
        
        # 应用聊天模板
        prompts = []
        for case in test_cases:
            prompt = tokenizer.apply_chat_template(
                case["messages"],
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append((case["name"], prompt, case["messages"]))
        
        print(f"\n将使用 {len(prompts)} 个测试用例进行推理...")
        
        # 使用更多 token
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,  # 增加 token 数
        )
        
        # 逐个生成
        for i, (name, prompt, messages) in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"测试用例 {i+1}: {name}")
            print(f"{'='*60}")
            
            # 显示用户输入
            last_user_msg = messages[-1]["content"] if messages[-1]["role"] == "user" else "未知"
            print(f"\n用户输入: {last_user_msg}")
            
            # 生成
            outputs = llm.generate([prompt], sampling_params)
            raw_output = outputs[0].outputs[0].text
            
            # 处理输出：去掉 think 标签
            final_answer = extract_final_answer(raw_output)
            
            print(f"\n原始输出 (含 think):")
            print("-" * 60)
            print(raw_output[:500] + "..." if len(raw_output) > 500 else raw_output)
            print("-" * 60)
            
            print(f"\n处理后输出 (去掉 think):")
            print("-" * 60)
            print(final_answer)
            print("-" * 60)
            
            # 显示处理信息
            if "<think>" in raw_output:
                print(f"✓ 已去掉 think 标签")
                print(f"  - 原始长度: {len(raw_output)} 字符")
                print(f"  - 处理后长度: {len(final_answer)} 字符")
        
        print("\n✓ 推理测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("Qwen3-4B vLLM 加载与推理测试")
    print("(去掉 <think> 标签版本)")
    print("=" * 60 + "\n")
    
    # 检查 CUDA 可用性
    if torch.cuda.is_available():
        print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
    else:
        print("⚠️ 警告: CUDA 不可用，测试可能使用 CPU（会非常慢）")
    
    print()
    
    # 执行测试步骤
    if not check_model_files():
        sys.exit(1)
    
    if not test_vllm_import():
        sys.exit(1)
    
    llm = test_model_loading()
    if llm is None:
        sys.exit(1)
    
    # 推理测试（去掉 think 标签）
    if not test_inference_without_think(llm):
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过!")
    print("结论: Qwen3-4B 可以被 vLLM 正常加载和推理")
    print("方法: 使用正则表达式提取 </think> 之后的内容")
    print("=" * 60)


if __name__ == "__main__":
    main()
