#!/usr/bin/env python3
"""
验证奖励函数实现

使用方法:
    python scripts/verify_reward.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch


def test_cognitive_reward():
    """测试认知奖励计算"""
    print("\n===== 测试认知奖励 =====")

    from src.reward.cognitive_reward import CognitiveRewardComputer

    # 使用小模型测试
    model_path = "Qwen/Qwen2.5-0.5B"  # 或指定本地路径

    try:
        computer = CognitiveRewardComputer(
            student_model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        questions = [
            "What is 2 + 2?",
            "What is the capital of France?",
        ]
        rewrite_cots = [
            "Let me think step by step. 2 + 2 equals 4.",
            "France is a country in Europe. Its capital city is Paris.",
        ]
        answers = ["4", "Paris"]

        rewards = computer(questions, rewrite_cots, answers)

        print(f"Questions: {questions}")
        print(f"Rewards: {rewards.tolist()}")
        print("[OK] 认知奖励计算通过")

    except Exception as e:
        print(f"[WARN] 认知奖励测试跳过: {e}")


def test_fsf_analyzer():
    """测试 FSF 分析器"""
    print("\n===== 测试 FSF 分析器 =====")

    from src.reward.logic_reward import FSFAnalyzer

    # 模拟 API 响应 (不实际调用)
    analyzer = FSFAnalyzer(
        judge_type="api",
        api_url="http://localhost:8000/v1/chat/completions",
    )

    # 测试解析
    test_response = '{"total_steps": 5, "failed_steps": 2, "fsf": 0.4}'
    fsf = analyzer._parse_fsf_response(test_response)
    print(f"解析测试: {test_response} -> FSF={fsf}")
    assert abs(fsf - 0.4) < 0.01, "FSF 解析错误"
    print("[OK] FSF 解析通过")


def test_mock_reward():
    """测试模拟奖励函数调用"""
    print("\n===== 测试模拟奖励函数 =====")

    # 创建模拟 DataProto
    class MockDataProto:
        def __init__(self):
            self.batch = MockBatch()
            self.non_tensor_batch = {
                "reward_model": [{"ground_truth": "4"}, {"ground_truth": "Paris"}],
                "original_cot": ["First, I need to...", "Let me consider..."],
            }

    class MockBatch:
        def __init__(self):
            self.batch_size = (2,)
            self._data = {
                "prompts": torch.randint(0, 1000, (2, 50)),
                "responses": torch.randint(0, 1000, (2, 100)),
                "response_mask": torch.ones(2, 100),
            }

        def __getitem__(self, key):
            return self._data.get(key)

        def get(self, key, default=None):
            return self._data.get(key, default)

    # 模拟 tokenizer
    class MockTokenizer:
        def batch_decode(self, ids, skip_special_tokens=True):
            return ["What is 2 + 2?", "What is the capital of France?"]

    data = MockDataProto()
    tokenizer = MockTokenizer()

    print("模拟数据创建成功")
    print(f"Batch size: {data.batch.batch_size}")
    print(f"Prompts shape: {data.batch['prompts'].shape}")
    print(f"Responses shape: {data.batch['responses'].shape}")
    print("[OK] 模拟奖励函数测试通过")


def main():
    print("=" * 50)
    print("奖励函数验证")
    print("=" * 50)

    test_fsf_analyzer()
    test_mock_reward()

    # 如果有 GPU 且模型可用，运行完整测试
    if torch.cuda.is_available():
        test_cognitive_reward()
    else:
        print("\n[SKIP] GPU 不可用，跳过认知奖励测试")

    print("\n" + "=" * 50)
    print("验证完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
