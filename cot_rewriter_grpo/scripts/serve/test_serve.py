#!/usr/bin/env python3
"""测试 serve_judge_model.sh 启动的模型服务"""

import sys
import requests
import json


def test_judge_model_serve():
    """测试 judge 模型服务是否正常响应"""

    # 服务地址
    base_url = "http://localhost:8000"

    # 测试列表模型接口
    try:
        response = requests.get(f"{base_url}/v1/models")
        response.raise_for_status()
        models = response.json()
        print("✅ 服务状态正常")
        print(f"📦 已加载的模型: {models}")
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务，请确保服务已启动")
        return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False

    # 测试聊天接口（包含大上下文和大输出）
    test_messages = [
        {
            "role": "user",
            "content": "你好，请写一篇关于人工智能发展历史的详细介绍，要求字数在20000字左右，涵盖从早期概念到现代深度学习的整个历程，包括重要的人物、事件和技术突破。",
        }
    ]

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "/data-share/yeesuanAI08/lihaitao/luoys/hf_home/models/openai/gpt-oss-20b",
                "messages": test_messages,
                "temperature": 0.7,
                "max_tokens": 20000,
            },
        )
        response.raise_for_status()
        result = response.json()
        print("\n✅ 聊天接口正常")

        # 输出回复的元数据
        print(f"📊 回复元数据:")
        print(f"   - 生成的 token 数: {result['usage']['completion_tokens']}")
        print(f"   - 总 token 数: {result['usage']['total_tokens']}")

        # 输出回复内容的前 500 个字符
        content = result["choices"][0]["message"]["content"]
        print(f"💬 回复内容前 500 字符:\n{content[:500]}...")

        return True
    except Exception as e:
        print(f"\n❌ 聊天接口测试失败: {e}")
        return False


if __name__ == "__main__":
    print("开始测试 judge 模型服务...")
    print("=" * 50)

    success = test_judge_model_serve()

    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过，服务正常运行！")
        sys.exit(0)
    else:
        print("❌ 测试失败，服务可能未正确启动")
        sys.exit(1)
