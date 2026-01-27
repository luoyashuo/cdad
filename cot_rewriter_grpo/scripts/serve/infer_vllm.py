#!/usr/bin/env python3
"""持续请求 judge 模型服务的脚本"""

import sys
import requests
import json
import time
import random


def continuous_request_service(interval=5, max_requests=None):
    """
    持续请求 judge 模型服务

    Args:
        interval: 请求间隔（秒）
        max_requests: 最大请求次数（None 表示无限循环）
    """

    # 服务地址
    base_url = "http://localhost:8000"
    model_path = (
        "/data-share/yeesuanAI08/lihaitao/luoys/hf_home/models/openai/gpt-oss-20b"
    )

    # 测试消息列表（用于轮流发送不同的请求）
    test_messages_list = [
        [
            {
                "role": "user",
                "content": "你好，请写一篇关于人工智能发展历史的详细介绍，要求字数在20000字左右，涵盖从早期概念到现代深度学习的整个历程，包括重要的人物、事件和技术突破。",
            }
        ],
        [
            {
                "role": "user",
                "content": "请详细解释什么是机器学习，包括其基本原理、主要算法类别和应用场景。",
            }
        ],
        [
            {
                "role": "user",
                "content": "什么是深度学习？它与传统机器学习有什么不同？请详细说明。",
            }
        ],
        [
            {
                "role": "user",
                "content": "请介绍自然语言处理的发展历程和当前主要技术方向。",
            }
        ],
        [
            {
                "role": "user",
                "content": "解释什么是计算机视觉，以及它在现实生活中的应用。",
            }
        ],
    ]

    request_count = 0
    success_count = 0
    failure_count = 0

    print("开始持续请求 judge 模型服务...")
    print("=" * 50)

    try:
        while True:
            # 检查是否达到最大请求次数
            if max_requests is not None and request_count >= max_requests:
                break

            request_count += 1
            print(f"\n🔄 请求 {request_count}: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            # 随机选择一个测试消息
            test_messages = random.choice(test_messages_list)

            try:
                # 发送聊天请求
                response = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "model": model_path,
                        "messages": test_messages,
                        "temperature": 0.7,
                        "max_tokens": 20000,
                    },
                    timeout=300,  # 增加超时时间，因为请求可能需要较长时间
                )
                response.raise_for_status()
                result = response.json()

                # 输出回复的元数据
                print(f"✅ 请求成功")
                print(f"   - 生成的 token 数: {result['usage']['completion_tokens']}")
                print(f"   - 总 token 数: {result['usage']['total_tokens']}")

                # 输出回复内容的前 200 个字符
                content = result["choices"][0]["message"]["content"]
                print(f"   - 回复内容前 200 字符:\n{content[:200]}...")

                success_count += 1

            except requests.exceptions.ConnectionError:
                print("❌ 无法连接到服务，请确保服务已启动")
                failure_count += 1
            except requests.exceptions.Timeout:
                print("❌ 请求超时")
                failure_count += 1
            except Exception as e:
                print(f"❌ 请求失败: {e}")
                failure_count += 1

            # 等待指定间隔
            if interval > 0:
                print(f"\n⏳ 等待 {interval} 秒后发送下一个请求...")
                time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n🛑 用户中断程序")

    finally:
        # 输出统计信息
        print("\n" + "=" * 50)
        print("📊 请求统计:")
        print(f"   - 总请求数: {request_count}")
        print(f"   - 成功数: {success_count}")
        print(f"   - 失败数: {failure_count}")
        if request_count > 0:
            print(f"   - 成功率: {success_count / request_count * 100:.2f}%")


if __name__ == "__main__":
    # 解析命令行参数
    import argparse

    parser = argparse.ArgumentParser(description="持续请求 judge 模型服务的脚本")
    parser.add_argument(
        "-i", "--interval", type=int, default=5, help="请求间隔（秒，默认: 5）"
    )
    parser.add_argument(
        "-m",
        "--max-requests",
        type=int,
        default=None,
        help="最大请求次数（默认: 无限循环）",
    )
    parser.add_argument(
        "-t", "--timeout", type=int, default=300, help="请求超时时间（秒，默认: 300）"
    )

    args = parser.parse_args()

    continuous_request_service(
        interval=args.interval,
        max_requests=args.max_requests,
    )
