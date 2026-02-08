#!/usr/bin/env python3
"""
简单测试 SGLang 的显存释放/恢复功能
模拟 Ray actor 中的使用场景

修复: 在调用 release/resume 前确保 event loop 存在
"""

import gc
import subprocess
import time
import asyncio


def ensure_event_loop():
    """确保当前线程有一个可用的 event loop"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            # 如果已关闭，创建新的
            asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError:
        # 没有 event loop，创建新的
        asyncio.set_event_loop(asyncio.new_event_loop())


def get_gpu_memory(gpu_id=0):
    """获取指定 GPU 的显存使用情况（MB）"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits',
             f'--id={gpu_id}'],
            capture_output=True, text=True, timeout=5
        )
        used_mb, total_mb = map(int, result.stdout.strip().split(','))
        return used_mb, total_mb
    except Exception as e:
        print(f"[ERROR] nvidia-smi failed: {e}")
        return 0, 0


def format_memory(mb):
    if mb < 1024:
        return f"{mb}MB"
    return f"{mb/1024:.2f}GB"


def main():
    print("=" * 60)
    print("SGLang 显存释放/恢复测试（简化版）")
    print("=" * 60)

    gpu_id = 0
    used_before, total_mb = get_gpu_memory(gpu_id)
    print(f"\n[初始状态] GPU {gpu_id}: {format_memory(used_before)} / {format_memory(total_mb)}")

    # 测试 1: 查看 SGLang Engine 是否有 release/resume 方法
    print("\n[测试 1] 检查 SGLang Engine 的显存管理方法...")
    import sglang as sgl
    from sglang.srt.entrypoints.engine import Engine as EngineClass

    engine_methods = dir(EngineClass)
    memory_methods = [m for m in engine_methods if 'memory' in m.lower() or 'release' in m.lower() or 'resume' in m.lower()]

    print(f"  找到的显存相关方法: {memory_methods}")

    if 'release_memory_occupation' in engine_methods and 'resume_memory_occupation' in engine_methods:
        print("  ✓ SGLang 支持 release_memory_occupation() 和 resume_memory_occupation()")
    else:
        print("  ✗ SGLang 不支持显存释放/恢复方法")
        return False

    # 测试 2: 创建 Engine 并测试显存变化
    print("\n[测试 2] 创建 Engine 并测试显存...")

    # SGLang 使用 uvloop
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass

    model_path = "/root/luoys/modelscope_cache/models/Qwen/Qwen3-1.7B"

    # 创建 Engine（启用 enable_memory_saver）
    print(f"  创建 Engine: {model_path}")
    print(f"  enable_memory_saver=True, mem_fraction_static=0.4")

    engine = sgl.Engine(
        model_path=model_path,
        tp_size=1,
        mem_fraction_static=0.4,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_memory_saver=True,  # 关键配置
        log_level="error",
    )

    time.sleep(2)
    used_after_load, _ = get_gpu_memory(gpu_id)
    model_memory = used_after_load - used_before
    print(f"  Engine 已创建，显存: {format_memory(used_after_load)} (模型: {format_memory(model_memory)})")

    # 测试 3: 调用 release_memory_occupation
    print("\n[测试 3] 测试 release_memory_occupation()...")

    try:
        # 关键修复: 确保 event loop 存在
        ensure_event_loop()
        engine.release_memory_occupation()
        print("  ✓ release_memory_occupation() 调用成功")
        time.sleep(2)

        used_after_release, _ = get_gpu_memory(gpu_id)
        released = used_after_load - used_after_release
        print(f"  显存: {format_memory(used_after_release)} (释放: {format_memory(released)})")

        if released > 100:
            print(f"  ✓✓ 显存释放有效！释放了 {format_memory(released)}")
        else:
            print(f"  ⚠ 释放量较小，可能需要 torch.cuda.empty_cache()")

    except Exception as e:
        print(f"  ✗ release_memory_occupation() 失败: {e}")
        return False

    # 清空 CUDA 缓存
    import torch
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    used_after_cache, _ = get_gpu_memory(gpu_id)
    print(f"  清空 CUDA cache 后: {format_memory(used_after_cache)}")

    # 测试 4: 调用 resume_memory_occupation
    print("\n[测试 4] 测试 resume_memory_occupation()...")

    try:
        # 关键修复: 确保 event loop 存在
        ensure_event_loop()
        engine.resume_memory_occupation()
        print("  ✓ resume_memory_occupation() 调用成功")
        time.sleep(2)

        used_after_resume, _ = get_gpu_memory(gpu_id)
        print(f"  显存: {format_memory(used_after_resume)}")

    except Exception as e:
        print(f"  ✗ resume_memory_occupation() 失败: {e}")
        return False

    # 清理
    print("\n[清理] 关闭 Engine...")
    engine.shutdown()
    del engine
    torch.cuda.empty_cache()
    gc.collect()

    # 总结
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"  模型加载显存: {format_memory(model_memory)}")
    print(f"  release 后显存: {format_memory(used_after_cache)}")
    print(f"  释放量: {format_memory(used_after_load - used_after_cache)}")
    print(f"  resume 后显存: {format_memory(used_after_resume)}")

    if used_after_load - used_after_cache > 100:
        print("\n  ✓ SGLang 的 release/resume 功能可用！")
        print("  建议：在代码中使用 enable_memory_saver=True")
        return True
    else:
        print("\n  ⚠ release 效果不明显，可能需要其他方式")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试中断")
        exit(130)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
