#!/usr/bin/env python3
"""
GPU清理工具 - 用于检查和清理卡死的GPU进程
当nvidia-smi卡住或GPU被僵尸进程占用时使用
"""

import subprocess
import sys
import time


def check_nvidia_smi():
    """检查nvidia-smi是否响应"""
    print("检查 nvidia-smi 响应...")
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ nvidia-smi 正常响应")
            return True
        else:
            print(f"✗ nvidia-smi 返回错误: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ nvidia-smi 超时无响应 - GPU驱动可能已卡死")
        return False
    except Exception as e:
        print(f"✗ 执行nvidia-smi时出错: {str(e)}")
        return False


def list_gpu_processes():
    """列出所有使用GPU的进程"""
    print("\n当前使用GPU的进程:")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(result.stdout)
            return True
    except Exception as e:
        print(f"获取GPU进程失败: {str(e)}")
    return False


def kill_gpu_processes():
    """杀死所有使用GPU的计算进程"""
    print("\n正在终止GPU进程...")
    try:
        # 获取所有GPU计算进程
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            pids = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
            
            if not pids:
                print("没有发现GPU进程")
                return
            
            for pid in pids:
                try:
                    # 获取进程信息
                    proc_result = subprocess.run(
                        ["ps", "-fp", str(pid)],
                        capture_output=True,
                        text=True
                    )
                    print(f"\n终止进程 PID={pid}:")
                    print(proc_result.stdout)
                    
                    # 杀死进程
                    subprocess.run(["kill", "-9", str(pid)], check=False)
                    print(f"  ✓ 已终止")
                except Exception as e:
                    print(f"  ✗ 终止失败: {str(e)}")
        
        time.sleep(2)
        print("\n清理完成")
        
    except Exception as e:
        print(f"清理过程中出错: {str(e)}")


def kill_vllm_processes():
    """专门杀死vLLM相关进程"""
    print("\n查找并终止vLLM相关进程...")
    try:
        # 查找vLLM进程
        result = subprocess.run(
            ["pgrep", "-f", "vllm.*api_server"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                pid = pid.strip()
                if pid:
                    print(f"终止vLLM进程 PID={pid}")
                    subprocess.run(["kill", "-9", pid], check=False)
        else:
            print("没有发现vLLM进程")
            
        # 查找VllmWorker进程
        result = subprocess.run(
            ["pgrep", "-f", "VllmWorker"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                pid = pid.strip()
                if pid:
                    print(f"终止VllmWorker进程 PID={pid}")
                    subprocess.run(["kill", "-9", pid], check=False)
        
        time.sleep(2)
        print("vLLM进程清理完成")
        
    except Exception as e:
        print(f"清理vLLM进程时出错: {str(e)}")


def main():
    print("=" * 60)
    print("GPU 清理工具")
    print("=" * 60)
    
    # 检查nvidia-smi
    if check_nvidia_smi():
        list_gpu_processes()
    
    print("\n选项:")
    print("1. 杀死所有GPU计算进程")
    print("2. 杀死vLLM相关进程")
    print("3. 执行完整清理（GPU进程 + vLLM进程）")
    print("4. 仅检查，不执行清理")
    print("q. 退出")
    
    choice = input("\n请选择 [1/2/3/4/q]: ").strip().lower()
    
    if choice == '1':
        kill_gpu_processes()
    elif choice == '2':
        kill_vllm_processes()
    elif choice == '3':
        kill_gpu_processes()
        kill_vllm_processes()
        print("\n重新检查GPU状态...")
        check_nvidia_smi()
    elif choice == '4':
        check_nvidia_smi()
    else:
        print("退出")
        return
    
    print("\n完成。如果GPU仍然无响应，可能需要重启机器。")


if __name__ == "__main__":
    main()
