#!/bin/bash
# vLLM快速修复脚本 - 自动重新安装vLLM以匹配当前PyTorch版本

set -e

echo "============================================================"
echo "vLLM快速修复（自动重新编译安装）"
echo "============================================================"

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$PROJECT_ROOT"

# 激活虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "错误：未找到虚拟环境"
    exit 1
fi

echo ""
echo "当前环境："
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA: {torch.version.cuda}')"

echo ""
echo "开始修复vLLM..."
echo "  1. 卸载旧版本"
pip uninstall -y vllm 2>/dev/null || true

echo "  2. 清理缓存"
pip cache purge 2>/dev/null || true

echo "  3. 重新安装vLLM 0.10.2（从源码编译，约需3-5分钟）"
echo "     请耐心等待..."
# 使用--no-binary强制从源码编译，确保与当前PyTorch版本兼容
CUDA_HOME=/usr/local/cuda pip install --no-binary vllm vllm==0.10.2 --force-reinstall --no-cache-dir

echo ""
echo "验证安装..."
if python -c "import vllm; print(f'✓ vLLM {vllm.__version__} 安装成功！')" 2>/dev/null; then
    echo ""
    echo "============================================================"
    echo "修复完成！现在可以重新运行训练任务"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "vLLM导入失败，尝试备用方案..."
    echo "============================================================"
    echo ""
    echo "使用预编译wheel重新安装..."
    pip uninstall -y vllm 2>/dev/null || true
    pip install vllm==0.10.2 --force-reinstall

    if python -c "import vllm" 2>/dev/null; then
        echo "✓ 使用预编译版本安装成功！"
    else
        echo "✗ 安装失败，请检查环境配置"
        exit 1
    fi
fi
