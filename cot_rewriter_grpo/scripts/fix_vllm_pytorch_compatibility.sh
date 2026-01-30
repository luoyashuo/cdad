#!/bin/bash
# 修复 vLLM 与 PyTorch 兼容性问题
#
# 问题描述：
# vLLM的C++扩展与当前的PyTorch版本不兼容，导致符号未定义错误
# ImportError: undefined symbol: _ZN3c106ivalue14ConstantString6createE...
#
# 解决方案：
# 重新安装与PyTorch 2.6.0+cu124兼容的vLLM版本

set -e

echo "============================================================"
echo "vLLM与PyTorch兼容性修复脚本"
echo "============================================================"

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$PROJECT_ROOT"

# 激活虚拟环境
if [ -d ".venv" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
else
    echo "错误：未找到虚拟环境 (.venv)"
    exit 1
fi

# 检查当前PyTorch版本
echo ""
echo "检查当前PyTorch版本..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA版本: {torch.version.cuda}')"

echo ""
echo "============================================================"
echo "开始修复vLLM..."
echo "============================================================"

# 方案1：重新安装vLLM（推荐）
echo ""
echo "[方案1] 卸载并重新安装vLLM（推荐）"
echo "这将确保vLLM与当前PyTorch版本完全兼容"
read -p "是否执行？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "卸载旧的vLLM..."
    pip uninstall -y vllm || true

    echo "清理缓存..."
    pip cache purge || true
    rm -rf ~/.cache/pip/wheels/vllm* || true

    echo "重新安装vLLM 0.10.2（从源码编译，确保与当前PyTorch兼容）..."
    # 使用--no-binary强制从源码编译，确保与当前PyTorch版本兼容
    pip install --no-binary vllm vllm==0.10.2 --force-reinstall

    echo ""
    echo "验证vLLM安装..."
    python -c "import vllm; print(f'vLLM版本: {vllm.__version__}')" && echo "✓ vLLM安装成功！" || echo "✗ vLLM安装失败"

    exit 0
fi

# 方案2：使用预编译wheel（如果源码编译失败）
echo ""
echo "[方案2] 尝试使用预编译wheel"
echo "如果方案1编译时间过长或失败，可以尝试此方案"
read -p "是否执行？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "卸载旧的vLLM..."
    pip uninstall -y vllm || true

    echo "安装vLLM 0.10.2（使用预编译wheel）..."
    pip install vllm==0.10.2 --force-reinstall

    echo ""
    echo "验证vLLM安装..."
    python -c "import vllm; print(f'vLLM版本: {vllm.__version__}')" && echo "✓ vLLM安装成功！" || echo "✗ vLLM安装失败"

    exit 0
fi

# 方案3：降级PyTorch（不推荐，仅当其他方案都失败时使用）
echo ""
echo "[方案3] 降级PyTorch到2.5.1（不推荐）"
echo "警告：这可能影响其他依赖"
read -p "是否执行？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "安装PyTorch 2.5.1+cu124..."
    pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

    echo "重新安装vLLM..."
    pip uninstall -y vllm || true
    pip install vllm==0.10.2

    echo ""
    echo "验证安装..."
    python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
    python -c "import vllm; print(f'vLLM版本: {vllm.__version__}')"

    exit 0
fi

echo ""
echo "未执行任何修复操作"
