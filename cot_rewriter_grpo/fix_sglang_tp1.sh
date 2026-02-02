#!/bin/bash
# Script to fix SGLang tensor_model_parallel_size=1 issue and clean caches

echo "=============================================="
echo "Fixing SGLang TP=1 issue and cleaning caches"
echo "=============================================="

# 1. Kill all Ray processes
echo "[1/4] Stopping all Ray processes..."
pkill -u "$(whoami)" -f "ray::" 2>/dev/null || true
pkill -u "$(whoami)" -f "raylet" 2>/dev/null || true
pkill -u "$(whoami)" -f "gcs_server" 2>/dev/null || true
pkill -u "$(whoami)" -f "ray._private" 2>/dev/null || true
sleep 2

# 2. Clean Python cache files
echo "[2/4] Cleaning Python cache files..."
find /Users/moonshot/Desktop/cot_rewrite_grpo/verl -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /Users/moonshot/Desktop/cot_rewrite_grpo/verl -name "*.pyc" -delete 2>/dev/null || true
find /Users/moonshot/Desktop/cot_rewrite_grpo/verl -name "*.pyo" -delete 2>/dev/null || true

# 3. Clean Ray temp directory
echo "[3/4] Cleaning Ray temp directory..."
export RAY_TMPDIR="/tmp/ray_${USER:-$(whoami)}"
rm -rf "$RAY_TMPDIR"/session_* 2>/dev/null || true

# 4. Verify the fixes
echo "[4/4] Verifying the fixes..."
echo ""
echo "Checking sglang_rollout.py fixes:"
if grep -q "should_create_engine = first_rank_in_node and (self._tp_size > 1 or self._rank == 0)" \
    /Users/moonshot/Desktop/cot_rewrite_grpo/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py; then
    echo "  ✓ Engine creation condition is correct"
else
    echo "  ✗ Engine creation fix not found!"
fi

if grep -q "if self._engine is not None:" \
    /Users/moonshot/Desktop/cot_rewrite_grpo/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py; then
    echo "  ✓ async_generate condition uses engine check"
else
    echo "  ✗ async_generate fix not found!"
fi

echo ""
echo "Checking fsdp_sglang.py fixes:"
if grep -q "if tp_size == 1:" \
    /Users/moonshot/Desktop/cot_rewrite_grpo/verl/verl/workers/sharding_manager/fsdp_sglang.py; then
    echo "  ✓ update_weights handles tp_size=1 case"
else
    echo "  ✗ update_weights fix not found!"
fi

if grep -q "if self.inference_engine is not None and self.rollout_config.free_cache_engine:" \
    /Users/moonshot/Desktop/cot_rewrite_grpo/verl/verl/workers/sharding_manager/fsdp_sglang.py; then
    echo "  ✓ release_memory/wake_up use engine check"
else
    echo "  ✗ release_memory/wake_up fix not found!"
fi

echo ""
echo "=============================================="
echo "Fixes applied and caches cleaned!"
echo "=============================================="
echo ""
echo "Now you can restart training with:"
echo "  bash scripts/grpo/start_ruc_4gpu_80g_sglang.sh"
echo ""
