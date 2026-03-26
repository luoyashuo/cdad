#!/usr/bin/env bash
# Build TokenSkip datasets with different skip ratios using 8-GPU data parallelism.
# Each GPU processes 1/8 of the data in parallel, then shards are merged.
#
# Usage:
#   bash dataset_build/run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/config"
WORLD_SIZE=8

run_config() {
    local config="$1"
    echo "=== Starting: $(basename "${config}") with ${WORLD_SIZE} GPUs ==="

    # Launch one worker per GPU in parallel.
    local pids=()
    for rank in $(seq 0 $((WORLD_SIZE - 1))); do
        CUDA_VISIBLE_DEVICES=${rank} python "${SCRIPT_DIR}/build_dataset.py" \
            --config "${config}" \
            --rank "${rank}" \
            --world-size "${WORLD_SIZE}" &
        pids+=($!)
    done

    # Wait for all workers; abort if any fails.
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            echo "ERROR: worker PID ${pid} failed" >&2
            failed=1
        fi
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "ERROR: one or more workers failed for ${config}" >&2
        exit 1
    fi

    # Merge shards into final output.
    echo "--- Merging shards for $(basename "${config}") ---"
    python "${SCRIPT_DIR}/build_dataset.py" \
        --config "${config}" \
        --world-size "${WORLD_SIZE}" \
        --merge-only

    echo "=== Done: $(basename "${config}") ==="
}

# skip_ratio = 0.25 (25% compressed, 75% original)
run_config "${CONFIG_DIR}/deepmath_plain_30k_tokenskip_mix_r25.yaml"

# skip_ratio = 0.50 (50% compressed, 50% original)
run_config "${CONFIG_DIR}/deepmath_plain_30k_tokenskip_mix_r50.yaml"
