#!/usr/bin/env bash
#
# Build FSF v2 datasets from existing scored_records.jsonl.
# V2 difference: zero and positive samples are drawn from completely
# disjoint question pools (no question binding).
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/luoys/cdad/.venv/bin/python}"

SCORED_INPUT="${SCORED_INPUT:-/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/fsf_10k_splits/scored_records.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/fsf_10k_splits_v2}"
TARGET_SIZE="${TARGET_SIZE:-10000}"
SELECTION="${SELECTION:-random}"
SEED="${SEED:-42}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/build_fsf_datasets.py" \
  --phase build \
  --scored-input "${SCORED_INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --target-size "${TARGET_SIZE}" \
  --selection "${SELECTION}" \
  --seed "${SEED}" \
  --no-question-binding
