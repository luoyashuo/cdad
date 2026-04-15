#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT="${INPUT:-/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/fsf_10k_splits}"
MODEL_PATH="${MODEL_PATH:-/data/hf_home/models/Yaseal-qwen3_4b_fsf_grm}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TARGET_SIZE="${TARGET_SIZE:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TP_SIZE="${TP_SIZE:-1}"
SELECTION="${SELECTION:-random}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/build_fsf_datasets.py" \
  --phase all \
  --input "${INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --gpu-ids "${GPU_IDS}" \
  --batch-size "${BATCH_SIZE}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --target-size "${TARGET_SIZE}" \
  --selection "${SELECTION}" \
  --resume
