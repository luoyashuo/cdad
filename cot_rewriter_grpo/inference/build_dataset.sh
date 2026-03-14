#!/usr/bin/env bash
# Build ShareGPT Dataset from Pipeline Output (Step 5)
#
# Usage:
#   bash inference/build_dataset.sh inference/config_example.yaml
#   bash inference/build_dataset.sh inference/config_example.yaml --strategy priority_correct
#   bash inference/build_dataset.sh inference/config_example.yaml \
#       --filter-fsf-positive --filter-correct-only
#   bash inference/build_dataset.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate virtual environment
VENV_PATH="${REPO_ROOT}/.venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "WARNING: Virtual environment not found at ${VENV_PATH}" >&2
    echo "         Continuing with current Python environment." >&2
fi

# Add verl to PYTHONPATH
export PYTHONPATH="${REPO_ROOT}/verl:${REPO_ROOT}:${PYTHONPATH:-}"

exec python -m inference.build_sharegpt "$@"
