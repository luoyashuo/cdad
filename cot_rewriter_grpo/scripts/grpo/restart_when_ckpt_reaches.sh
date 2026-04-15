#!/bin/bash

set -euo pipefail

TARGET_STEP="${1:?usage: $0 <target_step> <ckpt_dir> <start_script> [poll_seconds] [log_file]}"
CKPT_DIR="${2:?usage: $0 <target_step> <ckpt_dir> <start_script> [poll_seconds] [log_file]}"
START_SCRIPT="${3:?usage: $0 <target_step> <ckpt_dir> <start_script> [poll_seconds] [log_file]}"
POLL_SECONDS="${4:-20}"
LOG_FILE="${5:-${CKPT_DIR}/restart_watcher.log}"
TRACKER_FILE="${CKPT_DIR}/latest_checkpointed_iteration.txt"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*" | tee -a "${LOG_FILE}"
}

read_step() {
    if [ -f "${TRACKER_FILE}" ]; then
        tr -dc '0-9' < "${TRACKER_FILE}"
    fi
}

log "Watcher started: target_step=${TARGET_STEP}, ckpt_dir=${CKPT_DIR}, poll_seconds=${POLL_SECONDS}"

while true; do
    current_step="$(read_step)"
    if [ -z "${current_step}" ]; then
        log "Tracker file not ready: ${TRACKER_FILE}"
    else
        log "Observed latest checkpoint step=${current_step}"
        if [ "${current_step}" -ge "${TARGET_STEP}" ]; then
            log "Target reached. Restarting training via ${START_SCRIPT}"
            bash "${START_SCRIPT}" >> "${LOG_FILE}" 2>&1
            log "Restart command finished. Exiting watcher."
            exit 0
        fi
    fi
    sleep "${POLL_SECONDS}"
done
