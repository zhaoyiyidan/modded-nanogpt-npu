#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
if [[ -f source/env.sh ]]; then
    source source/env.sh
fi
export DATA_PATH="${DATA_PATH:-$(cd ../../.. && pwd -P)}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-240}"
exec torchrun --standalone --nproc_per_node=16 source/train_gpt.py
