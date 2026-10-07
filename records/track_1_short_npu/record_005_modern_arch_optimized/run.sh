#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f source/env.sh ]]; then
  source source/env.sh
fi
export DATA_PATH="${DATA_PATH:-$(cd ../../.. && pwd -P)}"
export PYTHONUNBUFFERED=1
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1

torchrun --standalone --nproc_per_node=16 source/train_gpt.py
