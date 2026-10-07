#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f source/env.sh ]]; then
  source source/env.sh
fi
export DATA_PATH="${DATA_PATH:-$(cd ../../.. && pwd -P)}"
export LOG_DIR="${LOG_DIR:-logs/run_$(date +%Y%m%d_%H%M%S_%N)}"

torchrun --standalone --nproc_per_node=16 source/train_gpt.py
