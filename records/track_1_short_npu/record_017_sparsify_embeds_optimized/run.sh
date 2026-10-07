#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f source/env.sh ]]; then
  # Optional, record-specific runtime configuration.
  source source/env.sh
fi
export DATA_PATH="${DATA_PATH:-$(cd ../../.. && pwd -P)}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec torchrun --standalone --nproc_per_node=16 source/train_gpt.py
