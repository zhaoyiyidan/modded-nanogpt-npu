#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [[ -f source/env.sh ]]; then
  source source/env.sh
fi
export DATA_PATH="${DATA_PATH:-$(cd ../../.. && pwd -P)}"
exec torchrun --standalone --nproc_per_node=16 source/train_gpt_stage2_qkvo_band_packed_embed_full.py
