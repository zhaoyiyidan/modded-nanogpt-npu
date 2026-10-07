#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f source/env.sh ]]; then
  source source/env.sh
fi
export DATA_PATH="${DATA_PATH:-$(cd ../../.. && pwd -P)}"
export INPUT_BIN="${INPUT_BIN:-$DATA_PATH/data/fineweb10B/fineweb_train_*.bin}"
export INPUT_VAL_BIN="${INPUT_VAL_BIN:-$DATA_PATH/data/fineweb10B/fineweb_val_*.bin}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export NANOGPT_SEED="${NANOGPT_SEED:-0}"
export DDP_BUCKET_MB="${DDP_BUCKET_MB:-50}"
export DDP_GRADIENT_AS_BUCKET_VIEW="${DDP_GRADIENT_AS_BUCKET_VIEW:-1}"
export DDP_STATIC_GRAPH="${DDP_STATIC_GRAPH:-1}"
export DDP_INIT_SYNC="${DDP_INIT_SYNC:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/logs/output}"
export RUN_DIR="${RUN_DIR:-$PWD/logs/runs/$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$OUTPUT_DIR" "$RUN_DIR"
torchrun --standalone --nproc_per_node=16 source/train_gpt.py
