#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f source/env.sh ]]; then source source/env.sh; fi
export DATA_PATH="${DATA_PATH:-$(cd ../../.. && pwd -P)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec torchrun --standalone --nproc_per_node=16 source/train_gpt.py \
  --input_bin "$DATA_PATH/data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "$DATA_PATH/data/fineweb10B/fineweb_val_*.bin" \
  --output_dir "${OUTPUT_DIR:-$PWD/logs/internal}" \
  --batch_size 32 --num_iterations 9536 \
  --learning_rate 0.0036 --warmup_iters 512 --warmdown_iters 4096 \
  --weight_decay 0.0 --val_loss_every 9536 \
  --save_every 20000
