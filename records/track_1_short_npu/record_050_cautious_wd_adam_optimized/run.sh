#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source source/env.sh
exec torchrun --standalone --nproc_per_node=16 source/train_gpy.py
