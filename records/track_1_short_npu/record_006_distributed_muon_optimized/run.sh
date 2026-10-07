#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [ -f source/env.sh ]; then source source/env.sh; fi
export DATA_ROOT="${DATA_ROOT:-$(cd ../../.. && pwd -P)/data}"
export METRICS_FILE="${METRICS_FILE:-logs/metrics.json}"
torchrun --standalone --nproc_per_node=16 source/train_gpt.py
