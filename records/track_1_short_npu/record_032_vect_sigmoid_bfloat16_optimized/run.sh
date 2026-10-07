#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f source/env.sh ]]; then
  # Optional site-specific environment overrides; not required by the benchmark image.
  source source/env.sh
fi
export DATA_PATH="${DATA_PATH:-$(cd ../../.. && pwd -P)}"

exec torchrun --standalone --nproc_per_node=16 \
  source/train_gpt_softcap_tuned_relu2_bf16ce_full.py "$@"
