#!/usr/bin/env bash
set -e
record_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$record_dir/source/env.sh"
set -uo pipefail
cd "$record_dir"

run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
if [[ ! "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo 'RUN_ID must contain only letters, digits, underscores or hyphens' >&2
    exit 2
fi
main_log="$record_dir/logs/verify_${run_id}.log"
metrics="$record_dir/logs/metrics_${run_id}.json"
if [[ -e "$main_log" || -e "$metrics" ]]; then
    echo "Existing output for $run_id; choose a new RUN_ID" >&2
    exit 2
fi
export DATA_ROOT='/SharePath/chenyupeng/cyp-testtest/modded-nanogpt-npu-master/data'
export METRICS_FILE="$metrics"

{
    printf 'RUN_ID=%s\nSOURCE_SHA256=%s\n' "$run_id" "$(sha256sum source/train_gpt.py | cut -d' ' -f1)"
    printf 'DATA_ROOT=%s\nMETRICS_FILE=%s\n' "$DATA_ROOT" "$METRICS_FILE"
    printf 'COMMAND=torchrun --standalone --nproc_per_node=16 source/train_gpt.py\n'
    printf 'START_UTC=%s\n' "$(date -u +%FT%TZ)"
    set +e
    torchrun --standalone --nproc_per_node=16 source/train_gpt.py 2>&1
    rc=$?
    printf 'EXIT_CODE=%s\nEND_UTC=%s\n' "$rc" "$(date -u +%FT%TZ)"
    exit "$rc"
} | tee -- "$main_log"
exit "${PIPESTATUS[0]}"
