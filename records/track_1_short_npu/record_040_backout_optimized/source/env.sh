#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export DATA_PATH="${DATA_PATH:-$(cd ../../.. && pwd -P)}"
export TRAIN_SEED=42 TASK_QUEUE_ENABLE=1 CPU_AFFINITY_CONF=1
export OUTPUT_DIR="$PWD/logs" TORCH_NPU_COMPILE_CACHE_DIR="$PWD/compile_cache"
[[ -s source/custom_op/fused_softcap_ce/libcodex_fused_softcap_ce_ops.so ]] || {
    echo 'Missing record 40 custom operator; see archive/build/build.sh' >&2
    return 1
}
