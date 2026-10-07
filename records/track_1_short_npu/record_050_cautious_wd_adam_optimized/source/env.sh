#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export DATA_PATH="${DATA_PATH:-$(cd ../../.. && pwd -P)}"
[[ -s source/custom_op/fused_softcap_ce/libcodex_fused_softcap_ce_ops.so ]] || {
    echo 'Missing record 50 custom operator; see source/custom_op/fused_softcap_ce/build.sh' >&2
    return 1
}
