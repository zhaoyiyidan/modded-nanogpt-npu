#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export DATA_PATH="${DATA_PATH:-$(cd ../../.. && pwd -P)}"
export TASK_QUEUE_ENABLE=2 CPU_AFFINITY_CONF=1
