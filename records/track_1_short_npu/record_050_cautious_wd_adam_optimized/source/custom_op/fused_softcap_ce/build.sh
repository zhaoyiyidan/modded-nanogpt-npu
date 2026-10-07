#!/usr/bin/env bash
set -u

source /root/miniconda3/etc/profile.d/conda.sh
conda activate llm_test

# CANN's environment script probes optional driver install metadata with grep.
# On this host one of those probes returns 1 even though the environment setup
# succeeds, so do not let errexit/pipefail turn the probe into a build failure.
set +e
set +u
set +o pipefail
source /usr/local/Ascend/cann-8.5.0/set_env.sh
ascend_env_rc=$?
set -o pipefail
set -u
set -e

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
export CMAKE_PREFIX_PATH=/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64/cmake

if [[ ${ascend_env_rc} -ne 0 ]]; then
    echo "warning: CANN set_env.sh returned ${ascend_env_rc}; validating required paths directly" >&2
fi
if [[ ! -x "${ASCEND_HOME_PATH}/tools/ccec_compiler/bin/ccec" ]]; then
    echo "error: AscendC compiler not found under ${ASCEND_HOME_PATH}" >&2
    exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_dir="${script_dir}/build"
cmake -S "${script_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel 1
