#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Per-rank Triton and NPU caches are set at process startup in train_gpt.py,
# after torchrun has assigned RANK. The original training settings remain.
