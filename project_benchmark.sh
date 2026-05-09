#!/bin/bash
#
# Reproduce the six benchmark rows:
#   steps = 50, 100
#   GPUs  = 1, 2, 4 on one node
#
# Override MODEL_SIZE or backend knobs if needed, for example:
#   MODEL_SIZE=760m ATTENTION_BACKEND=flash bash project_benchmark.sh
#   DRY_RUN=true bash project_benchmark.sh

set -euo pipefail

MODEL_SIZE=${MODEL_SIZE:-125m}
NODES=${NODES:-1}
SEQ_LEN=${SEQ_LEN:-4096}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto}
TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-transformer_engine}

for steps in 50 100; do
    for gpus in 1 2 4; do
        echo "Submitting: model=${MODEL_SIZE} steps=${steps} gpus=${gpus} seq_len=${SEQ_LEN} backend=${ATTENTION_BACKEND}"
        GPUS_PER_NODE=$gpus \
        SEQ_LEN=$SEQ_LEN \
        ATTENTION_BACKEND=$ATTENTION_BACKEND \
        TRANSFORMER_IMPL=$TRANSFORMER_IMPL \
            ./launch.sh throughput "$MODEL_SIZE" "$steps" "$NODES"
    done
done
