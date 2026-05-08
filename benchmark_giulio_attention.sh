#!/bin/bash
#
# Giulio's attention/backend throughput sweep.
#
# Defaults are intentionally short and single-node so the script can be used
# to smoke-test kernels before spending larger allocations.
#
# Examples:
#   bash benchmark_giulio_attention.sh
#   MODEL_SIZE=760m STEPS=30 NODES=1 SEQ_LENS="512 1024 2048 4096 8192" bash benchmark_giulio_attention.sh
#   BACKENDS="flash fused unfused local" PROFILE_NSYS=true bash benchmark_giulio_attention.sh

set -euo pipefail

MODEL_SIZE=${MODEL_SIZE:-125m}
STEPS=${STEPS:-20}
NODES=${NODES:-1}
SEQ_LENS=${SEQ_LENS:-"512 1024 2048 4096 8192 16384"}
BACKENDS=${BACKENDS:-"auto flash fused unfused"}
PROFILE_NSYS=${PROFILE_NSYS:-false}

for seq_len in $SEQ_LENS; do
    for backend in $BACKENDS; do
        transformer_impl=${TRANSFORMER_IMPL:-transformer_engine}
        kernel_tag="te-${backend}"

        if [ "$backend" = "local" ]; then
            transformer_impl=local
            kernel_tag="local"
        fi

        echo "Submitting: model=${MODEL_SIZE} backend=${backend} seq=${seq_len} steps=${STEPS} nodes=${NODES}"
        SEQ_LEN=$seq_len \
        ATTENTION_BACKEND=$backend \
        TRANSFORMER_IMPL=$transformer_impl \
        KERNEL_TAG=$kernel_tag \
        PROFILE_NSYS=$PROFILE_NSYS \
            ./launch.sh throughput "$MODEL_SIZE" "$STEPS" "$NODES"
    done
done
