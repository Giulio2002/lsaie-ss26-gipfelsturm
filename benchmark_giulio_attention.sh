#!/bin/bash
#
# Giulio's 125m throughput sweeps.
#
# Default run:
#   bash benchmark_giulio_attention.sh
#
# Useful variants:
#   SWEEP=backend bash benchmark_giulio_attention.sh
#   SWEEP=mbs bash benchmark_giulio_attention.sh
#   SWEEP=gbs MBS=16 bash benchmark_giulio_attention.sh
#   SWEEP=batch_grid bash benchmark_giulio_attention.sh
#   SWEEP=cuda_graph MBS=16 bash benchmark_giulio_attention.sh
#   SWEEP=seq MBS=16 bash benchmark_giulio_attention.sh
#   SWEEP=all DRY_RUN=true bash benchmark_giulio_attention.sh
#
# Common overrides:
#   STEPS=100 GPUS_PER_NODE=4 SEQ_LEN=4096 BACKEND="auto flash fused local" bash benchmark_giulio_attention.sh

set -euo pipefail

MODEL_SIZE=125m
STEPS=${STEPS:-100}
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SEQ_LEN=${SEQ_LEN:-4096}
BACKEND=${BACKEND:-auto}
TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-transformer_engine}
PROFILE_NSYS=${PROFILE_NSYS:-false}
SWEEP=${SWEEP:-batch_grid}
MBS_VALUES=${MBS_VALUES:-"4 8 16 24 32"}
GBS_VALUES=${GBS_VALUES:-"64 128 196 256"}
SEQ_LENS=${SEQ_LENS:-"1024 2048 4096 8192"}

submit_run() {
    local label=$1
    local seq_len=$2
    local backend=$3
    local transformer_impl=$4
    local mbs=$5
    local gbs=$6
    local cuda_graph_impl=$7
    local cuda_graph_scope=$8

    local kernel_tag=$label
    if [ "$backend" = "local" ]; then
        transformer_impl=local
    fi

    echo "Submitting: label=${label} model=${MODEL_SIZE} steps=${STEPS} gpus=${GPUS_PER_NODE} seq=${seq_len} backend=${backend} mbs=${mbs} gbs=${gbs} cuda_graph=${cuda_graph_impl}:${cuda_graph_scope}"
    SEQ_LEN=$seq_len \
    GPUS_PER_NODE=$GPUS_PER_NODE \
    ATTENTION_BACKEND=$backend \
    TRANSFORMER_IMPL=$transformer_impl \
    MBS=$mbs \
    GBS=$gbs \
    CUDA_GRAPH_IMPL=$cuda_graph_impl \
    CUDA_GRAPH_SCOPE=$cuda_graph_scope \
    KERNEL_TAG=$kernel_tag \
    PROFILE_NSYS=$PROFILE_NSYS \
        ./launch.sh throughput "$MODEL_SIZE" "$STEPS" "$NODES"
}

run_backend_sweep() {
    local mbs=${MBS:-16}
    local gbs=${GBS:-256}
    for backend in $BACKEND; do
        local transformer_impl=$TRANSFORMER_IMPL
        local label="backend-${backend}"
        if [ "$backend" = "local" ]; then
            transformer_impl=local
        fi
        submit_run "$label" "$SEQ_LEN" "$backend" "$transformer_impl" "$mbs" "$gbs" none ""
    done
}

run_mbs_sweep() {
    local gbs=${GBS:-256}
    for backend in $BACKEND; do
        for mbs in $MBS_VALUES; do
            submit_run "${backend}-mbs-${mbs}" "$SEQ_LEN" "$backend" "$TRANSFORMER_IMPL" "$mbs" "$gbs" none ""
        done
    done
}

run_gbs_sweep() {
    local mbs=${MBS:-16}
    for backend in $BACKEND; do
        for gbs in $GBS_VALUES; do
            submit_run "${backend}-gbs-${gbs}" "$SEQ_LEN" "$backend" "$TRANSFORMER_IMPL" "$mbs" "$gbs" none ""
        done
    done
}

run_batch_grid_sweep() {
    for backend in $BACKEND; do
        for mbs in $MBS_VALUES; do
            for gbs in $GBS_VALUES; do
                submit_run "${backend}-mbs-${mbs}-gbs-${gbs}" "$SEQ_LEN" "$backend" "$TRANSFORMER_IMPL" "$mbs" "$gbs" none ""
            done
        done
    done
}

run_cuda_graph_sweep() {
    local mbs=${MBS:-16}
    local gbs=${GBS:-256}
    for backend in $BACKEND; do
        submit_run "${backend}-cuda-graph-off" "$SEQ_LEN" "$backend" "$TRANSFORMER_IMPL" "$mbs" "$gbs" none ""
        submit_run "${backend}-cuda-graph-attn" "$SEQ_LEN" "$backend" "$TRANSFORMER_IMPL" "$mbs" "$gbs" transformer_engine attn
    done
}

run_seq_sweep() {
    local mbs=${MBS:-16}
    local gbs=${GBS:-256}
    for backend in $BACKEND; do
        for seq_len in $SEQ_LENS; do
            submit_run "${backend}-seq-${seq_len}" "$seq_len" "$backend" "$TRANSFORMER_IMPL" "$mbs" "$gbs" none ""
        done
    done
}

case $SWEEP in
    backend)
        run_backend_sweep
        ;;
    mbs)
        run_mbs_sweep
        ;;
    gbs)
        run_gbs_sweep
        ;;
    batch_grid)
        run_batch_grid_sweep
        ;;
    cuda_graph)
        run_cuda_graph_sweep
        ;;
    seq)
        run_seq_sweep
        ;;
    all)
        run_backend_sweep
        run_batch_grid_sweep
        run_cuda_graph_sweep
        run_seq_sweep
        ;;
    *)
        echo "Unknown SWEEP: $SWEEP. Choose: backend, mbs, gbs, batch_grid, cuda_graph, seq, all"
        exit 1
        ;;
esac
