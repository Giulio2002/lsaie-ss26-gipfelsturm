#!/bin/bash
#
# Usage: ./launch.sh <mode> <model_size> [steps] [nodes]
#
# Modes:     throughput  (50 steps, with W&B)
#            train       (N steps, with W&B and Tensorboard)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     optional, default 4 (max 8)
#
# Examples:  ./launch.sh throughput 760m
#            ./launch.sh throughput 8b 50 1
#            ./launch.sh train 760m 5000
#            ./launch.sh train 1.5b 3000 8
#
# Optional throughput knobs, set as environment variables:
#   SEQ_LEN=8192 ATTENTION_BACKEND=flash ./launch.sh throughput 760m 50 1
#   TRANSFORMER_IMPL=local ATTENTION_BACKEND=local ./launch.sh throughput 125m 20 1
#   TP=4 MBS=1 GBS=128 ./launch.sh throughput 8b 50 1
#   PROFILE_NSYS=true PROFILE_STEP_START=10 PROFILE_STEP_END=12 ./launch.sh throughput 760m 20 1

set -euo pipefail

source "$(dirname "$0")/config.sh"

MODE=${1:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}
MODEL_SIZE=${2:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}
MBS_OVERRIDE=${MBS:-}

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        NODES=${4:-4}
        TIME=00:30:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA=""
        WANDB=true
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./launch.sh train <model_size> <steps> [nodes]}
        NODES=${4:-4}
        TIME=02:30:00
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train"
        exit 1
        ;;
esac

################ Model config ################
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4
        MBS=16
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        MBS=8
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        MBS=4
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        MBS=4
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        MBS=4
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        MBS=2
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b"
        exit 1
        ;;
esac

MBS=${MBS_OVERRIDE:-$MBS}
GBS=${GBS:-256}
SEQ_LEN=${SEQ_LEN:-4096}
TP=${TP:-1}
PP=${PP:-1}
TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-transformer_engine}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto}
CUDA_GRAPH_IMPL=${CUDA_GRAPH_IMPL:-none}
CUDA_GRAPH_SCOPE=${CUDA_GRAPH_SCOPE:-}
PROFILE_NSYS=${PROFILE_NSYS:-false}
PROFILE_STEP_START=${PROFILE_STEP_START:-10}
PROFILE_STEP_END=${PROFILE_STEP_END:-12}
PROFILE_RANKS=${PROFILE_RANKS:-0}
DRY_RUN=${DRY_RUN:-false}

case $ATTENTION_BACKEND in
    auto|flash|fused|unfused|local) ;;
    *)
        echo "Unknown ATTENTION_BACKEND: $ATTENTION_BACKEND. Choose: auto, flash, fused, unfused, local"
        exit 1
        ;;
esac

case $TRANSFORMER_IMPL in
    transformer_engine|local) ;;
    *)
        echo "Unknown TRANSFORMER_IMPL: $TRANSFORMER_IMPL. Choose: transformer_engine, local"
        exit 1
        ;;
esac

if [ "$ATTENTION_BACKEND" = "local" ] && [ "$TRANSFORMER_IMPL" != "local" ]; then
    echo "ATTENTION_BACKEND=local requires TRANSFORMER_IMPL=local."
    exit 1
fi

KERNEL_TAG=${KERNEL_TAG:-${TRANSFORMER_IMPL}-${ATTENTION_BACKEND}}
JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-${KERNEL_TAG}-${SEQ_LEN}seq-${TRAINING_STEPS}s-${NODES}n"

################ W&B block ################
if [ "$WANDB" = true ]; then
    WANDB_BLOCK='
# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'
else
    WANDB_BLOCK='export WANDB_MODE=disabled'
fi

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY_HEAD'

echo "START TIME: \$(date)"

################ Configs ################
BODY_HEAD

cat >> "$SCRIPT" << BODY_WORKDIR
WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/cache
BODY_WORKDIR

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}
TP=${TP}
PP=${PP}
TRANSFORMER_IMPL=${TRANSFORMER_IMPL}
ATTENTION_BACKEND=${ATTENTION_BACKEND}
CUDA_GRAPH_IMPL=${CUDA_GRAPH_IMPL}
CUDA_GRAPH_SCOPE="${CUDA_GRAPH_SCOPE}"
PROFILE_NSYS=${PROFILE_NSYS}
PROFILE_STEP_START=${PROFILE_STEP_START}
PROFILE_STEP_END=${PROFILE_STEP_END}
PROFILE_RANKS="${PROFILE_RANKS}"

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=${MODE}-${MODEL_SIZE}-${KERNEL_TAG}-${SEQ_LEN}seq-tp${TP}-pp${PP}-\${SLURM_NNODES}n
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ARGS=(
    --transformer-impl ${TRANSFORMER_IMPL}
)

if [ "${TRANSFORMER_IMPL}" = "transformer_engine" ]; then
    TRANSFORMER_ARGS+=(
        --use-precision-aware-optimizer
        --main-grads-dtype bf16
    )
fi

ATTENTION_ARGS=(
    --attention-backend ${ATTENTION_BACKEND}
)

if [ "${ATTENTION_BACKEND}" = "local" ]; then
    ATTENTION_ARGS+=(--spec local)
fi

CUDA_GRAPH_ARGS=()
if [ "${CUDA_GRAPH_IMPL}" != "none" ]; then
    CUDA_GRAPH_ARGS+=(--cuda-graph-impl ${CUDA_GRAPH_IMPL})
    if [ -n "${CUDA_GRAPH_SCOPE}" ]; then
        CUDA_GRAPH_ARGS+=(--cuda-graph-scope ${CUDA_GRAPH_SCOPE})
    fi
    if [ "${CUDA_GRAPH_IMPL}" = "transformer_engine" ]; then
        CUDA_GRAPH_ARGS+=(--te-rng-tracker)
    fi
fi

PROFILE_ARGS=()
if [ "${PROFILE_NSYS}" = "true" ]; then
    PROFILE_ARGS+=(
        --profile
        --profile-step-start ${PROFILE_STEP_START}
        --profile-step-end ${PROFILE_STEP_END}
        --profile-ranks ${PROFILE_RANKS}
    )
fi

SETUP

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --lr-decay-style constant
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)
TRAINING

cat >> "$SCRIPT" << 'REST'

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
REST

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
)
LOGGING_EXTRA

cat >> "$SCRIPT" << 'TOKENIZER'

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ARGS[@]} \
    ${ATTENTION_ARGS[@]} \
    ${CUDA_GRAPH_ARGS[@]} \
    ${PROFILE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

# Replace placeholder with actual W&B block
sed -i.bak '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
rm -f "$SCRIPT.bak"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'FOOTER'

echo "CMD: $TRAINING_CMD"
if [ "${PROFILE_NSYS}" = "true" ]; then
    mkdir -p "$LOG_DIR/nsys"
    TRAINING_CMD="nsys profile -s none -t nvtx,cuda -o $LOG_DIR/nsys/${EXP_NAME}-${SLURM_JOB_ID} --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop $TRAINING_CMD"
fi
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 $TRAINING_CMD"

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
if [ "$DRY_RUN" = "true" ]; then
    echo "DRY_RUN=true, not submitting $SCRIPT"
else
    sbatch "$SCRIPT"
fi
