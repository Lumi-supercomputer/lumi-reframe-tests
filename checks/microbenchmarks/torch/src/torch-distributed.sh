#!/bin/bash -e

# The usual PyTorch initialisations (also needed on NVIDIA)
# Note that since we fix the port ID it is not possible to run, e.g., two
# instances via this script using half a node each.
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID
GPUS_PER_NODE=$SLURM_GPUS_PER_NODE

# This is to workaround AITER issue https://github.com/ROCm/aws-ofi-rccl/issues/16
export AITER_JIT_DIR=/tmp/my-aiter-jit-dir-$SLURM_LOCALID
mkdir -p $AITER_JIT_DIR/build
export JIT_WORKSPACE_DIR=$AITER_JIT_DIR
export CC=clang
export CXX=clang++

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
"
# Run application
torchrun $DISTRIBUTED_ARGS "$@"
