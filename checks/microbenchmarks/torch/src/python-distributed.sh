#!/bin/bash -e

# The usual PyTorch initialisations (also needed on NVIDIA)
# Note that since we fix the port ID it is not possible to run, e.g., two
# instances via this script using half a node each.
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# This is to workaround AITER issue https://github.com/ROCm/aws-ofi-rccl/issues/16
export AITER_JIT_DIR=/tmp/my-aiter-jit-dir-$SLURM_LOCALID
mkdir -p $AITER_JIT_DIR/build
#export JIT_WORKSPACE_DIR=$AITER_JIT_DIR
export CC=clang
export CXX=clang++

# Run application
python "$@"
