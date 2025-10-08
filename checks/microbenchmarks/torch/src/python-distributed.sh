#!/bin/bash -e

# The usual PyTorch initialisations (also needed on NVIDIA)
# Note that since we fix the port ID it is not possible to run, e.g., two
# instances via this script using half a node each.
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export AITER_JIT_DIR=/tmp/my-aiter-jit-dir-$SLURM_LOCALID
mkdir -p $AITER_JIT_DIR/build
export CC=clang
export CXX=clang++

# Run application
python "$@"
