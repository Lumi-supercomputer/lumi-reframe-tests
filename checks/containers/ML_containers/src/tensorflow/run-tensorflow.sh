#!/bin/bash -e
$WITH_CONDA
set -x

  export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
  export MIOPEN_USER_DB_PATH="/tmp/tiksmihk2-miopen-cache-$SLURM_NODEID"
  export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

  # Set MIOpen cache out of the home folder.
  if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
  fi
  sleep 3
  
  # Report affinity
  echo "Rank $SLURM_PROCID --> $(taskset -p $$)"
  
  python -u tensorflow2_synthetic_benchmark.py --batch-size 256
