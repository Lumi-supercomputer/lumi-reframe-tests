#!/bin/bash -e

# Make sure GPUs are up
if [ $SLURM_LOCALID -eq 0 ] ; then
    rocm-smi
fi
sleep 2

# Report affinity
echo "Rank $SLURM_PROCID --> $(taskset -p $$)"

# Start conda environment inside the container
$WITH_CONDA

# Set interfaces to be used by RCCL.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

# Set environment for the app
export MASTER_ADDR=$(python get-master.py "$SLURM_NODELIST")
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export RANK=$SLURM_PROCID
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
export PYTHONPATH="/opt/rocm/share/amd_smi:$PYTHONPATH"

# Run app
cd mnist
python -u mnist_DDP.py --gpu --modelpath model

