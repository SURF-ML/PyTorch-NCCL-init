#!/bin/bash

# This simple launcher script sets job-specific environment variables
# and executes the provided command with python.



export WORLD_SIZE=$SLURM_NTASKS    # Note: only valid if ntasks==ngpus
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

if [[ "$SLURM_PROCID" == "0" ]]; then
  export NCCL_DEBUG=INFO
fi

python3 -u launch.py
