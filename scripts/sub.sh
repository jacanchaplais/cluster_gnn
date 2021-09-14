#!/bin/bash
#SBATCH --partition=ecsstaff
#SBATCH --account=ecsstaff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=ui
#SBATCH --output=/home/jlc1n20/projects/cluster_gnn/log/%x-%j.out
#SBATCH --gres-flags=enforce-binding

# setting up environment
source activate ptg
# export CUDA_VISIBLE_DEVICES='0,1,2,3'
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# filesystem
projdir=$HOME/projects/cluster_gnn
progfile=$projdir/src/cluster_gnn/models/train_model.py

# execution
python $progfile -c $projdir/configs/train/dev.yml
