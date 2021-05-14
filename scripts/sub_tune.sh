#!/bin/bash
#SBATCH --partition=ecsall
#SBATCH --account=ecsstaff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --time=120:00:00
#SBATCH --job-name=tune
#SBATCH --output=/home/jlc1n20/projects/cluster_gnn/log/%x-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlc1n20@soton.ac.uk

# setting up environment
source activate ptg
# export CUDA_VISIBLE_DEVICES='0,1,2,3'
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# filesystem
projdir=$HOME/projects/cluster_gnn
progfile=$projdir/scripts/tune.py

# execution
python $progfile
