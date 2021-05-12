#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --job-name=prof
#SBATCH --output=/home/jlc1n20/projects/cluster_gnn/log/%x-%j.out

source activate PyG

projdir=$HOME/projects/cluster_gnn
progfile=$projdir/scripts/train.py

# executing the script
python $progfile
