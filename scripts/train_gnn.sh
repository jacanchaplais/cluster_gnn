#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=gnntrain
#SBATCH --mem=40G

source activate PyG

projdir=$HOME/projects/cluster_gnn
progfile=$projdir/scripts/train.py

# executing the script
python $progfile
