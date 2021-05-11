#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=bcel8
#SBATCH --output=/home/jlc1n20/projects/cluster_gnn/log/%x-%j.out
#SBATCH --mem=40G

source activate PyG

projdir=$HOME/projects/cluster_gnn
progfile=$projdir/scripts/train_bce.py

# executing the script
python $progfile
