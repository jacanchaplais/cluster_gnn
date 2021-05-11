#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=lr14
#SBATCH --output=/home/jlc1n20/projects/cluster_gnn/log/%x-%j.out
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlc1n20@soton.ac.uk

source activate PyG

projdir=$HOME/projects/cluster_gnn
progfile=$projdir/scripts/train.py

# executing the script
python $progfile
