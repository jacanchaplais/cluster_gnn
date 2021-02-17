#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --job-name=lgntrain
#SBATCH --mem=48G

source activate lgn

projdir=$HOME/projects/particle_train
workdir=$projdir/models/lgn
datadir=$projdir/data/processed

pkgdir=$HOME/projects/external/LorentzGroupNetwork
progfile=$pkgdir/scripts/train_lgn.py

# command line args for script
args="--workdir=$workdir"
args="$args --datadir=$datadir"
args="$args --num-epoch=8"

# because they don't seem to have implemented it properly
cd $workdir

# executing the script
echo $args | xargs python $progfile
