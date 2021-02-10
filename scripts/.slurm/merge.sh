#!/bin/bash

#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

datadir=$1
projdir=$2
outdir=$3

# setting up conda
source activate jet-tools

# identifying the location of the script to run
progfile=$projdir/hepwork/data/make_dataset.py

# executing the script
python $progfile merge ${datadir}'/*/Events/*/*.hdf5' $outdir
