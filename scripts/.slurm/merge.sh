#!/bin/bash

#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

projdir=$1
datadir=$2

# setting up conda
source activate jet-tools

# identifying the location of the script to run
progfile=$projdir/hepwork/data/make_dataset.py
fileglob="$datadir/*/Events/*/*.hdf5"
outdir="$projdir/data/interim/"

# command eine args for script
args="merge"
args="$args $fileglob"
args="$args $outdir"

# executing the script
echo $args | xargs python $progfile
