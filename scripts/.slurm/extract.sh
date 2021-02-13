#!/bin/bash

#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#SBATCH --mem=20G

projdir=$1
simdir=$2
rundir=$3
tagmcpid=$4

# setting up conda
source activate jet-tools

# identifying the location of the script to run
progfile=$projdir/hepwork/data/make_dataset.py
datafile=$simdir/Events/$rundir/*.hepmc.gz
runconfigfile=$simdir/Cards/run_card.dat

numevts=`grep -Po '\d*(?=(.*= nevents))' $runconfigfile`

runnum=`echo $rundir | cut -d"_" -f 1`
offset=$(( (10#$runnum - 1) * $numevts ))

# command eine args for script
args="extract"
args="$args --stride 625"
args="$args --offset $offset"
args="$args --num-procs 8"
args="$args $datafile"
args="$args $numevts"
args="$args $tagmcpid"

# executing the script
echo $args | xargs python $progfile
