#!/bin/bash

#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=00:10:00          # walltime

simdir=$1
condenv=$2
rundir=$3

# setting up conda
source activate $condenv

# identifying the location of the script to run
progfile=$simdir/bin/generate_events
runconfigfile=$simdir/Cards/run_card.dat

#------------------ MADGRAPH DIRECTORY STRUCTURE AND CONFIG ------------------#
# scramble the rng seed in the config file
sed -i "s/\([^[\d]]*\).*\(\d*\)\(.*=.*iseed\)/\1 $RANDOM \3/" $runconfigfile


# command line args for script
args="$rundir"
args="$args -f"
args="$args --multicore"
args="$args --nb_core=4"

# executing the script
echo $args | xargs $progfile
