#!/bin/bash

#SBATCH --ntasks-per-node=20
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=00:40:00          # walltime

datadir=/scratch/jlc1n20/data
package=toptag
eventdir=$datadir/$package/Events/

# setting up conda
source activate lgn

# identifying the location of the script to run
progfile=$datadir/$package/bin/generate_events
runconfigfile=$datadir/$package/Cards/run_card.dat

#------------------ MADGRAPH DIRECTORY STRUCTURE AND CONFIG ------------------#
# scramble the rng seed in the config file
sed -i -e "s/[0-9] *=* iseed/$RANDOM = iseed/" $runconfigfile

#-----------------------------------------------------------------------------#

# command eine args for script
args="001_run"
args="$args -f"
args="$args --multicore"
args="$args --nb_core=20"

# executing the script
echo $args | xargs $progfile
