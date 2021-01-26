#!/bin/bash

#SBATCH --ntasks-per-node=20
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=00:40:00          # walltime
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlc1n20@soton.ac.uk

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

# increment num in dir where run data is stored
prevruns=$(ls -l $eventdir | grep ^d) # set a variable if there were

if [ -z "$prevruns" ] ; then
    lastrun=0 # initialise as if no previous runs
else # if previous runs, store number from dir name
    for dir in $eventdir/*/ ; do
        dirname=$(basename $dir)
    done

    lastrun=$(echo $dirname | cut -d'_' -f 1)
fi

runnum=`printf "%03d" $((10#$lastrun + 1))` # increment run counter

#-----------------------------------------------------------------------------#

# command eine args for script
args="$runnum""_run"
args="$args -f"
args="$args --multicore"
args="$args --nb_core=20"

# executing the script
echo $args | xargs $progfile
