#!/bin/bash
datadir=/scratch/jlc1n20/data
package=toptag
eventdir=$datadir/$package/Events/

# identifying the location of the script to run
progfile=$datadir/$package/bin/generate_events
runconfigfile=$datadir/$package/Cards/run_card.dat

#------------------ MADGRAPH DIRECTORY STRUCTURE AND CONFIG ------------------#

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

printf "%03d" $((10#$lastrun + 1))
#-----------------------------------------------------------------------------#

# # command eine args for script
# args="$runnum""_run"
# echo $runnum
