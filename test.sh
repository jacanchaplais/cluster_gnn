#!/bin/bash
datadir=/scratch/jlc1n20/data
package=sm_tpair2
eventdir=$datadir/$package/Events/

# identifying the location of the script to run
progfile=$datadir/$package/bin/generate_events
runconfigfile=$datadir/$package/Cards/run_card.dat

# increment num in dir where run data is stored
lastrun=0 # initialise as if no previous runs
prevruns=$( ls -l $eventdir | grep ^d ) # set a variable if there were

if [ -z $prevruns ] ; then # if previous runs, store number from dir name
    for dir in $eventdir/*/ ; do
        dirname=$( basename $dir )
    done

    lastrun=$(echo $dirname | cut -d'_' -f 1)
fi

printf "%02d" -v runnum $(( lastrun + 1 )) # increment run counter

#-----------------------------------------------------------------------------#

# command eine args for script
args="$runnum""_run"
echo $prevruns
