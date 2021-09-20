#!/bin/bash
#SBATCH --partition=ecsall
#SBATCH --account=ecsstaff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=ignn
#SBATCH --output=log/%x-%j.out
#SBATCH --gres-flags=enforce-binding

if [ -z $1 ] ; then
    echo "you must supply a config file"
    exit 2
else
    configfile=$1
fi

# filesystem
projdir=$(git rev-parse --show-toplevel)
progfile=$projdir/src/cluster_gnn/models/train_model.py

if [ -z "${DEV+x}" ] ; then
    conda_ptg=ptg
else
    if [[ $projdir == *"$(readlink -m $DEV)"* ]] ; then
        conda_ptg=ptg
    else
        conda_ptg=ptg_prod
    fi
fi

# setting up environment
source activate $conda_ptg
# export CUDA_VISIBLE_DEVICES='0,1,2,3'
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# execution
python $progfile -c $configfile
