#!/bin/bash


set -e

usage() {
    echo "Usage: bash pyg-pip.sh ENV_NAME"
    echo "Description:
    Install PyTorch Geometric into prepared conda env."
    exit 2
}

ENV_NAME=$1

if [ -z $ENV_NAME ] ; then
    usage
fi

source activate $ENV_NAME
TORCH=1.7.0
CUDA=cu102

python -m pip install torch-scatter --no-cache-dir --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-sparse --no-cache-dir --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-cluster --no-cache-dir --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-spline-conv --no-cache-dir --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-geometric --no-cache-dir 
