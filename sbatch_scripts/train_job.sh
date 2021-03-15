#!/bin/bash

#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --output=/home/utah/jiecao/dgx_jobs/amr/train.txt
#SBATCH --ntasks=1
#SBATCH --time=80:40:00
#SBATCH --mem=40G
pushd $CODE_BASE/mrp/Expt/mrp-scripts/commands/
echo "assign gpus ids:"$CUDA_VISIBLE_DEVICES
./train.sh $1
popd
