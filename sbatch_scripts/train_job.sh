#!/bin/bash

#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --output=/home/utah/jiecao/dgx_jobs/misc-bert/train.txt
#SBATCH --ntasks=1
#SBATCH --time=80:40:00
#SBATCH --mem=40G
pushd $CODE_BASE/bert-therapy/sbatch_scripts/
echo "assign gpus ids:"$CUDA_VISIBLE_DEVICES
train_script=$1
gpus=$CUDA_VISIBLE_DEVICES
task=$2
bert_type=$3
${train_script} $gpus $task $bert_type
popd
