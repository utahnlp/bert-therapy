#!/bin/bash

# source envoronment variables
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

EVAL_DIR=$HOME/git-workspace/bert-therapy/

gpuid=$1
task_name=$2
encoder_name=$3
model_name_or_path=$4

EXP_DIR=$EVAL_DIR/output/$encoder_name/$task_name/CLS_256_nocopysep_slr/
DATA_DIR=$EVAL_DIR/generated_data/$encoder_name/$task_name/

### CHECK WORK & DATA DIR
if [ ! -e ${EXP_DIR} ]; then
  echo "folder does not exist:"${EXP_DIR}
  exit -1
fi

pargs="
--encoder_model_name_or_path $encoder_name \
--model_name_or_path $model_name_or_path \
--task_name $task_name \
--use_CLS \
--special_token_lr=2e-4 \
--no_pad_to_max_length \
--train_file $DATA_DIR/train.csv \
--validation_file $DATA_DIR/dev.csv \
--test_file $DATA_DIR/test.csv \
--output_dir $EXP_DIR \
--do_predict \
--fp16 \
--per_device_train_batch_size 64 \
--adafactor \
--group_by_length \
--learning_rate 2e-5 \
--warmup_steps 1000 \
--weight_decay 0.1 \
--num_train_epochs 7 \
--load_best_model_at_end \
--eval_steps 2000 \
--max_seq_length 256 \
--evaluation_strategy steps \
--metric_for_best_model f1_macro \
--label_smoothing_factor 0.1 \
--overwrite_output_dir
"

pushd $EVAL_DIR
CUDA_VISIBLE_DEVICES=$gpuid python modules/run_classifier.py $pargs 2>&1 &> $EXP_DIR/eval.log
popd
