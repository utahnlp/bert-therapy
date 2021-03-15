#!/bin/bash

# source envoronment variables
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

EVAL_DIR=$HOME/git-workspace/bert-therapy/

gpuid=$1
task_name=$2
encoder_name=$3

EXP_DIR=$EVAL_DIR/output/$encoder_name/$task_name/speaker_span_SU_256_slr/
DATA_DIR=$EVAL_DIR/generated_data/$encoder_name/$task_name/

### CHECK WORK & DATA DIR
if [ -e ${EXP_DIR} ]; then
  today=`date +%m-%d.%H:%M`
  mv ${EXP_DIR} ${EXP_DIR%?}_${today}
  echo "rename original training folder to "${EXP_DIR%?}_${today}
fi

mkdir -p $EXP_DIR

pargs="
--encoder_model_name_or_path $encoder_name
--copy_sep
--task_name $task_name \
--use_CLS=false \
--use_start_U \
--special_token_lr=2e-4 \
--no_pad_to_max_length \
--train_file $DATA_DIR/train.csv \
--validation_file $DATA_DIR/dev.csv \
--test_file $DATA_DIR/test.csv \
--output_dir $EXP_DIR \
--do_train \
--do_eval \
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
--eval_steps 1000 \
--max_seq_length 256 \
--evaluation_strategy steps \
--metric_for_best_model f1_macro \
--label_smoothing_factor 0.1 \
--overwrite_output_dir
"

pushd $EVAL_DIR
CUDA_VISIBLE_DEVICES=$gpuid python modules/run_classifier.py $pargs 2>&1 &> $EXP_DIR/train.log
popd
