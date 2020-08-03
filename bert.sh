python modules/run_classifier.py --data_dir ../data/psyc_MISC11_ML_17_padding \
--model_type bert \
--model_path bert-base-uncased \
--task_name categorize-span-both-4 \
--num_train_epochs 20 \
--output_dir ../results/categorize-span-both-4 \
--local_rank -1 \
--cache_dir $HOME/.cache \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 16 \
--gradient_accumulation_steps 8 \
--save_steps 3000 \
--warmup_steps 1000 \
--weight_decay 0.1 \
--learning_rate 3e-5 \
--do_train \
--evaluate_during_training \
--overwrite_output_dir

# --do_train \
# --evaluate_during_training \
# --overwrite_output_dir

# --do_eval \
# --eval_all_checkpoints

# --do_test \
# --eval_all_checkpoints
