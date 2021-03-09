# bert-therapy
Transformer-based observers in Psychotherapy

Generate Data:
```
python modules/processors/TASK.py
```
Make the appropriate directory

Sample Training:
```python modules/run_classifier.py --model_name_or_path bert-base-uncased --copy_sep --task_name speaker --train_file ./generated_data/bert-base-uncased/speaker/train.csv --validation_file ./generated_data/bert-base-uncased/speaker/dev.csv --output_dir ./output/bert-base-uncased/speaker/ --do_train --do_eval --fp16 --per_device_train_batch_size 64 --adafactor --group_by_length --learning_rate 2e-5 --warmup_steps 1000 --weight_decay 0.1 --num_train_epochs 7 --load_best_model_at_end --eval_steps 2000 --evaluation_strategy steps --metric_for_best_model f1_weighted --label_smoothing_factor 0.1```

Sample Evaluation:
```python modules/run_classifier.py --model_name_or_path output/roberta-large/speaker_span/checkpoint-4000 --task_name speaker_span --train_file ./generated_data/roberta-large/speaker_span/train.csv --validation_file ./generated_data/roberta-large/speaker_span/dev.csv --test_file ./generated_data/roberta-large/speaker_span/test.csv --output_dir ./output/roberta-large/speaker_span/checkpoint-4000 --per_device_eval_batch_size 128 --do_predict --fp16 --label_smoothing_factor 0.1 ```

