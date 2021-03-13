#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import os
import logging
import random
import sys
import re
from dataclasses import dataclass, field
from typing import Optional, List
from sklearn.metrics import precision_recall_fscore_support, f1_score

import numpy as np
import pandas as pd
from datasets import Dataset
from misc_config import MISCConfig
import misc_constants

import transformers

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from misc_bert import MISCBERTModel, MISCClassificationHead
from my_trainer import MyTrainer
import misc_constants

task_to_keys = {
    "utter": ("utterance", None),
    "concat": ("utterance", "context"),
    "sep": ("utterance", "context"),
    "speaker": ("utterance", "context"),
    "speaker_span": ("utterance", "context"),
}

logger = logging.getLogger(__name__)

@dataclass
class MISCTrainingArguments(TrainingArguments):
    # adding special training arguments
    special_token_lr: Optional[float]= field(
        default=None,
        metadata={
            "help": "Whether using learning rate for special tokens."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: str = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    context_first: bool = field(
        default=False,
        metadata={"help": "use context first in sentence pair tasks"},
    )
    ## custom
    tokenizer_additional_tokens: Optional[List[str]] = field(
        default_factory= lambda: misc_constants.special_speaker_tags,
        metadata={
            "help": "Additional tokens to add to the tokenizer"
        },
    )

    def __post_init__(self):
        if self.train_file is None or self.validation_file is None:
            raise ValueError("Need training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    encoder_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained encoder model or model identifier from huggingface.co/models"}
    )
    model_name_or_path: str = field(
        default="", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    copy_sep: bool = field(
        default=False,
        metadata={
            "help": "Initialized any additional tokens to the value of tokenizer.cls_token."
        },
    )
    use_CLS: bool = field(
        default=True,
        metadata={
            "help": "Whether use CLS for classification head."
        },
    )
    use_start_U: bool = field(
        default=False,
        metadata={
            "help": "Whether use the start U tag for classification head."
        },
    )
    use_end_U: bool = field(
        default=False,
        metadata={
            "help": "Whether use the end U tag for classification head."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MISCTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need either a test file for `do_predict`.")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")
    # Loading a dataset from local csv files
    datasets = {k: pd.read_csv(v, delimiter='ך', engine='python') for k, v in data_files.items()}
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = datasets["train"]["label"].unique()
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # for the original config, we add in extra configurations to support other variants such as classification head, simply by assign values to those attributes, the type of config class is according to the model_name or path, or model_type in the config dict

    # here, model is for the whole MISC BERT model, not for the BERT encoder models.
    if model_args.model_name_or_path:
        config = MISCConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            special_token_lr=training_args.special_token_lr,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # no need to specifiy the args when loading model
        logger.info("config is {}".format(config))
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.encoder_model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # still load original tokenizer, add readd the new special tokens
        num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': data_args.tokenizer_additional_tokens })

        model = MISCBERTModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            tokenizer=tokenizer
        )
    else:
        logger.info("{} is not existed, training model from scratch and load encoder_model_name_or_path".format(model_args.model_name_or_path))
        config = MISCConfig()
        encoder_config = MISCConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.encoder_model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config.update_config(
            encoder_model_name_or_path=model_args.encoder_model_name_or_path,
            use_CLS=model_args.use_CLS,
            use_start_U=model_args.use_start_U,
            use_end_U=model_args.use_end_U,
            tokenizer_name=model_args.tokenizer_name,
            use_fast_tokenizer=model_args.use_fast_tokenizer,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            special_token_lr=training_args.special_token_lr,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config.update_encoder_config(encoder_config)
        logger.info("config is {}".format(config))
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.encoder_model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # Custom tokens
        num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': data_args.tokenizer_additional_tokens })
        model = MISCBERTModel(
            config=config,
            tokenizer=tokenizer
        )
        embeddings = model.encoder.resize_token_embeddings(len(tokenizer)) # doesn't mess with existing tokens
        if model_args.copy_sep:
            embeddings.weight.data[tokenizer.additional_special_tokens_ids, :] = embeddings.weight.data[tokenizer.sep_token_id, :].repeat(num_added_tokens, 1)
    # Preprocessing the datasets
    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [name for name in datasets["train"].columns if name != "label"]
    if "context" in non_label_column_names and "utterance" in non_label_column_names:
        if data_args.context_first:
            sentence1_key, sentence2_key = "context", "utterance"
        else:
            sentence1_key, sentence2_key = "utterance", "context"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    logger.info(" Using sentence1_key = {}, sentence2_key = {}".format(sentence1_key, sentence2_key))
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    logger.info(" Using max_seq_length = {}, padding = {}".format(max_seq_length, padding))
    logger.info("  Model Architecture = %s", model)
    logger.info("  Trainable Model Parameters = %s", sum(p.numel() for p in model.parameters() if p.requires_grad))

    def preprocess_function(examples):
        # Tokenize the texts
        # https://github.com/huggingface/transformers/blob/e6ce636e02ec1cd3f9893af6ab1eec4f113025db/src/transformers/tokenization_utils_base.py#L2110
        # all the default truncation will break the paired tags. We use our own truncation
        batch_input_ids = []
        from collections import Counter
        cnt = Counter()
        if sentence2_key is None:
            # only has the utterance, then we truncate the utterance, but adding the end tags
            num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
            for snt in examples[sentence1_key].tolist():
                total_length = max_seq_length - num_speaker_tokens
                all_subwords1 = tokenizer.tokenize(snt)
                ori_snt1_len = len(all_subwords1)
                # check the paired tags only when speaker_span
                if data_args.task_name == 'speaker_span' and len(all_subwords1) >= total_length:
                    # the last two tag are not splittable.
                    all_subwords1[total_length-1] = all_subwords1[-1]
                    all_subwords1[total_length-2] = all_subwords1[-2]
                # when it is less then total_length, no padding will happen here.
                all_subwords1 = all_subwords1[:total_length]
                subwords1_ids = tokenizer.convert_tokens_to_ids(all_subwords1)
                subwords2_ids = None
                batch_input_ids.append((subwords1_ids, subwords2_ids2))
                cnt[ori_snt1_len + num_special_tokens] += 1
        else:
            # for sentence pair
            # we truncate the context first, but adding the end tags
            num_special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
            for idx, (snt1,snt2) in enumerate(zip(examples[sentence1_key].tolist(), examples[sentence2_key].tolist())):
                total_length = max_seq_length - num_special_tokens
                current_subwords1 = []
                current_subwords2 = []
                all_subwords1 = tokenizer.tokenize(snt1)
                ori_snt1_len = len(all_subwords1)
                # check the paired tags only when speaker_span
                if 'speaker_span' in data_args.task_name and len(all_subwords1) >= total_length:
                    # the last two tag are not splittable.
                    all_subwords1[total_length-1] = all_subwords1[-1]
                    all_subwords1[total_length-2] = all_subwords1[-2]
                # when it is less then total_length, no padding will happen here.
                all_subwords1 = all_subwords1[:total_length]
                subwords1_ids = tokenizer.convert_tokens_to_ids(all_subwords1)

                all_subwords2 = tokenizer.tokenize(snt2)
                ori_snt2_len = len(all_subwords2)
                total_length_for_2 = total_length - len(all_subwords1)
                all_subwords2 = all_subwords2[:total_length_for_2]
                subwords2_ids = tokenizer.convert_tokens_to_ids(all_subwords2)
                batch_input_ids.append((subwords1_ids, subwords2_ids))
                cnt[ori_snt1_len + ori_snt2_len + num_special_tokens] += 1

        histgram_key = [128, 256, 512]
        for m in histgram_key:
           summ = sum([c  for k, c in cnt.items() if k > m])
           logger.info("{}/{} examples have larger length than {}".format(summ, len(examples[sentence1_key]), m))



        batch_outputs = {}
        for idx, (first_ids, second_ids) in enumerate(batch_input_ids):
            # setting it as False wil have a warning from pytorch, but we don't need truncation any more after the previous
            outputs = tokenizer.prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=True,
                padding=padding,  # we pad in batch afterward
                truncation=False,
                max_length=max_seq_length,
                return_attention_mask=True,  # we pad in batch afterward
                return_token_type_ids=True,
            )

            if len(outputs['input_ids']) != len(outputs['token_type_ids']):
                logger.error("for {}, invalid length {}, {}".format('token_type_ids', len(outputs['input_ids']), len(outputs['token_type_ids']), outputs['token_type_ids']))
            if len(outputs['input_ids']) != len(outputs['attention_mask']):
                logger.error("for {}, invalid length {}, {}".format('attention_mask', len(outputs['input_ids']), outputs['attention_mask']))

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # no need for padding for all
        #result = tokenizer.pad(
        #    batch_outputs,
        #    padding=padding,
        #    max_length=max_seq_length,
        #    return_attention_mask=True
        #)

        if label_to_id is not None and "label" in examples:
            batch_outputs["label"] = [label_to_id[l] for l in examples["label"]]
        return batch_outputs

    datasets = {k: preprocess_function(v) for k, v in datasets.items()}

    train_dataset = Dataset.from_dict(datasets["train"])
    eval_dataset = Dataset.from_dict(datasets["validation"])
    if data_args.test_file is not None:
        test_dataset = Dataset.from_dict(datasets["test"])

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 5):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        precision, recall, f1, support = precision_recall_fscore_support(p.label_ids, preds, average=None)
        return {
            "accuracy": (preds == p.label_ids).astype(np.float32).mean().item(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "labels": label_list.tolist(),
            "f1_macro": f1_score(p.label_ids, preds, average='macro'),
            "f1_micro": f1_score(p.label_ids, preds, average='micro'),
            "f1_weighted": f1_score(p.label_ids, preds, average='weighted'),
            "support": support.tolist()
        }

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        with open(os.path.join(training_args.output_dir, "cmd_args.sh"), 'w') as f:
            f.write(' '.join(['python'] + sys.argv))

    eval_results = {}
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        task = data_args.task_name
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)
        eval_results.update(eval_result)

    # Test
    if training_args.do_predict:
        predict_results = {}
        logger.info("*** Test ***")
        task = data_args.task_name
        eval_result = trainer.evaluate(eval_dataset=test_dataset)
        trainer.log_metrics("test", eval_result)
        trainer.save_metrics("test", eval_result)
        eval_results.update(eval_result)

    return eval_results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
