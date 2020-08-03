import glob
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
                          WEIGHTS_NAME, AdamW, AutoConfig,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from arguments import argparser
from processors.text_processor import PsychDataset
from utils import compute_metrics, logits_masked, processors, set_seed, add_special_tokens

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)
symbol_dict = dict()


def train(model, tokenizer, train_dataset, processor, args):
  """ Train the model """

  ### Get train Dataloader
  if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter()
  else:
    tb_writer = None

  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
  train_dataloader = DataLoader(
    train_dataset, 
    sampler=train_sampler,
    batch_size=args.train_batch_size, 
    collate_fn=processor.collate(return_token_type_ids=False),
  )

  ## Calculate training steps
  if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = (
      args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    )
  else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    num_train_epochs = args.num_train_epochs

  ## Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
  ]

  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
  )

  #@ Check if saved optimizer or scheduler states exist
  if (
    args.model_path
    and os.path.isfile(os.path.join(args.model_path, "optimizer.pt")) 
    and os.path.isfile(os.path.join(args.model_path, "scheduler.pt"))
  ):
    # Load in optimizer and scheduler states
    optimizer.load_state_dict(
      torch.load(os.path.join(args.model_path, "optimizer.pt"), map_location=args.device)
    )
    scheduler.load_state_dict(torch.load(os.path.join(args.model_path, "scheduler.pt")))

  if args.fp16:
    try:
      from apex import amp
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

  # multi-gpu training (should be after apex fp16 initialization)
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

  if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
      model, 
      device_ids=[args.local_rank],
      output_device=args.local_rank, 
      find_unused_parameters=True,
    )

  total_train_batch_size =  (args.train_batch_size
    * args.gradient_accumulation_steps
    * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
  )
  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataset))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
  logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
  logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
  logger.info("  Total optimization steps = %d", t_total)

  global_step = 0
  epochs_trained = 0
  steps_trained_in_current_epoch = 0
  # Check if continuing training from a checkpoint
  if os.path.exists(args.model_path):
    # set global_step to gobal_step of last saved checkpoint from model path
    try:
      global_step = int(args.model_path.split("-")[-1].split("/")[0])
      epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
      steps_trained_in_current_epoch = global_step % (
        len(train_dataloader) // args.gradient_accumulation_steps
      )

      logger.info("  Continuing training from checkpoint, will skip to saved global_step")
      logger.info("  Continuing training from epoch %d", epochs_trained)
      logger.info("  Continuing training from global step %d", global_step)
      logger.info("  Will skip the first %d steps in the epoch", steps_trained_in_current_epoch)
    except ValueError:
      global_step = 0
      logger.info("  Starting fine-tuning.")

  tr_loss, logging_loss = 0.0, 0.0
  model.zero_grad()
  train_iterator = trange(
    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
  )
  set_seed(args)  # Added here for reproductibility

  for epoch in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):

      # Skip past any already trained steps if resuming training
      if steps_trained_in_current_epoch > 0:
        steps_trained_in_current_epoch -= 1
        continue

      ## Training step
      model.train()
      batch = {key: input_tensor.to(args.device) for key, input_tensor in batch.items()}
      inputs = {k: v for k, v in batch.items() if k != "labels"}

      outputs = model(**inputs)
      logits, labels = outputs[0], batch["labels"]  # model outputs are always tuple in transformers (see doc)
      m_logits = logits_masked(logits, labels, args.task_name)
      loss = F.cross_entropy(m_logits, labels)
      if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
      
      if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
          scaled_loss.backward()
      else:
        loss.backward()
      tr_loss += loss.item()

      ## Post processing
      if (step + 1) % args.gradient_accumulation_steps == 0 or (
        # last step in epoch but step is always smaller than gradient_accumulation_steps
        len(epoch_iterator) <= args.gradient_accumulation_steps
        and (step + 1) == len(epoch_iterator)
      ):
        if args.fp16:
          torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1

        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
          logs = {}
          if (
            args.local_rank == -1 and args.evaluate_during_training
          ):  # Only evaluate when single GPU otherwise metrics may not average well
            results = evaluate(model, tokenizer, processor, "dev", args)
            for key, value in results.items():
              eval_key = "eval_{}".format(key)
              logs[eval_key] = value

          loss_scalar = (tr_loss - logging_loss) / args.logging_steps
          learning_rate_scalar = scheduler.get_lr()[0]
          logs["learning_rate"] = learning_rate_scalar
          logs["loss"] = loss_scalar
          logging_loss = tr_loss

          for key, value in logs.items():
            tb_writer.add_scalar(key, value, global_step)
          print(json.dumps({**logs, **{"step": global_step}}))

        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
          save(model, optimizer, scheduler, global_step, args)
      if args.max_steps > 0 and global_step > args.max_steps:
        epoch_iterator.close()
        break
    if args.max_steps > 0 and global_step > args.max_steps:
      train_iterator.close()
      break

  if args.local_rank in [-1, 0]:
    tb_writer.close()

  return global_step, tr_loss / global_step

def save(model, optimizer, scheduler, global_step, args):
  # Save model checkpoint
  output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  model_to_save = (
    model.module if hasattr(model, "module") else model
  )  # Take care of distributed/parallel training
  model_to_save.save_pretrained(output_dir)

  torch.save(args, os.path.join(output_dir, "training_args.bin"))
  logger.info("Saving model checkpoint to %s", output_dir)

  torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
  torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
  logger.info("Saving optimizer and scheduler states to %s", output_dir)

def evaluate(model, tokenizer, processor, mode, args, prefix=""):
  eval_output_dir = args.output_dir

  results = {}
  eval_dataset = load_and_cache_examples(tokenizer, mode, args)

  if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
    os.makedirs(eval_output_dir)

  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
  # Note that DistributedSampler samples randomly
  eval_sampler = SequentialSampler(eval_dataset)
  eval_dataloader = DataLoader(
    eval_dataset, 
    sampler=eval_sampler, 
    batch_size=args.eval_batch_size, 
    collate_fn=processor.collate(return_token_type_ids=False))

  # multi-gpu eval
  if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)

  # Eval!
  logger.info("***** Running {} evaluation {} *****".format(mode, prefix))
  logger.info("  Num examples = %d", len(eval_dataset))
  logger.info("  Batch size = %d", args.eval_batch_size)
  eval_loss = 0.0
  nb_eval_steps = 0
  preds = None
  out_label_ids = None
  for batch in tqdm(eval_dataloader, desc="Evaluating"):
    model.eval()
    batch = {key: input_tensor.to(args.device) for key, input_tensor in batch.items()}
    inputs = {k: v for k,v in batch.items() if k != "labels"}
    with torch.no_grad():
      outputs = model(**inputs)
      logits, labels = outputs[0], batch["labels"]  # model outputs are always tuple in transformers (see doc)
      m_logits = logits_masked(logits, labels, args.task_name)
      tmp_eval_loss = F.cross_entropy(m_logits, labels)
      eval_loss += tmp_eval_loss.mean().item()
    
    inputs["labels"] = labels # For the segment of code below
    nb_eval_steps += 1
    if preds is None:
      preds = logits.detach().cpu().numpy()
      out_label_ids = inputs["labels"].detach().cpu().numpy()
    else:
      preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
      out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

  eval_loss = eval_loss / nb_eval_steps
  preds = np.argmax(preds, axis=1)
  result = compute_metrics(preds, out_label_ids, processors[args.task_name].get_labels())
  results.update(result)


  output_eval_file = os.path.join(eval_output_dir, prefix, "{}_results.txt".format(mode))
  with open(output_eval_file, "w") as writer:
    logger.info("***** {} results {} *****".format(mode, prefix))
    for key in sorted(result.keys()):
      logger.info("  %s = %s", key, str(result[key]))
      writer.write("%s = %s\n" % (key, str(result[key])))

  return results

"""
  mode is one of "train", "dev", "test"
"""
def load_and_cache_examples(tokenizer, mode, args):
  task = args.task_name
  if args.local_rank not in [-1, 0] and mode == "train":
    torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

  processor = processors[task]
  cached_features_file = os.path.join(
    args.data_dir,
    "cached_{}_{}_{}_{}".format(
      mode,
      list(filter(None, args.model_path.split("/"))).pop(),
      str(args.max_seq_length),
      str(task),
    ),
  )

  if os.path.exists(cached_features_file) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s", cached_features_file)
    dataset = PsychDataset.load(cached_features_file)
  else:
    logger.info("Creating features from dataset file at %s", args.data_dir)
    dataset = processor.get_examples(tokenizer, args.data_dir, mode)
    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file %s", cached_features_file)
      PsychDataset.save(dataset, cached_features_file)

  dataset = processor.convert_examples_to_features(dataset, tokenizer)
  if args.local_rank == 0 and mode == "train":
    torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
  return dataset


def main():
  # used from arguments.py
  args = argparser().parse_args()
  if (
    os.path.exists(args.output_dir)
    and os.listdir(args.output_dir)
    and args.do_train
    and not args.overwrite_output_dir
  ):
    raise ValueError(
      "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
        args.output_dir
      )
    )

  # Setup distant debugging if needed
  if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd

    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

  # Setup CUDA, GPU & distributed training
  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() and not args.no_cuda else 0
  else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
  args.device = device

  # Setup logging
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
  )
  logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    args.local_rank,
    device,
    args.n_gpu,
    bool(args.local_rank != -1),
    args.fp16,
  )

  # Set seed
  set_seed(args)

  # Prepare our task
  args.task_name = args.task_name.lower()
  if args.task_name not in processors:
    raise ValueError("Task not found: %s" % (args.task_name))
  processor = processors[args.task_name]

  label_list = processor.get_labels()
  num_labels = len(label_list)

  # Load pretrained model and tokenizer
  if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

  args.model_type = args.model_type.lower()
  config = AutoConfig.from_pretrained(
    args.config_name if args.config_name else args.model_path,
    num_labels=num_labels,
    finetuning_task=args.task_name,
    cache_dir=args.cache_dir if args.cache_dir else None
  )
  tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  model = AutoModelForSequenceClassification.from_pretrained(
    args.model_path,
    config=config,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )

  add_special_tokens(model, tokenizer, processor)

  if args.local_rank == 0:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
  model.to(args.device)

  logger.info("Training/evaluation parameters %s", args)
  
  # Training
  if args.do_train:
    train_dataset = load_and_cache_examples(tokenizer, "train", args)
    global_step, tr_loss = train(model, tokenizer, train_dataset, processor, args)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

  # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
  if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
      model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Load a trained model and vocabulary that you have fine-tuned
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model.to(args.device)

  # Evaluation
  assert not (args.do_test and args.do_eval)
  results = {}
  if (args.do_eval or args.do_test) and args.local_rank in [-1, 0]:
    mode = "dev" if args.do_eval else "test"
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
      checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
      )
      logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate(%s) the following checkpoints: %s", mode, checkpoints)
    for checkpoint in checkpoints:
      logger.info("Checkpoint: %s", checkpoint)
      global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
      prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

      model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
      model.to(args.device)
      result = evaluate(model, tokenizer, processor, mode, args, prefix=prefix)
      result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
      results.update(result)
  return results

if __name__ == "__main__":
  main()
