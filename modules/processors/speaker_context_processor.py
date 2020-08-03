import logging
import os
import re
from collections import defaultdict
from functools import reduce

import numpy as np
import pandas as pd
import torch
import torchtext.data as data
from transformers import RobertaTokenizer, BertTokenizer
from transformers.data.processors.utils import DataProcessor

from .text_processor import (PsychDataset, context_processor, filter_text,
                             lru_encode, utterance_processor)

logger = logging.getLogger(__name__)

class SpeakerContextProcessor(DataProcessor):

  def __init__(self, label_dict, task, context_len=9, agent=None):
    assert task == "forecast" or task == "categorize"
    assert agent == None or agent == "therapist" or agent == "patient"
    assert context_len > 0

    self.flattened_labels = reduce(lambda a,b: a+b, label_dict.values())
    self.label_dict = dict(zip(self.flattened_labels, range(len(self.flattened_labels))))
    self.label_dict['_SEP'] = len(next(iter(label_dict.values()))) 
    self.context_len = context_len
    self.task = task
    self.agent = agent
    self.additional_tokens = ["<P>", "<T>", "<U>"]
    self.additional_token_ids = {}

  def get_examples(self, tokenizer, data_dir, name):
    return self._create_examples(tokenizer, os.path.join(data_dir, "{}.json".format(name)))
  
  def get_labels(self):
    return self.flattened_labels

  def _create_examples(self, tokenizer, data_dir):
    return PsychDataset(data.TabularDataset(data_dir, 'json', 
                          {'options-for-correct-answers': ('utterance', data.RawField(utterance_processor(tokenizer, speaker=self.agent[0].upper() if self.agent else None))),
                           'messages-so-far': ('context', data.RawField(context_processor(tokenizer)))}))
  
  def convert_examples_to_features(self, dataset, tokenizer):
    empty_context = tokenizer.encode('N/A')
    examples = []

    for token in self.additional_tokens:
      token_id = tokenizer.encode(token, add_special_tokens=False)
      if len(token_id) > 1:
        raise NotImplementedError("special token is not a part of tokenizer vocabulary")
      self.additional_token_ids[token] = token_id[0]

    for (ex_index, example) in enumerate(dataset):
      if ex_index % 10000 == 0:
        logger.info("Preprocessing example %d/%d" % (ex_index, len(dataset)))
      # there's no point of this example if there's no proper utterance
      if not example['utterance']:
        continue 
      if isinstance(tokenizer, RobertaTokenizer):
        if not example['context_encoded']:
          example['context_encoded'] = empty_context # <s> N/A </s>
        else:
          # select the last c context messages and speakers
          example['context_encoded'] = (np.array(example['context_encoded'])[-self.context_len:]).tolist()
          example['context_speaker'] = (np.array(example['context_speaker'])[-self.context_len:]).tolist()
          # replace the start token with the speaker or 
          # keep the start token for the first context message
          for idx in range(len(example['context_encoded'])):
            example['context_encoded'][idx][0] = self.additional_token_ids[f"<{example['context_speaker'][idx]}>"]
            if idx == 0:
              example['context_encoded'][idx] = (
                [tokenizer.bos_token_id] + 
                example['context_encoded'][idx]
              )
          # adds seperators between messages, so we will still get <s> (first context message), 
          # </s> (from encoded representation) and </s> from reduce adding here.  
          example['context_encoded'] = reduce(lambda a,b: a + [tokenizer.sep_token_id] + b, example['context_encoded'])
        if self.task == "forecast":
          # no utterance
          example['utterance_encoded'] = []
        else:
          # add the utterance token as well, get rid of the utterance start token
          example['utterance_encoded'] = (
            [tokenizer.sep_token_id] +  
            [self.additional_token_ids[f"<{example['utterance_speaker']}>"]] + 
            [self.additional_token_ids['<U>']] + example['utterance_encoded'][1:]
          )
      elif isinstance(tokenizer, BertTokenizer):
        if not example['context_encoded']:
          example['context_encoded'] = empty_context # [CLS] no context [SEP]
        else:
          # select the last c context messages and speakers
          example['context_encoded'] = (np.array(example['context_encoded'])[-self.context_len:]).tolist()
          example['context_speaker'] = (np.array(example['context_speaker'])[-self.context_len:]).tolist()
          # replace the start token with the speaker or 
          # keep the start token for the first context message
          for idx in range(len(example['context_encoded'])):
            example['context_encoded'][idx][0] = self.additional_token_ids[f"<{example['context_speaker'][idx]}>"]
            if idx == 0:
              example['context_encoded'][idx] = (
                [tokenizer.cls_token_id] + 
                example['context_encoded'][idx]
              )
          # adds seperators between messages, so we will still get [CLS] (first context message), 
          # [SEP] from encoded representation.  
          example['context_encoded'] = reduce(lambda a,b: a + b, example['context_encoded'])
        if self.task == "forecast":
          # no utterance
          example['utterance_encoded'] = []
        else:
          # add the utterance token as well, get rid of the utterance start token
          example['utterance_encoded'] = (
            [self.additional_token_ids[f"<{example['utterance_speaker']}>"]] + 
            [self.additional_token_ids['<U>']] + example['utterance_encoded'][1:]
          )
      else:
        raise NotImplementedError(f"Haven't implemented logic for {type(tokenizer)}")
      if len(example['utterance_encoded']) + len(example['context_encoded']) > 512:
        example['context_encoded'] = example['context_encoded'][:len(example['utterance_encoded'])-512]
      if ex_index < 25:
        logger.info("\n*** Example ***")
        logger.info("utterance: %s" % tokenizer.decode(example['utterance_encoded']))
        logger.info("context:  %s" % tokenizer.decode(example['context_encoded']))
        logger.info("guid: %s" % (example['utterance_uid']))
        logger.info("label: %s (id = %d)" % (
          example['utterance_label'], 
          self.label_dict[example['utterance_label']]
        ))

      example['utterance_label'] = self.label_dict[example['utterance_label']]
      examples.append(example)
    return PsychDataset(examples)

  """
    This method is used to convert the sparse batch into tensors which will be fed to the model.
  """
  @staticmethod
  def collate(return_token_type_ids=False):
    def _closure(batch):
      lengths = [len(example["context_encoded"]) + len(example["utterance_encoded"]) for example in batch]
      input_ids = torch.zeros([len(batch), max(lengths)], dtype=torch.long)
      attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
      if return_token_type_ids:
          token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

      for idx, example in enumerate(batch):
          input_ids[idx, :lengths[idx]] = torch.LongTensor(example["context_encoded"] + example["utterance_encoded"])
          attention_mask[idx, :lengths[idx]] = 1
          if return_token_type_ids:
              token_type_ids[idx, lengths[idx]-len(example["utterance_encoded"]): lengths[idx]] = 1
      labels = torch.LongTensor([seq["utterance_label"] for seq in batch])
      ret_dict = {
          "input_ids": input_ids,
          "attention_mask": attention_mask,
          "labels": labels
      }
      if return_token_type_ids:
          ret_dict["token_type_ids"] = token_type_ids
      return ret_dict
    return _closure

  # def generate_dialogue_attention_mask(self, batch):
  # mask = -10000 * torch.ones((batch.shape[0], batch.shape[1], batch.shape[1])) # 12 heads is fixed
  # for i in np.arange(batch.shape[0]):
  #   example_special_idx = torch.nonzero(sum(batch[i] == t for t in (set.union(symbol_dict['SPECIAL_START_TOKEN_IDS'], symbol_dict['SPECIAL_END_TOKEN_IDS'])))).flatten().tolist()
  #   last_idx = None
  #   for idx, token_id in enumerate(batch[i].tolist()):
  #     if token_id == symbol_dict['PAD_TOKEN_ID']:
  #       break
  #     if token_id == symbol_dict['BOS_TOKEN_ID'] or token_id == symbol_dict['EOS_TOKEN_ID']:
  #       mask[i, idx, example_special_idx] = 0 # attend to other special tokens
  #       mask[i, example_special_idx, idx] = 0 # let other special tokens attend to this
  #       mask[i, idx, idx] = 0 # attend to self
  #       if token_id == symbol_dict['EOS_TOKEN_ID']:
  #         mask[i, idx, 0] = 0 # eos attends to bos
  #         mask[i, 0, idx] = 0 # bos attends to eos
  #     elif token_id in symbol_dict['SPECIAL_START_TOKEN_IDS']:
  #        mask[i, idx, example_special_idx] = 0 # attend to other special tokens
  #        last_idx = idx
  #     elif token_id in symbol_dict['SPECIAL_END_TOKEN_IDS']:
  #        mask[i, idx, example_special_idx] = 0
  #        span_range = np.arange(last_idx, idx+1) # starts from the last opening special token to including this special token
  #        x, y = np.meshgrid(span_range, span_range)
  #        x, y = x.flatten(), y.flatten()
  #        span_product = np.array(list(zip(x, y))) # 2-D array
  #        mask[i, span_product[:, 0], span_product[:, 1]] = 0
  # # because we use multi-headed attention
  # return mask
