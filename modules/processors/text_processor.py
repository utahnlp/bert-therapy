import re
from functools import lru_cache
import torch
from torchtext.data import TabularDataset
from transformers import RobertaTokenizer, BertTokenizer

def lru_encode(tokenizer):
  @lru_cache(maxsize=100)
  def _closure(sentence):
    return tokenizer.encode(sentence)
  return _closure

@lru_cache(maxsize=100, typed=True)
def filter_text(text):
    text = re.sub(r"(\[\d*:*\d*\])", "", text) # removed timestamps
    paren_matches = re.findall(r"(\(.+?\))", text) + re.findall(r"(\[.+?\])", text)
    for match in paren_matches:
      t = match[1:-1].strip() # don't want the open and close braces
      if len(t.split()) <= 2 and t not in {'du', 'sp'}: # this is a code
        text = text.replace(match, "( {} )".format(t)) # add only the first verb lemmatized in the sequence
      else:
        text = text.replace(match, "")
    return re.sub(r'\[\]|\(\)', '', text).strip() # remove unnecessary tags and spaces

class PsychDataset(torch.utils.data.Dataset):
  def __init__(self, tabular_dataset):
    if isinstance(tabular_dataset, list):
      self.data = tabular_dataset
    elif isinstance(tabular_dataset, TabularDataset):
      self.data = [{**tabular_dataset[i].__dict__['context'], 
                    **tabular_dataset[i].__dict__['utterance']} for i in range(len(tabular_dataset))]
    else:
      raise NotImplementedError()

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    return self.data[index]
  
  def __add__(self, other):
    return self.data + other.data
  
  @staticmethod
  def load(file):
    return PsychDataset(torch.load(file))
  
  @staticmethod
  def save(dataset, directory):
    torch.save(dataset.data, directory) 


def utterance_processor(tokenizer, speaker=None):
  def _closure(x):
    x = x[0]
    utterance = filter_text(x['utterance'])
    if utterance and (speaker == None or speaker == x['speaker']):
      utterance_encoded = tokenizer.encode(utterance)
    else:
      utterance = ''
      utterance_encoded = []
    return {'utterance': utterance, 
            'utterance_encoded': utterance_encoded,
            'utterance_speaker': x['speaker'],
            'utterance_label': x['agg_label'],
            'utterance_uid': x['uid']}
  return _closure

def context_processor(tokenizer):
  encode_fn = lru_encode(tokenizer)
  def _closure(x):
    context = []
    context_encoded = []
    context_labels = []
    speaker = []

    for turn in x:
      if turn['speaker'] == 'PAD':
        continue
      turn['utterance'] = filter_text(turn['utterance'])
      if turn['utterance']: # could be empty
        context.append(turn['utterance'])
        context_labels.append(turn['agg_label'])
        context_encoded.append(encode_fn(turn['utterance']))
        speaker.append(turn['speaker'])

    return {'context': context, 
            'context_encoded': context_encoded,
            'context_labels': context_labels,
            'context_speaker': speaker}
  return _closure