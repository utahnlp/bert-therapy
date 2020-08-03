from sklearn.metrics import f1_score
from processors.speaker_context_processor import SpeakerContextProcessor
from processors.speaker_span_context_processor import SpeakerSpanContextProcessor
from processors.future_span_context_processor import FutureSpanContextProcessor
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import dataclasses
import random
import numpy as np

logger = logging.getLogger(__name__)

# https://github.com/utahnlp/therapist-observer/blob/3ed5332df1b2d761868e2519da1099834c963c90/tensorflow/psyc_utils.py
MISC11_P_labels = ["change_talk","sustain_talk","follow_neutral"]
MISC11_T_labels = ["facilitate","reflection_simple","reflection_complex","giving_info","question_closed","question_open","MI_adherent","MI_non-adherent"]

MISC11_BRIEF_T_labels = ["FA","RES","REC","GI","QUC","QUO","MIA","MIN"]
MISC11_BRIEF_P_labels = ["POS","NEG","FN"]

# https://forums.fast.ai/t/focalloss-with-multi-class/35588/3  
def one_hot_embedding(labels, num_classes):
    return (torch.eye(num_classes)[labels]).to(labels.device)

class FocalLoss(nn.Module):
  def __init__(self, alpha, gamma=0, eps=1e-7):
    super(FocalLoss, self).__init__()
    self.alpha = torch.FloatTensor(alpha)
    self.gamma = gamma
    self.eps = eps

  def forward(self, logits, labels):
    y = one_hot_embedding(labels, logits.size(-1))
    probs= F.softmax(logits, dim=-1)
    probs = probs.clamp(self.eps, 1. - self.eps)
    alphas = self.alpha[labels].unsqueeze(1).to(labels.device)
    loss = -1 * alphas * y * torch.log(probs) # alpha-weighted cross entropy
    loss = loss * (1 - probs) ** self.gamma # focal loss
    return loss.sum(dim=1).mean()

processors = {
  "categorize-both-9": SpeakerContextProcessor(
    {"P": MISC11_P_labels, 
     "T": MISC11_T_labels}, 
    task="categorize", 
    agent=None,
    context_len=9),
  "categorize-span-both-9": SpeakerSpanContextProcessor(
    {"P": MISC11_P_labels, 
     "T": MISC11_T_labels}, 
    task="categorize", 
    agent=None,
    context_len=9),
  "categorize-span-both-4": SpeakerSpanContextProcessor(
    {"P": MISC11_P_labels, 
     "T": MISC11_T_labels}, 
    task="categorize", 
    agent=None,
    context_len=4),
  "categorize-future-span-both": FutureSpanContextProcessor(
    {"P": MISC11_P_labels, 
     "T": MISC11_T_labels}, 
    task="categorize", 
    agent=None,
    context_len=9,
    future=2),
  "categorize-future-span-both-4-2": FutureSpanContextProcessor(
    {"P": MISC11_P_labels, 
     "T": MISC11_T_labels}, 
    task="categorize", 
    agent=None,
    context_len=4,
    future=2)
}

"""
"categorize-therapist": SpeakerContextProcessor(
    {"P": MISC11_P_labels, 
     "T": MISC11_T_labels}, 
    task="categorize", 
    agent="therapist",
    context_len=9),
  "categorize-patient": SpeakerContextProcessor(
    {"P": MISC11_P_labels, 
      "T": MISC11_T_labels}, 
    task="categorize", 
    agent="patient",
    context_len=9),
  "categorize-span-therapist": SpeakerSpanContextProcessor(
    {"P": MISC11_P_labels, 
     "T": MISC11_T_labels}, 
    task="categorize", 
    agent="therapist",
    context_len=9),
  "categorize-span-patient": SpeakerSpanContextProcessor(
    {"P": MISC11_P_labels, 
      "T": MISC11_T_labels}, 
    task="categorize", 
    agent="patient",
    context_len=9),

"""
"""
categorize-*: 
  <s><T> right</s>
    </s><T> mm-hmm</s>
    </s> <P> my job wants me to fly down to atlanta next week and i just don’t think i’m ready for that </s>
    </s><T> mm-hmm </s>
    </s><P><U> yes 
  </s>

categorize-span:

"""

def compute_metrics(preds, labels, label_names):
  results = {"macro": f1_score(labels, preds, average="macro")}
  category_f1s = f1_score(labels, preds, average=None)
  for cat, score in zip(label_names, category_f1s):
    results[cat] = score
  return results


""" Binary multi task logits masker"""
sep = {name: processor.label_dict['_SEP'] for name, processor in processors.items()}
def logits_masked(logits, labels, task_name):
  s = sep[task_name]
  m_logits = logits.clone()
  m_logits[(labels < s).view((-1, )), s:]= -10000
  m_logits[(labels >= s).view((-1, )), :s] = -10000
  return m_logits

def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

def add_special_tokens(model, tokenizer, processor, copy_sep=True):
  """ Add special tokens to the tokenizer and the model if they have not already been added. """
  num_added_tokens = tokenizer.add_special_tokens({
    'additional_special_tokens': processor.additional_tokens
  }) # doesn't add if they are already there
  embeddings = model.resize_token_embeddings(len(tokenizer)) # doesn't mess with existing tokens
  assert(embeddings.num_embeddings == len(tokenizer))
  if copy_sep:
    for i in range(num_added_tokens):
      embeddings.weight.data[-i, :] = embeddings.weight.data[tokenizer.sep_token_id, :]