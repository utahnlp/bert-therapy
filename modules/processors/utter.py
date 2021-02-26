import sys
import os
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer
from text_processor import lru_encode

def main(tokenizer_name, files):
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
  max_length = tokenizer.model_max_length
  max_context_length = 8
  lru_tokenize=lru_encode(tokenizer)

  for file in files:
    dataset = {'utterance': [], 'label': []}
    print(f'Generating csv for file: {file}')
    with open(file) as json_file:
      for (idx, line) in enumerate(json_file):
        if idx % 10000 == 0:
          print(f"Preprocessing example {idx}")
        example = json.loads(line)
      
        # for the start and end tokens
        length = 2
        utterance = example['options-for-correct-answers'][0]
        label = f"{utterance['speaker']}_" + utterance['agg_label']
        utterance = utterance['utterance']
        length += len(lru_tokenize(utterance))
        if length > max_length:
          continue # skip examples where the utterance crosses the max length
        dataset['utterance'].append(utterance)
        dataset['label'].append(label)
    pd.DataFrame(dataset).to_csv(f"generated_data/{tokenizer_name}/utter/{file.split('/')[-1].split('.')[0]}.csv", index=False, sep='ך')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
