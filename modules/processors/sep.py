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
    dataset = {'context': [], 'utterance': [], 'label': []}
    print(f'Generating csv for file: {file}')
    with open(file) as json_file:
      for (idx, line) in enumerate(json_file):
        if idx % 10000 == 0:
          print(f"Preprocessing example {idx}")
        example = json.loads(line)
      
        # for the start and end tokens
        current_context, length = [], 4
        utterance = example['options-for-correct-answers'][0]
        label = f"{utterance['speaker']}_" + utterance['agg_label']
        utterance = utterance['utterance']
        length += len(lru_tokenize(utterance))
        if length > max_length:
          continue # skip examples where the utterance crosses the max length

        context = example['messages-so-far']
        context.reverse()

        for context_message in context:
          if len(current_context) > max_context_length:
            break
          if context_message['turn_number'] == -1:
            continue
          context_utterance = context_message['utterance']
          length += len(lru_tokenize(context_utterance)) + 1 # sep token
          if length > max_length:
            break
          current_context.insert(0, context_utterance)
        if len(current_context) > 0:
          dataset['context'].append(f'{tokenizer.sep_token}'.join(current_context))
        else:
          continue
        dataset['utterance'].append(utterance)
        dataset['label'].append(label)
    pd.DataFrame(dataset).to_csv(f"generated_data/{tokenizer_name}/sep/{file.split('/')[-1].split('.')[0]}.csv", index=False, sep='ך')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
