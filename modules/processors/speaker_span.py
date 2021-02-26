import sys
import os
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer
from text_processor import lru_encode

def main(tokenizer_name, files):
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
  tokenizer.add_special_tokens({'additional_special_tokens': ['<T>', '</T>', '<P>', '</P>', '<U>', '</U>']})

  max_length = 512
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
        current_context, length = [], 6 # 4 span + 2
        utterance = example['options-for-correct-answers'][0]
        label = f"{utterance['speaker']}_" + utterance['agg_label']
        utterance = f"<U><{utterance['speaker']}>" + utterance['utterance'] + f"</{utterance['speaker']}></U>"
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
          context_utterance = f"<{context_message['speaker']}>" + context_message['utterance'] + f"</{context_message['speaker']}>" 
          length += len(lru_tokenize(context_utterance)) + 1
          if length > max_length:
            break
          current_context.insert(0, context_utterance)
        if len(current_context) > 0:
          dataset['context'].append(f"{tokenizer.sep_token}".join(current_context))
        else:
          continue
        dataset['utterance'].append(utterance)
        dataset['label'].append(label)
    pd.DataFrame(dataset).to_csv(f"generated_data/{tokenizer_name}/speaker_span_{file.split('/')[-1].split('.')[0]}.csv", index=False, sep='ך')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
