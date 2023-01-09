# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import numpy as np
import torch
import random

from .constants import TEXT_CUTOFF_POINTS

def get_text_window(all_text, context_start, tgt_token='[MASK1]', window_size=10, widen_rate=0.5, max_widen_len=2, min_sent_len=10):
  '''
  window_size        : measured as number of full sentences.
  min_sent_len       : minimum size of sentences in characters
  widen_rate         : rate that window size increases or decreases randomly by 1
  '''
  assert widen_rate >= 0.0 and widen_rate <= 1.0
  assert max_widen_len >= 0
    
  sentence_split = []
  text_len = 0
  context_sent_idx = None
  for s in all_text.split('. '):
    sent_len = len(s)
    text_len += sent_len
    
    if sent_len >= min_sent_len:
      sentence_split.append(s.strip())
    
    #Track when context starts within sentences
    if text_len >= context_start and context_sent_idx is None:
      context_sent_idx = len(sentence_split)
      sentence_split.append(tgt_token)
      
  if len(all_text) >= context_start or context_sent_idx is None:
    #Random spot in middle
    mid_pt = len(sentence_split) // 2
    sentence_split.insert(mid_pt, tgt_token)
    context_sent_idx = mid_pt #sum([len(s) for s in sentence_split[:mid_pt]])

  #Calculate window size flexibly
  if random.random() < widen_rate:
    #increase or decrease ?
    inc_dec = 1 if random.random() < 0.5 else -1
    #randomly pick a change size
    change_size = random.randint(0, max_widen_len)
    window_size += (inc_dec * change_size)

  start_idx = context_sent_idx - int(np.ceil(window_size / 2))
  end_idx = context_sent_idx + window_size // 2
  
  #print(window_size, start_idx, end_idx)
  if start_idx < 0:
    end_idx -= start_idx
    start_idx = 0
  elif end_idx > len(sentence_split):
    start_idx = start_idx - (end_idx - len(sentence_split))
    end_idx = len(sentence_split)
  #ensure limits
  start_idx = max(start_idx, 0)
  end_idx = min(end_idx, len(sentence_split))
  
  return '. '.join(sentence_split[start_idx:end_idx])

def stack_dict(list_batch):
    collated_batch = {}
    for batch in list_batch:
      if batch is None: continue
      for name, data in batch.items():
          if name not in collated_batch:
              collated_batch[name] = []
          collated_batch[name].append(data)
    return collated_batch

def stack_tensor_dict(dict_tensors):
    for k in list(dict_tensors.keys()):
      dict_tensors = torch.stack(dict_tensors[k], dim=0)
    return dict_tensors

def shift_right_vec(x, start_values=0.0):

    x_len = len(x.shape)
    one_frame = [1] * x_len
    if x_len == 3:
      one_frame[0] = x.size(0)
      one_frame[-1] = x.size(-1)
    elif x_len == 2:
      one_frame[0] = x.size(0)
    elif x_len == 1:
      pass
    else:
      raise ValueError("Input has invalid shape")

    start_seq = torch.ones(one_frame, device=x.device, dtype=torch.float32)
    start_seq = start_seq * start_values

    if x_len == 3:
      return torch.cat([start_seq, x[:,:-1, :]], dim=1)
    elif x_len == 2:
      return torch.cat([start_seq, x[:, :-1]], dim=1)
    elif x_len == 1:
      return torch.cat([start_seq, x[:-1]], dim=0)
    else:
      raise NotImplementedError()

def shift_tokens_right(input_ids, pad_token_id, dim=-1):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=dim) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens
  
class BaseDataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer, window_size, widen_rate, max_widen_len, 
               min_sent_len, max_source_len, max_target_len, 
               pad_to_max_len, ignore_pad_token_for_loss, tgt_token='<mask_1>'):
    super().__init__()

    if isinstance(tokenizer, dict) and 'caption' in tokenizer:
      tokenizer = tokenizer['caption']

    self.tokenizer = tokenizer
    self.window_size = window_size
    self.widen_rate = widen_rate
    self.max_widen_len = max_widen_len
    self.min_sent_len = min_sent_len
    self.max_source_len = max_source_len
    self.max_target_len = max_target_len
    self.pad_to_max_len = pad_to_max_len
    
    self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
    self.padding = "max_length" if pad_to_max_len else False
    self.tgt_token = tgt_token


  def preprocess_all_text(self, all_text, min_cutoff_rate=0.8):
    #Remove text beyond the conclusion. 
    #Method: Find the first mention of a list of words. Remove all text beyond the first mention.

    text_len = len(all_text)
    min_cutoff_idx = text_len
    for text in TEXT_CUTOFF_POINTS:
      cutoff_idx = all_text.lower().find(text.lower())

      if cutoff_idx >= (text_len * min_cutoff_rate):

        if cutoff_idx < min_cutoff_idx:
          min_cutoff_idx = cutoff_idx
    
    all_text = all_text[:min_cutoff_idx]
    assert len(all_text) >= (text_len * min_cutoff_rate), "bug found: text >> {}".format(all_text)
    return all_text

  def get_caption_label(self, captions, all_text, context_start):
    caption_label = ''
    for key in ['title','p']:
      cap = captions.get(key)
      if isinstance(cap, list):
        for c in cap:
          #Remove location of context
          all_text = all_text.replace(c, '')
          context_start -= len(c)
            
          # split caption into senteces and remove parts that have \\
          c = '. '.join([s.replace('\n', ' ') for s in c.strip().split('. ')])
          caption_label += c 
      elif cap is None:
        continue
      else:
        raise cap
    return caption_label, all_text, context_start
  
  def preprocess_captions(self, inputs, targets):  

    model_inputs, _ = self._tokenize(self.tokenizer, 
      inputs, max_source_len=self.max_source_len, 
      max_target_len=self.max_target_len, is_target=False)

    # Setup the tokenizer for targets
    with self.tokenizer.as_target_tokenizer():
      labels = self.tokenizer(targets, max_length=self.max_target_len, padding=self.padding, truncation=True)

    model_inputs["decoder_input_ids"] = shift_tokens_right(torch.tensor(labels["input_ids"]), self.tokenizer.pad_token_id)
    
    if self.padding == "max_length" and self.ignore_pad_token_for_loss:
      labels["input_ids"] = [
          l if l != self.tokenizer.pad_token_id else -100 for l in labels["input_ids"]
      ]
    
    model_inputs["labels"] = torch.tensor(labels["input_ids"])

    for k in list(model_inputs.keys()):
      model_inputs[k] = model_inputs[k].squeeze(0)
    
    model_inputs['raw'] = targets
    return model_inputs
  
  def _tokenize(self, tokenizer, text, max_source_len, max_target_len=32, is_target=False, shift_right=False):
    
    inputs = tokenizer(
      text, max_length=max_source_len, 
      padding=self.padding, truncation=True, return_tensors="pt")
    
    labels = None
    if is_target:
      with tokenizer.as_target_tokenizer():
        labels = tokenizer(text, max_length=max_target_len, padding=self.padding, truncation=True)

        if shift_right:
          inputs["decoder_input_ids"] = shift_tokens_right(torch.tensor(labels["input_ids"]).long(), tokenizer.pad_token_id, dim=-1)
          inputs["decoder_attention_mask"] = shift_tokens_right(torch.tensor(labels["attention_mask"]).long(), tokenizer.pad_token_id, dim=-1)

      if self.padding == "max_length" and self.ignore_pad_token_for_loss:

        labels["input_ids"] = [
            l if l != tokenizer.pad_token_id else -100 for l in labels["input_ids"]
        ]
        labels["input_ids"] = torch.tensor(labels["input_ids"]).long()
        
    return inputs, labels
