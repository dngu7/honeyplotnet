# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import torch

import os
import lmdb
import pickle
from random import randint

from .base import BaseDataset, get_text_window

class PmcCaptionPreDataset(BaseDataset):
  def __init__(self, db_path, tokenizer, 
    tgt_token='<mask_1>', delete_flag='<delete>', 
    window_size=5, widen_rate=0.0, max_widen_len=0, 
    min_sent_len=10, max_source_len=1024, 
    max_target_len=1024, ignore_pad_token_for_loss=True, 
    pad_to_max_len=True, **kwargs
    ):
    super().__init__(tokenizer, window_size, widen_rate, max_widen_len, 
               min_sent_len, max_source_len, max_target_len, pad_to_max_len, 
               ignore_pad_token_for_loss, tgt_token=tgt_token)

    self.db_path = db_path
    self.delete_flag = delete_flag
    self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                          readonly=True, lock=False,
                          readahead=False, meminit=False)
    with self.env.begin(write=False) as txn:
      self.length = pickle.loads(txn.get(b'__len__'))
      self.keys = pickle.loads(txn.get(b'__keys__'))

  def __len__(self):
    return self.length - 1

  def get_id(self, index):
    with self.env.begin(write=False, buffers=True) as txn:
      byteflow = txn.get(self.keys[index])
      data = pickle.loads(byteflow)
    return data['pmc_id']

  def __getitem__(self, index):
    with self.env.begin(write=False, buffers=True) as txn:
      byteflow = txn.get(self.keys[index])
      data = pickle.loads(byteflow)
    
    pmc_id       = data['pmc_id']
    all_captions = data['fig_captions']
    all_indices  = data['fig_index']
    all_text     = data['all_text']
    fig_ids      = list(data['fig_ids'])
    
    #Randomly pick any figure
    fig_idx = randint(0, len(fig_ids)-1)
    fig_id  = fig_ids[fig_idx]
    #Obtain caption and indices for that figure
    captions      = all_captions[fig_id]
    context_start = sorted(all_indices[fig_id])[0]

    all_text = self.preprocess_all_text(all_text)
    #all_text, length = remove_doc_class(all_text)
    #context_start -= length

    caption_label, all_text, context_start = self.get_caption_label(
        captions, all_text, context_start)

    context = get_text_window(
      all_text, context_start, 
      tgt_token=self.tgt_token,
      window_size=self.window_size, 
      widen_rate=self.widen_rate, 
      max_widen_len=self.max_widen_len, 
      min_sent_len=self.min_sent_len)
    
    model_inputs = self.preprocessing(context, caption_label)
    return model_inputs

  def preprocessing(self, inputs, targets):  

    model_inputs, _ = self._tokenize(self.tokenizer, 
        inputs, max_source_len=self.max_source_len, 
        max_target_len=self.max_target_len, is_target=False)

    # Setup the tokenizer for targets
    with self.tokenizer.as_target_tokenizer():
      labels = self.tokenizer(targets, max_length=self.max_target_len, padding=self.padding, truncation=True)

    if self.padding == "max_length" and self.ignore_pad_token_for_loss:
      labels["input_ids"] = [
          l if l != self.tokenizer.pad_token_id else -100 for l in labels["input_ids"]
      ]
    
    model_inputs["labels"] = torch.tensor(labels["input_ids"])

    for k in list(model_inputs.keys()):
      model_inputs[k] = model_inputs[k].squeeze(0)

    return model_inputs

  