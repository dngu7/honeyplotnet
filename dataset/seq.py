# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import torch
from .base import stack_dict
from .continuous import PmcContinuousDataset

class PmcSeqDataset(PmcContinuousDataset):
  def __init__(self, data,
                tokenizer, 
                active_tasks=None, 
                active_charts=None,
                scale_mode='log10', 
                scale_eps=1.00001, 
                scale_floor=-1.0, 
                norm_mode='minmax', 
                debug=False, 
                **kwargs):
    super().__init__(data=data, active_tasks=active_tasks, active_charts=active_charts, 
      scale_mode=scale_mode, scale_eps=scale_eps, scale_floor=scale_floor, norm_mode=norm_mode, debug=debug)

    self.tokenizer = tokenizer

  def __getitem__(self, index):
      
    while True:
      d = self.data[index % len(self.data)]

      outputs = {}
      outputs['chart_type'] = self.get_chart_type(d)
      outputs['chart_data'] = self.preprocess_data_series(d)

      captions = self.preprocess_captions(d['captions'])
      outputs['captions'] = self.tokenize(captions)
      outputs['captions']['text'] = captions
      
      if self.perform_checks(outputs):
        break
      else:
        index += 1
        
    return index, outputs


  def collate_fn(self, list_batch):
    list_idx = [b[0] for b in list_batch]
    list_batch = [b[1] for b in list_batch]
    collated_batch = stack_dict(list_batch)
    
    collated_batch['chart_data'] = self.collate_data(collated_batch['chart_data'])
    collated_batch['captions']   = self.collate_captions(collated_batch['captions'])
    
    return list_idx, collated_batch
  
  def preprocess_captions(self, captions):
    caption_str = ''
    if 'title' in captions:
      caption_str += ' '.join(captions['title'])
    
    for k, str_list in captions.items():
      if k != 'title':
        caption_str += ' '.join(str_list)
    return caption_str

  def tokenize(self, text, max_source_len=1024):
    inputs = self.tokenizer(
      text, max_length=max_source_len, 
        padding="max_length", truncation=True, return_tensors="pt")
    return inputs
  
  def collate_captions(self, caption_batch):
    output = {'text': [], 'input_ids': [], 'attention_mask': []}
    
    for cap in caption_batch:
      for k, item in cap.items():
        output[k].append(item)

    for k in ['input_ids', 'attention_mask']:
      output[k] = torch.cat(output[k], dim=0)

    return output