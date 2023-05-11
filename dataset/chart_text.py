# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import random
import torch

from utils import TASK2PREPEND
from .data import PmcDataDataset
from .base import get_text_window

class PmcChartTextDataset(PmcDataDataset):
  def __init__(self, **kwargs):
    super().__init__( **kwargs)

    # chart_text_output: controls which task is being trained. 
    assert self.chart_text_output is not None or len(self.chart_text_output) > 0, "Must have atleast one enabled."
    self.data = self.filter_data(self.data)
    

  def __getitem__(self, index):
      
    while True:
      d = self.data[index % len(self.data)]

      try:
        fig_id = d['fig_id']
        context_start = d['fig_index'][fig_id][0]
      except:
        self.del_list.append(index)
        index += 1
        continue
      
      all_text = d['all_text']
      captions = d['captions']

      all_text = self.preprocess_all_text(all_text)

      caption_label, all_text, context_start = self.get_caption_label(
        captions, all_text, context_start)
      
      ###################### 
      # Prepare targets (randomly pick a task)
      #####################
      inputs = {}
      outputs = {}

      if 'caption' in self.chart_text_output:
        _, outputs['caption'] = self._tokenize(self.tokenizer, 
        caption_label, max_source_len=self.max_source_len, 
        max_target_len=self.max_target_len, is_target=True)

      series_name = self.get_series_names(d)

      if len(series_name) and 'series' in self.chart_text_output:
        series_name = self.sep_token.join(series_name)
        outputs['series_name'] = self.tokenize_tgt_flatten(series_name)

      categorical_data = self.get_categorical_values(d)

      if len(categorical_data) > 0 and 'categorical' in self.chart_text_output:
        categorical_data = self.sep_token.join(categorical_data)
        outputs['categorical'] = self.tokenize_tgt_flatten(categorical_data)

      axis_data = self.get_axis_names(d)

      if len(axis_data) > 0 and 'axis' in self.chart_text_output:
        axis_data = self.sep_token.join(axis_data)
        outputs['axis'] = self.tokenize_tgt_flatten(axis_data)

      #Pick a task randomly
      task = random.choice(list(outputs.keys()))

      ######################
      # Prepare inputs (context)
      ######################    

      #Tokenize context (document text or caption)
      if task == 'caption':
        context = get_text_window(
          all_text, context_start, 
          tgt_token=self.tgt_token,
          window_size=self.window_size, 
          widen_rate=self.widen_rate, 
          max_widen_len=self.max_widen_len, 
          min_sent_len=self.min_sent_len)
        
        #Prepend task to document context
        context = TASK2PREPEND[task] + context
      else:
        #Prepend task to caption
        context = TASK2PREPEND[task] + caption_label

      inputs, _ = self._tokenize(self.tokenizer, 
        context, max_source_len=self.max_source_len, 
        max_target_len=self.max_target_len, is_target=False)
      
      #Add a task
      inputs['labels'] = outputs[task]['input_ids']

      for k in list(inputs.keys()):
        inputs[k] = inputs[k].squeeze(0)
      
      return inputs

  def filter_data(self, data):
    new_data = []

    if 'caption' in self.chart_text_output:
      return data
    
    #Ensure a balanced number of every data type
    for d in data:
      series_name = self.get_series_names(d)
      categorical_data = self.get_categorical_values(d)
      axis_data = self.get_axis_names(d)
      if (len(series_name) and 'series' in self.chart_text_output) \
        or (len(categorical_data) and 'categorical' in self.chart_text_output) \
           or (len(axis_data) and 'axis' in self.chart_text_output):
        new_data.append(d)
    return new_data

  def tokenize_tgt_flatten(self, text, flatten=False):
    #list of integers. 
    with self.tokenizer1.as_target_tokenizer():
      tokens = self.tokenizer1(text, max_length=self.max_target_len, padding="max_length", truncation=True)

    input_ids = [l if l != self.tokenizer1.pad_token_id else -100 for l in tokens['input_ids']]

    if flatten:
      input_ids = torch.tensor([item for sublist in input_ids for item in sublist], dtype=torch.long)
    else:
      input_ids = torch.tensor(input_ids, dtype=torch.long)

    output = {}
    output['input_ids'] = input_ids
    return output
  
  def get_axis_names(self, d):
    axis_title_ids = []
    for b in d['task3']['output']['text_roles']:
      if b['role'] == 'axis_title':
        axis_title_ids.append(b['id'])

    axis_title_text = []
    for b in d['task2']['output']['text_blocks']:
      if b['id'] in axis_title_ids:
        if 'unnamed' not in b['text']:
          axis_title_text.append(b['text'])
    
    return axis_title_text

  def get_series_names(self, d):
    all_series_name = []
    for _, series_data in enumerate(d['task6']['output']['data series']):
      series_name = None
      if 'unnamed' not in series_data['name']:
        series_name = series_data['name']
      if series_name is not None:
        v = series_name.replace('\n',' ').strip()
        if len(v):
          all_series_name.append(v)
    return all_series_name
  
  def get_categorical_values(self, d):

    all_categorical_data = []

    #Categorical data is repeated. Only go through first series
    series_data = d['task6']['output']['data series'][0]

    for _, data in enumerate(series_data['data']):
      for _, (_, v) in enumerate(data.items()):
        if type(v) == str:
          cat_name = None
          if 'unnamed' not in v:
            cat_name = v
          if cat_name is not None:
            cat_name = cat_name.replace('\n',' ').strip()
            if len(cat_name):
              all_categorical_data.append(cat_name)
    return all_categorical_data
