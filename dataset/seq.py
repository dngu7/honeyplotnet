# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import math
import torch
import numpy as np
import random

from utils import TASK2PREPEND, CHART_TYPE_MAP, PAD_IDX

from .base import (
  get_text_window,
  stack_dict, 
  shift_right_vec
)

from .chart_text import PmcChartTextDataset

class PmcSeqDataset(PmcChartTextDataset):
  def __init__(self, data, **kwargs):
    super().__init__(data, **kwargs)

    self.tokenizer1 = kwargs['tokenizer1']

    # Filter the pmc dataset for certain tasks or charts
    self.active_tasks = ['task6'] if kwargs.get('active_tasks') is None else kwargs.get('active_tasks')
    self.active_charts = []  if kwargs.get('active_charts') is None else kwargs.get('active_charts')
    self.data = self.filter_data(data)
    
    # Settings for scaling
    assert kwargs.get('norm_mode') in ['minmax'] 
    self.norm_mode = kwargs.get('norm_mode')

    assert kwargs.get('scale_mode') in ['log10', 'log'], "scale mode not implemented"
    scale_mode = kwargs.get('scale_mode')
    self.scale_mode  = math.log10 if scale_mode == 'log10' else math.log
    self.scale_eps   = kwargs.get('scale_eps')   
    self.scale_floor = kwargs.get('scale_floor')  

    #Assignments
    self.box_plot_keys = ['min', 'first_quartile', 'median',  'third_quartile', 'max']
    self.node_map      = {'pad': PAD_IDX, 'point': 1, 'eos': 0}

  def __getitem__(self, index):

    while True:
      d = self.data[index % len(self.data)]

      outputs = {}
      outputs['chart_type'] = self.get_chart_type(d)
      outputs['data'] = self.preprocess_data_series(d)

      if self.perform_checks(outputs):
        _ = outputs.pop('chart_type')
        break
      else:
        index += 1

    ##############################################################
    all_text = d['all_text']
    captions = d['captions']

    all_text = self.preprocess_all_text(all_text)

    context_start = d['fig_index'][d['fig_id']][0]

    caption_label, all_text, context_start = self.get_caption_label(
      captions, all_text, context_start)
    
    ###################### 
    # Prepare targets (randomly pick a task)
    #####################
    inputs = {}

    if 'caption' in self.discrete_output:
      _, outputs['caption'] = self._tokenize(self.tokenizer1, 
        caption_label, max_source_len=self.max_source_len, 
        max_target_len=self.max_target_len, is_target=True)

    series_name = self.get_series_names(d)

    if len(series_name) and 'series' in self.discrete_output:
      series_name = self.sep_token.join(series_name)
      outputs['series_name'] = self.tokenize_tgt_flatten(series_name)

    categorical_data = self.get_categorical_values(d)

    if len(categorical_data) > 0 and 'categorical' in self.discrete_output:
      categorical_data = self.sep_token.join(categorical_data)
      outputs['categorical'] = self.tokenize_tgt_flatten(categorical_data)

    axis_data = self.get_axis_names(d)

    if len(axis_data) > 0 and 'axis' in self.discrete_output:
      axis_data = self.sep_token.join(axis_data)
      outputs['axis'] = self.tokenize_tgt_flatten(axis_data)

    task = random.choice(list(o for o in outputs.keys() if o in self.discrete_output))

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

    text_inputs = {}

    text_inputs, _ = self._tokenize(self.tokenizer1, 
      context, max_source_len=self.max_source_len, 
      max_target_len=self.max_target_len, is_target=False)
    
    if task == 'data':
      text_inputs['labels'] = torch.ones(self.max_target_len) * PAD_IDX
    else:
      text_inputs['labels'] = outputs[task]['input_ids']
      

    for k in list(text_inputs.keys()):
      text_inputs[k] = text_inputs[k].squeeze(0)  
        
    return index, {'task': task, 'text': text_inputs, 'data': outputs['data']}


  def collate_fn(self, list_batch):
    list_idx = [b[0] for b in list_batch]
    list_batch = [b[1] for b in list_batch]
    collated_batch = stack_dict(list_batch)
    
    collated_batch['data']   = self.collate_data(collated_batch['data'])
    collated_batch['text']   = self.collate_captions(collated_batch['text'])
    
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
    inputs = self.tokenizer1(
      text, max_length=max_source_len, 
        padding="max_length", truncation=True, return_tensors="pt")
    return inputs
  
  def collate_captions(self, caption_batch):
    output = {'text': [], 'input_ids': [], 'attention_mask': [] ,'labels': []}
    
    for cap in caption_batch:
      for k, item in cap.items():
        output[k].append(item)

    for k in ['input_ids', 'attention_mask', 'labels']:
      output[k] = torch.stack(output[k], dim=0).long()

    return output


  def __len__(self):
      return len(self.data)

  def filter_data(self, data):
    task_req = []
    if isinstance(self.active_tasks, list):
      assert 'task6' in self.active_tasks
      task_req = self.active_tasks
      data = [d for d in data if all(t in d and d[t] != None for t in task_req)]

    if len(self.active_charts):
      data = [d for d in data if d['task1']['output']['chart_type'].lower().strip() in self.active_charts]

    return data

  def perform_checks(self, outputs):
      flag = False
      if len(outputs['data']['norm_series']) == 0 or \
          len(outputs['data']['norm_series'][0]['data']) == 0 or \
            len(outputs['data']['norm_scale']) == 0:
        # Check there are scales, data and series 
        flag = False

      elif outputs['data']['chart_type'] == 'vertical box':
        flag = True
      elif outputs['data']['chart_type']   in ['line', 'scatter'] and len(outputs['data']['norm_scale']) == 2:
        #Check all data is float
        flag = all(v==float for k, v in outputs['data']['data_type'].items())

      elif outputs['data']['chart_type'] in ['vertical bar', 'horizontal bar'] and len(outputs['data']['norm_scale']) == 1:
        flag = True if outputs['data']['data_type']['x'] == str else False
      else:
        flag = False
        #raise ValueError("chart type not recognized: {}".format(outputs['data']['chart_type']))
      return flag

  def get_chart_type(self, d):
    '''Returns an index'''
    ct = d['task1']['output']['chart_type'].lower().strip()
    return CHART_TYPE_MAP.index(ct)

  def get_data_with_idx(self, index):
    if isinstance(index, list):
      return [self.data[i % len(self.data)] for i in index]
    elif isinstance(index, int):
      return self.data[index % len(self.data)]
    else:
      raise ValueError("Invalid index given: {}".format(index))

  def preprocess_captions(self, captions):
    caption_str = ''
    if 'title' in captions:
      caption_str += ' '.join(captions['title'])
    
    for k, str_list in captions.items():
      if k != 'title':
        caption_str += ' '.join(str_list)
    return caption_str

  def get_text_with_idx(self, index):
    d = self.get_data_with_idx(index)
    outputs = d['task6']['output']

    #captions
    caption = self.preprocess_captions(d['captions'])

    #Get categorical data
    data_series = outputs['data series']
    cat_names = []
    for s in data_series[0]['data']:
      for v in s.values():
        if isinstance(v, str):
          cat_names.append(v)
    
    #Get series
    series_names = []
    for s in data_series:
      series_names.append(s['name'])

    axis_title_ids = []
    for b in d['task3']['output']['text_roles']:
      if b['role'] == 'axis_title':
        axis_title_ids.append(b['id'])

    axis_title_text = []
    for b in d['task2']['output']['text_blocks']:
      if b['id'] in axis_title_ids:
        axis_title_text.append(b['text'])


    text = {}
    text['caption'] = caption
    text['categorical'] = cat_names
    text['series_name'] = series_names
    text['axis_titles'] = axis_title_text

    return text

  def batch_tokens(self, list_batch):
    dict_of_tokens = stack_dict(list_batch)
    for key in list(dict_of_tokens.keys()):
      dict_of_tokens[key] = torch.cat(dict_of_tokens[key], dim=0)
    return dict_of_tokens

  def preprocess_data_series(self, d, fake_flag=0):
    
    output = {}
    output['unnorm_series'] = []
    output['norm_series']   = []
    output['unnorm_scale']  = {}
    output['data_type']  = {}

    chart_type = d['task1']['output']['chart_type'].lower().strip()
    output['chart_type'] = chart_type

    #Setup containers to obtain scales
    for s in ['y', 'x']:
      output['unnorm_scale'][s] = {}
      output['unnorm_scale'][s]['min'] =  [float('inf')] * len(d['task6']['output']['data series'])
      output['unnorm_scale'][s]['max'] = [-float('inf')] * len(d['task6']['output']['data series'])


    '''
    First loop to collect scales and organize data from json. 
    This goes through the series. 
    '''
    for ds_idx, series_data in enumerate(d['task6']['output']['data series']):

      unnorm_series = {}
      unnorm_series['name'] = None
      unnorm_series['data'] = []

      if 'unnamed' not in series_data['name']:
        unnorm_series['name'] = series_data['name']

      prev_v = {}
      #This goes through each data point
      for pt_idx, data in enumerate(series_data['data']):
        
        series_keys = list(data.keys())

        #ignore series that do not have 'x', 'y' or 'y2'
        if chart_type not in ['vertical box'] and \
            (('x' not in series_keys) or \
              ('y' not in series_keys and \
              'y2' not in series_keys)):
          continue
        
        #Replace y2 with y where possible
        y2_replacement_flag = False
        if 'y2' in series_keys and 'y' not in series_keys and 'x' in series_keys:
          y2_replacement_flag = True

        pt_store = {}

        #This goes through each property of each data point
        for _, (k, v) in enumerate(data.items()):
          
          #Replace y2 with y where possible
          if 'y2-' in k: continue 
          if y2_replacement_flag and k == 'y2':
            k = 'y'
          
          #Determine data type of the value
          data_type = type(v)
          output['data_type'][k] = data_type
          
          #Remove spaces if string
          if data_type == str:
            pt_store[k] = v.replace('\n',' ')
          
          #Store the data AND update max, min for each series
          elif isinstance(v, (float, int)):
            
            # Assign keys to each grouping
            if k in ['y', 'y2', 'first_quartile', 'min', 'max', 'third_quartile', 'median']:
              scale_key = 'y'
            elif 'x' == k:
              scale_key = 'x'
            else:
              raise ValueError("Unsupported axis key: {}".format(k))

            #Add perturbation if fake
            if fake_flag == 1:
              v = v * random.uniform(self.perturb_min, self.perturb_max)

            if v < -self.scale_eps[0]:
              v = 0

            pt_store[k] = v
            output['unnorm_scale'][scale_key]['min'][ds_idx] = min(output['unnorm_scale'][scale_key]['min'][ds_idx], v)
            output['unnorm_scale'][scale_key]['max'][ds_idx] = max(output['unnorm_scale'][scale_key]['max'][ds_idx], v)

        if len(pt_store) > 1:
          unnorm_series['data'].append(pt_store)
          keys = list(pt_store.keys())
          assert True if 'y' in keys else 'y2' not in keys, "{}, {}, {}".format(series_keys, keys, y2_replacement_flag)
      

      output['unnorm_series'].append(unnorm_series)

    ##########################################
    #Second loop to calculate statistics on collected data and then normalise data. e.g. min max or mean std

    output['norm_scale'] = {}

    for s in ['y', 'x']:
      series_count = len(output['unnorm_scale'][s]['min'])

      min_container, max_container = [],[]
      for ds_idx in range(series_count):
        #Check the scale is being used
        if output['unnorm_scale'][s]['min'][ds_idx] < float('inf') and \
          output['unnorm_scale'][s]['max'][ds_idx] > -float('inf'): # and \
          #output['unnorm_scale'][s]['min'][ds_idx] !=  output['unnorm_scale'][s]['max'][ds_idx]:

          #Create container
          if s not in output['norm_scale']:
            output['norm_scale'][s] = {}
            for n in ['min','max','range']:
              output['norm_scale'][s][n] = [None] * series_count

          #Apply scale to the scale
          output['norm_scale'][s]['min'][ds_idx] = self.scale_mode(output['unnorm_scale'][s]['min'][ds_idx] + self.scale_eps[0])
          output['norm_scale'][s]['max'][ds_idx] = self.scale_mode(output['unnorm_scale'][s]['max'][ds_idx] + self.scale_eps[0])
          
          min_container.append(output['unnorm_scale'][s]['min'][ds_idx])
          max_container.append(output['unnorm_scale'][s]['max'][ds_idx])

          #Calculate the scale before the floor to prevent out of domain errors
          scale_range = output['unnorm_scale'][s]['max'][ds_idx] - output['unnorm_scale'][s]['min'][ds_idx]
          output['norm_scale'][s]['min'][ds_idx] = max(output['norm_scale'][s]['min'][ds_idx], self.scale_floor[0])

          output['norm_scale'][s]['range'][ds_idx] = self.scale_mode(scale_range + self.scale_eps[1])
          output['norm_scale'][s]['range'][ds_idx] = max(output['norm_scale'][s]['range'][ds_idx], self.scale_floor[1])

      #Check if any is none. IF so then make it the average of the other scales
      for ds_idx in range(series_count):
        if s in output['norm_scale']:
          if output['norm_scale'][s]['min'][ds_idx] is None:
            avg_min = np.array(min_container).mean()
            avg_max = np.array(max_container).mean()
            avg_range = avg_max - avg_min
            #output['norm_scale'][s]['max'][ds_idx] = self.scale_mode(avg_max + self.scale_eps)
            output['norm_scale'][s]['min'][ds_idx]   = max(self.scale_mode(avg_min + self.scale_eps[0]), self.scale_floor[0])
            output['norm_scale'][s]['range'][ds_idx] = max(self.scale_mode(avg_range + self.scale_eps[1]), self.scale_floor[1])

    for ds_idx, series in enumerate(output['unnorm_series']):
      norm_series = {}
      norm_series['name'] = series['name']
      norm_series['data'] = []

      for pt_idx, x_data in enumerate(series['data']):
        norm_x = {}
        for _, (k, v) in enumerate(x_data.items()):

          if isinstance(v, (float, int)):

            if k in ['y', 'y2', 'first_quartile', 'min', 'max', 'third_quartile', 'median']:
              scale_key = 'y'
            elif 'x' == k:
              scale_key = 'x'
            else:
              raise NotImplementedError(k)

            if self.norm_mode == 'minmax':
              min_val = output['unnorm_scale'][scale_key]['min'][ds_idx]
              minmax = output['unnorm_scale'][scale_key]['max'][ds_idx] - output['unnorm_scale'][scale_key]['min'][ds_idx]
              v = (v - min_val) / minmax if minmax > 0 else 0

            elif self.norm_mode == 'offset':
              # if ds_idx == 0 and pt_idx == 0 and k in ['min', 'x', 'y', 'y2']:
              #   v = 0.0
              # else:
              mean = output['norm_scale'][scale_key]['mean']
              std = output['norm_scale'][scale_key]['std']
              v = (v - mean) / std if std > 0 else 0.0

          norm_x[k] = v

        if len(norm_x):
          norm_series['data'].append(norm_x)
            
      if len(norm_series['data']):
        output['norm_series'].append(norm_series)

    return output

  def collate_data(self, batch_data):

    batch_chart_type = []
    batch_node_type = []
    batch_node_mask = []
    batch_reg_targets = []
    batch_reg_mask = []

    max_node_len = 0
    max_series_len = 0

    batch_scale_tgt = []

    max_series_len = max(len(d['norm_series']) for d in batch_data)

    for data in batch_data:
      series_node_type = [] 
      series_node_mask = []
      series_reg_tgt = []
      series_reg_mask = []
      series_scale_tgt = []

      chart_type = data['chart_type']
      batch_chart_type.append(chart_type)

      for s_idx, series in enumerate(data['norm_series']):
        node_type, node_mask = [], []
        reg_targets, reg_mask = [], []

        #TARGET 1 Scales
        scale_flag = True
        scale_tgt = []
        for s in ['x', 'y']:
          if s in data['norm_scale']:
            if self.norm_mode == 'minmax':
              scale_tensor =   torch.tensor([data['norm_scale'][s]['min'][s_idx], data['norm_scale'][s]['range'][s_idx]], dtype=torch.float32)
            elif self.norm_mode == 'offset':
              scale_tensor = torch.tensor([data['norm_scale'][s]['scale_mean'], data['norm_scale'][s]['scale_std']], dtype=torch.float32)

            if not torch.isfinite(scale_tensor).any():
              print("Scale tensor contains non finite data: {}".format(data))
              raise
            
            scale_tgt += scale_tensor

            if None in scale_tgt:
              scale_flag = False
              break
        
        #Check scales are correct. If not remove the series
        if not scale_flag:          
          continue

        if len(scale_tgt):
          scale_tgt =  torch.stack(scale_tgt, dim=-1).view(1, -1)
          #print(scale_tgt.shape)
          series_scale_tgt += scale_tgt

        
        #TARGET 3: Sequence of points (node type)
        prev_pt = None
        for pidx, point in enumerate(series['data']):
           
          #TARGET 4: Regression data
          if data['chart_type'] == 'vertical box': 
            # Original: [min, first-quartile, median, third-quartile, max]
            # Prediction: [min_val, first_to_min, median_to_first, third_to_median, max_to_third]
            # Prediction head must contain relu final layer

            #print("ds", point)
            min_val        = point['min']
            first_to_min   = point['first_quartile'] - point['min']
            median_to_first  = point['median'] - point['first_quartile']
            third_to_median = point['third_quartile'] - point['median']
            max_to_third   = point['max'] - point['third_quartile']
            reg_tgt = [min_val, first_to_min, median_to_first, third_to_median, max_to_third]

          elif data['chart_type'] in ['vertical bar', 'horizontal bar']:
            if pidx == 0 or self.norm_mode == 'minmax':
              reg_tgt = [point[k] for k in ['y']]
            else:
              reg_tgt = [point[k] - prev_pt[k] for k in ['y']]

          elif data['chart_type'] in ['line', 'scatter']: 
            try:
              if pidx == 0 or self.norm_mode == 'minmax':
                reg_tgt = [point[k] for k in ['x', 'y']]
              else:
                reg_tgt = [point[k] - prev_pt[k] for k in ['x', 'y']]
            except:
              #print("others", series['data'])
              #print("error with number of points in line/scatter => ", data['norm_series'])
              print(point.keys())
              raise
              continue

          else:
            raise NotImplementedError("Invalid chart given")
          
          #save for offsetting
          prev_pt = point

          node_type.append(self.node_map['point'])
          node_mask.append(1)

          reg_targets.append(reg_tgt)
          reg_mask.append(1)

        node_type.append(self.node_map['eos'])
        node_mask.append(1)

        reg_len = len(reg_targets[-1])
        reg_targets.append([0.] * reg_len)
        reg_mask.append(0)

        series_node_type += [node_type] 
        series_node_mask += [node_mask] 

        series_reg_tgt += [reg_targets]
        series_reg_mask += [reg_mask]

        if sum(reg_mask) == 0:
          print("reg_targets", reg_targets)
          print("series['data']", series['data'])
          raise

        #Ensure all are the same length
        cur_node_len = len(node_type)
        for idx, l in enumerate([node_type, node_mask, reg_targets, reg_mask]):
          assert len(l) == cur_node_len, "l={} idx={} cur_node_len={}".format(l, idx, cur_node_len)

        if cur_node_len > max_node_len:
          max_node_len = cur_node_len
        
      batch_node_type += [series_node_type] 
      batch_node_mask += [series_node_mask]
      batch_reg_targets += [series_reg_tgt] 
      batch_reg_mask += [series_reg_mask] 

      #Stacks by series
      if self.norm_mode == 'minmax':
        batch_scale_tgt += [torch.stack(series_scale_tgt, dim=0)]
      elif self.norm_mode == 'offset':
        batch_scale_tgt += [series_scale_tgt[0]]


    padded_batch_node_type = []
    padded_batch_node_mask = []
    padded_batch_reg_tgt = []
    padded_batch_reg_mask = []
    padded_batch_scale_tgt = []
    padded_batch_scale_mask = []

    #Padding to ensure whole batch is same length
    for series_node_type, series_node_mask, series_reg_tgt, series_reg_mask, series_scale_tgt in zip(
      batch_node_type, batch_node_mask, batch_reg_targets, batch_reg_mask , batch_scale_tgt
      ):
      padded_series_node_type = []
      padded_series_node_mask = []
      padded_series_reg_tgt = []
      padded_series_reg_mask = []

      #max_token_len = 0 
      ### Pad by column
      for idx, (node_type, node_mask, reg_tgt, reg_mask) in enumerate(zip(
        series_node_type, series_node_mask, series_reg_tgt, \
        series_reg_mask
        )):

        pad_node_len = max_node_len - len(node_type)

        assert sum(reg_mask) > 0, "must be more than zero"

        if pad_node_len > 0:
          node_type += [self.node_map['pad']] * pad_node_len
          node_mask += [0] * pad_node_len

          mask_pad = [0] * pad_node_len
          reg_mask += mask_pad
          
          reg_len = len(reg_tgt[-1])
          reg_pad = [[0.] * reg_len for _ in range(pad_node_len)]
          reg_tgt += reg_pad

        padded_series_node_type += [node_type]    
        padded_series_node_mask += [node_mask]    

        padded_series_reg_tgt  += [reg_tgt]            
        padded_series_reg_mask  += [reg_mask] 

      pad_series_len = max_series_len - len(padded_series_node_type)
      pad_node_len = max(len(p) for p in padded_series_node_type)
      reg_len = len(padded_series_reg_tgt[0][0])

      padded_series_reg_mask = torch.tensor(padded_series_reg_mask, dtype=torch.int32)
      padded_series_node_mask = torch.tensor(padded_series_node_mask, dtype=torch.int32)

      assert padded_series_reg_mask.sum() > 0, "mask is zero"
      for _ in range(pad_series_len):
        
        padded_series_node_type += [[self.node_map['eos']] + [self.node_map['pad']] * (pad_node_len-1)]
        reg_pad = [[0.] * reg_len for _ in range(pad_node_len)]
        padded_series_reg_tgt += [reg_pad]

      if pad_series_len > 0 and pad_node_len > 0:
        pad_mask = torch.zeros((pad_series_len, pad_node_len), dtype=torch.long)
        padded_series_reg_mask = torch.cat([padded_series_reg_mask, pad_mask], dim=0)
        padded_series_node_mask = torch.cat([padded_series_node_mask, pad_mask], dim=0)

      #####################################
      # SCALE PADDING
      #Pad the scales and make a mask
      series_scale_mask = None
      if self.norm_mode == 'minmax':
        series_len, scale_dim = series_scale_tgt.shape
        series_scale_mask = torch.ones((series_len), dtype=torch.int32)

        pad_len = max_series_len - series_len
        if pad_len > 0:
          pad_scale_tgt = torch.zeros((pad_len, scale_dim), dtype=torch.int32)
          series_scale_tgt = torch.cat([series_scale_tgt, pad_scale_tgt], dim=0)

          pad_scale_mask = torch.zeros((pad_len), dtype=torch.int32)
          series_scale_mask = torch.cat([series_scale_mask, pad_scale_mask], dim=0)
          #print("pad series_scale_tgt", series_scale_tgt.shape, series_scale_mask.shape)

      #Pad the text? Not needed if all text is the same. series name and categorical names

      padded_series_reg_tgt = torch.tensor(padded_series_reg_tgt, dtype=torch.float32)

      padded_batch_node_type += [padded_series_node_type]
      padded_batch_node_mask += [padded_series_node_mask]
      padded_batch_reg_tgt += [padded_series_reg_tgt]
      padded_batch_reg_mask += [padded_series_reg_mask]
      padded_batch_scale_tgt += [series_scale_tgt]
      padded_batch_scale_mask += [series_scale_mask]

    #Stack non sequence problems
    inputs = {}
    inputs['chart_type'] = batch_chart_type
    
    #### Continuous data (Can be different dimension depending on chart type. cannot stack these.)

    inputs['scale'] = {}
    inputs['scale']['inputs_embeds'] = padded_batch_scale_tgt #batch_scale_tgt #torch.stack(batch_scale_tgt, dim=0)
    inputs['scale']['attention_mask'] = padded_batch_scale_mask #batch_scale_tgt #torch.stack(batch_scale_tgt, dim=0)

    if self.norm_mode == 'minmax':
      inputs['scale']['decoder_inputs_embeds'] = [shift_right_vec(inp, start_values=0.0) for inp in inputs['scale']['inputs_embeds']]
      inputs['scale']['decoder_attention_mask'] = [shift_right_vec(inp, start_values=0.0) for inp in inputs['scale']['attention_mask']]

    inputs['continuous'] = {}
    inputs['continuous']['inputs_embeds'] = padded_batch_reg_tgt # torch.stack(padded_batch_reg_tgt, dim=0)

    continuous_mask = torch.stack(padded_batch_reg_mask, dim=0)
    assert continuous_mask.sum() > 0, "must be atleast reg :{}".format(continuous_mask)


    inputs['continuous']['attention_mask']  = continuous_mask
    inputs['continuous']['decoder_inputs_embeds'] = [shift_right_vec(inp, start_values=0.0) for inp in inputs['continuous']['inputs_embeds']]
    inputs['continuous']['decoder_attention_mask']  = [shift_right_vec(inp, start_values=0.0) for inp in inputs['continuous']['attention_mask']]

      
    inputs['node'] = {}
    inputs['node']['input_ids'] = torch.tensor(padded_batch_node_type, dtype=torch.int32)
    inputs['node']['attention_mask'] = torch.stack(padded_batch_node_mask, dim=0)  #torch.tensor(padded_batch_node_mask, dtype=torch.long)
    #inputs['node']['decoder_input_ids'] = torch.stack([shift_tokens_right(inp, self.node_map['pad'], dim=-1) for inp in inputs['node']['input_ids']], dim=0)   
    
    inputs['labels'] = {}
    inputs['labels']['col'] = inputs['node']['input_ids'][:,0,:]
    inputs['labels']['row'] = inputs['node']['input_ids'][:,:,0]

    return inputs
