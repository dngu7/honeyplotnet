# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import math
import torch
import numpy as np

from utils import CHART_TYPE_MAP, PAD_IDX

from .base import (
  BaseDataset, get_text_window,
  stack_dict, shift_right_vec, 
)


class PmcDataDataset(BaseDataset):
  def __init__(self, data, 
                tokenizer1, 
                tokenizer2=None, 
                active_tasks=None, 
                scale_mode='log10', 
                scale_eps=1.00001, scale_floor=-1.0, window_size=16, widen_rate=0.0, max_widen_len=1, 
                min_sent_len=10, max_source_len=1024, max_target_len=256, max_source_len2=8, max_target_len2=8, 
                pad_to_max_len=True, ignore_pad_token_for_loss=True, norm_mode='minmax', chart_text_input=None, 
                chart_text_output=['categorical'], sep_token='<SEP>', sep_token1='<SEP1>', sep_token2='<SEP2>',
               **kwargs):
    super().__init__(tokenizer1, window_size, widen_rate, max_widen_len, 
               min_sent_len, max_source_len, max_target_len, pad_to_max_len, 
               ignore_pad_token_for_loss)

    # Filter the pmc dataset for certain tasks or charts
    self.active_tasks = ['task6'] if active_tasks is None else active_tasks
    self.active_charts = [] 

    self.tokenizer2 = tokenizer1 if tokenizer2 is None else tokenizer2
    self.max_source_len2 = max_source_len if max_source_len2 is None else max_source_len2
    self.max_target_len2 = max_target_len if max_target_len2 is None else max_target_len2

    self.chart_text_input = chart_text_input
    self.sep_token = sep_token
    self.sep_token1 = sep_token1
    self.sep_token2 = sep_token2
    self.chart_text_output = chart_text_output

    self.data = self.filter_data(data)
    
    assert scale_mode in ['log10', 'log'], "scale mode not implemented"
    self.scale_mode        = math.log10 if scale_mode == 'log10' else math.log
    self.scale_eps         = scale_eps
    self.scale_floor       = scale_floor

    self.box_plot_keys = ['min', 'first_quartile', 'median',  'third_quartile', 'max']
    self.node_map = {'pad': PAD_IDX, 'point': 1, 'eos': 0}
    
    self.del_list = []

    assert norm_mode in ['minmax','offset']
    self.norm_mode = norm_mode
    
  def filter_data(self, data):
    task_req = []
    if isinstance(self.active_tasks, list):
      assert 'task6' in self.active_tasks
      task_req = self.active_tasks
      data = [d for d in data if all(t in d and d[t] != None for t in task_req)]

    if len(self.active_charts):
      data = [d for d in data if d['task1']['output']['chart_type'].lower().strip() in self.active_charts]

    return data
  
  def __len__(self):
      return len(self.data)
    
  def get_data_with_idx(self, index):
    if isinstance(index, list):
      return [self.data[i % len(self.data)] for i in index]
    elif isinstance(index, int):
      return self.data[index % len(self.data)]
    else:
      raise ValueError("Invalid index given: {}".format(index))

  def __getitem__(self, index):
      
    while True:
      d = self.data[index % len(self.data)]

      #Get context
      fig_id = d['fig_id']
      context_start = d['fig_index'][fig_id][0]

      all_text = d['all_text']
      captions = d['captions']

      #Remove caption from text
      all_text = self.preprocess_all_text(all_text)

      caption_label, all_text, context_start = self.get_caption_label(
        captions, all_text, context_start)

      context = get_text_window(
        all_text, context_start, 
        tgt_token=self.tgt_token,
        window_size=self.window_size, 
        widen_rate=self.widen_rate, 
        max_widen_len=self.max_widen_len, 
        min_sent_len=self.min_sent_len)

      ######################
      # Prepare inputs
      ######################
      input_text = ''
      if 'captions' in self.chart_text_input:
        input_text += caption_label
      if 'context' in self.chart_text_input:
        input_text += context
      
      outputs = {}

      context_inputs = None
      if self.tokenizer is not None:
        context_inputs, _ = self._tokenize(self.tokenizer, 
          input_text, max_source_len=self.max_source_len, 
          max_target_len=self.max_target_len, is_target=False)
        outputs['context'] = context_inputs

      
      outputs['chart_type'] = self.get_chart_type(d)
      outputs['chart_data'] = self.preprocess_data_series(d)
      
      if self.enforce_data_to_chart_type(outputs):
        break
      else:
        index += 1
        
    return index, outputs

  def collate_fn(self, list_batch):
    list_idx = [b[0] for b in list_batch]
    list_batch = [b[1] for b in list_batch]
    collated_batch = stack_dict(list_batch)
    
    if self.tokenizer is not None:
      collated_batch['context'] = self.batch_tokens(collated_batch['context'])
    
    collated_batch['chart_data'] = self.collate_data(collated_batch['chart_data'])
    return list_idx, collated_batch
  
  def batch_tokens(self, list_batch):
    dict_of_tokens = stack_dict(list_batch)

    
    for key in list(dict_of_tokens.keys()):

      dict_of_tokens[key] = torch.cat(dict_of_tokens[key], dim=0)
  
    return dict_of_tokens

  def get_chart_type(self, d):
    '''Returns an index'''
    ct = d['task1']['output']['chart_type'].lower().strip()
    return CHART_TYPE_MAP.index(ct)

  def preprocess_data_series(self, d):
    
    output = {}
    output['unnorm_series'] = []
    output['norm_series'] = []
    output['unnorm_scale'] = {}
    
    output['chart_type'] = {}
    output['data_type'] = {}

    chart_type = d['task1']['output']['chart_type'].lower().strip()
    output['chart_type'] = chart_type

    #For normalising the data
    for s in ['y', 'x']:
      output['unnorm_scale'][s] = {}
      if self.norm_mode == 'offset':
        output['unnorm_scale'][s]['all'] = [] #Used to compute mean and scales
      elif self.norm_mode == 'minmax':
        output['unnorm_scale'][s]['min'] =  [float('inf')] * len(d['task6']['output']['data series'])
        output['unnorm_scale'][s]['max'] = [-float('inf')] * len(d['task6']['output']['data series'])
      else:
        raise NotImplementedError()

    #First loop to collect data from json. data is organised for scaling later
    first_v = {}
    #This goes through the series
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
        if ('x' not in series_keys) or \
            ('y' not in series_keys and \
             'y2' not in series_keys):
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
          
          data_type = type(v)
          output['data_type'][k] = data_type
          
          #Remove spaces
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

            #Use offset mode (mean,std) or not min,range
            if self.norm_mode == 'minmax':
              if v < -self.scale_eps[0]:
                v = 0
              pt_store[k] = v
              output['unnorm_scale'][scale_key]['min'][ds_idx] = min(output['unnorm_scale'][scale_key]['min'][ds_idx], v)
              output['unnorm_scale'][scale_key]['max'][ds_idx] = max(output['unnorm_scale'][scale_key]['max'][ds_idx], v)

            elif self.norm_mode == 'offset':
              pt_store[k] = v
              output['unnorm_scale'][scale_key]['all'].append(v)
              
            else:
              raise NotImplementedError(self.norm_mode)
          
        if len(pt_store) > 1:
          unnorm_series['data'].append(pt_store)
          keys = list(pt_store.keys())
          assert True if 'y' in keys else 'y2' not in keys, "{}, {}, {}".format(series_keys, keys, y2_replacement_flag)

      output['unnorm_series'].append(unnorm_series)

    ##########################################
    #Second loop to calculate statistics on collected data and then normalise data. e.g. min max or mean std

    output['norm_scale'] = {}

    #Compute the mean and std 
    if self.norm_mode == 'offset':
      for s in list(output['unnorm_scale'].keys()):
        all_data = np.array(output['unnorm_scale'][s]['all'])
        if len(all_data):
          #if chart_type not in ['vertical box']:
          #print(s, list(output['unnorm_scale'].keys()), "all_data", all_data)
          #d_start = all_data[0]
          d_mean  = np.mean(all_data)
          d_std   = np.std(all_data)

          #Clip means lower than eps
          d_mean = max(d_mean, -self.scale_eps[0] + 0.0001)

          scale_mean = self.scale_mode(d_mean + self.scale_eps[0])
          scale_std  = self.scale_mode(d_std + self.scale_eps[1])

          if not np.isfinite(all_data).any():
            print("Data contains non finite: {}".format(all_data))
            raise

          output['norm_scale'][s] = {
            'mean': d_mean, 
            'std': d_std,
            'scale_mean': scale_mean,
            'scale_std': scale_std,
          }
          #print("output['norm_scale'][s]", output['norm_scale'][s])

    elif self.norm_mode == 'minmax':
      for s in ['y', 'x']:
        series_count = len(output['unnorm_scale'][s]['min'])

        min_container, max_container = [],[]
        for ds_idx in range(series_count):
          if output['unnorm_scale'][s]['min'][ds_idx] < float('inf') and \
            output['unnorm_scale'][s]['max'][ds_idx] > -float('inf'): # and \
            #output['unnorm_scale'][s]['min'][ds_idx] !=  output['unnorm_scale'][s]['max'][ds_idx]:

            if s not in output['norm_scale']:
              output['norm_scale'][s] = {}
              for n in ['min','max','range']:
                output['norm_scale'][s][n] = [None] * series_count

            output['norm_scale'][s]['min'][ds_idx] = self.scale_mode(output['unnorm_scale'][s]['min'][ds_idx] + self.scale_eps[0])
            output['norm_scale'][s]['max'][ds_idx] = self.scale_mode(output['unnorm_scale'][s]['max'][ds_idx] + self.scale_eps[1])
            
            min_container.append(output['unnorm_scale'][s]['min'][ds_idx])
            max_container.append(output['unnorm_scale'][s]['max'][ds_idx])

            #Calculate the scale before the floor to prevent out of domain errors
            scale_range = output['unnorm_scale'][s]['max'][ds_idx] - output['unnorm_scale'][s]['min'][ds_idx]
            output['norm_scale'][s]['range'][ds_idx] = self.scale_mode(scale_range + self.scale_eps[1])

            output['norm_scale'][s]['min'][ds_idx] = max(output['norm_scale'][s]['min'][ds_idx], self.scale_floor[0])
            output['norm_scale'][s]['range'][ds_idx] = max(output['norm_scale'][s]['range'][ds_idx], self.scale_floor[1])
        
        #Check if any is none, then just average the rest
        for ds_idx in range(series_count):
          if s in output['norm_scale']:
            if output['norm_scale'][s]['min'][ds_idx] is None:
              avg_min = np.array(min_container).mean()
              avg_max = np.array(max_container).mean()
              avg_range = avg_max - avg_min
              #output['norm_scale'][s]['max'][ds_idx] = self.scale_mode(avg_max + self.scale_eps[0])
              output['norm_scale'][s]['min'][ds_idx] = max(self.scale_mode(avg_min + self.scale_eps[0]), self.scale_floor[0])
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

  def enforce_data_to_chart_type(self, outputs):
      flag = False
      if len(outputs['chart_data']['norm_series']) == 0 or len(outputs['chart_data']['norm_series'][0]['data']) == 0:
        flag = False
      elif len(outputs['chart_data']['norm_scale']) == 0:
        flag = False
      elif outputs['chart_data']['chart_type'] == 'vertical box':
        flag = True
      elif outputs['chart_data']['chart_type']   in ['line', 'scatter'] and len(outputs['chart_data']['norm_scale']) == 2:
        flag = all(v==float for k, v in outputs['chart_data']['data_type'].items())
      elif outputs['chart_data']['chart_type'] in ['vertical bar', 'horizontal bar'] and len(outputs['chart_data']['norm_scale']) == 1:
        flag = True if outputs['chart_data']['data_type']['x'] == str else False
      else:
        flag = False
        #raise ValueError("chart type not recognized: {}".format(outputs['data']['chart_type']))
      return flag
  
  def collate_captions(self, list_batch):
    collated_batch = stack_dict(list_batch)
    for key in list(collated_batch.keys()):
      if isinstance(collated_batch[key], torch.Tensor):
        collated_batch[key] = torch.stack(collated_batch[key], dim=0)
    return collated_batch
  
  def tokenize_text_batch(self, padded_batch_txt_tgt, depth=3):
    all_txt_inputs, all_txt_labels = {},{}
    for padded_series_txt_tgt in padded_batch_txt_tgt:
      length = len(padded_series_txt_tgt)
      collector = []
      for txt_raw in padded_series_txt_tgt:
        if depth == 2:
          collector += [txt_raw]
          if len(collector) < length:
            continue
          else:
            txt_raw = collector
    
        txt_inputs, txt_labels = self._tokenize(
          self.tokenizer2, txt_raw, 
          max_source_len=self.max_source_len2, 
          max_target_len=self.max_target_len2, 
          is_target=True, shift_right=True)

        for k, v in txt_inputs.items():
          if k not in all_txt_inputs:
            all_txt_inputs[k] = []
          all_txt_inputs[k] += [v]
        
        for k, v in txt_labels.items():
          if k not in all_txt_labels:
            all_txt_labels[k] = []
          all_txt_labels[k] += [v]
    #all_txt_inputs = stack_tensor_dict(all_txt_inputs)
    #all_txt_labels = stack_tensor_dict(all_txt_labels)

    ################################################
    #Pad and batch
    #def pad_n_batch()
    label = all_txt_labels['input_ids']
    col_counts = [l.size(0) for l in label]
    max_token_len = label[0].size(1)

    padded_labels = []
    for lbl in label:
      pad_len = max(col_counts) - lbl.size(0)
      if pad_len > 0:
        pad = torch.ones((pad_len, max_token_len), dtype=torch.int32) * self.tokenizer2.pad_token_id
        lbl = torch.cat([lbl, pad], dim=0)
      padded_labels.append(lbl)
    label = torch.stack(padded_labels, dim=0)
    label[label == self.tokenizer2.pad_token_id] = PAD_IDX
    all_txt_labels['input_ids'] = label

    return all_txt_inputs, all_txt_labels
    
  def collate_data(self, batch_data):

    batch_chart_type = []
    batch_node_type = []
    batch_node_mask = []
    batch_reg_targets = []
    batch_reg_mask = []
    batch_text_targets = []
    batch_text_mask = []

    max_node_len = 0
    max_series_len = 0

    batch_name_targets = [] 
    batch_name_mask = []

    batch_scale_tgt = []

    max_series_len = max(len(d['norm_series']) for d in batch_data)

    for data in batch_data:
      series_node_type = [] 
      series_node_mask = []
      series_reg_tgt = []
      series_reg_mask = []
      series_txt_tgt = []
      series_txt_mask = []

      series_name_tgt = []
      series_name_mask = []
      series_scale_tgt = []

      chart_type = data['chart_type']
      batch_chart_type.append(chart_type)

      for s_idx, series in enumerate(data['norm_series']):
        node_type, node_mask = [], []
        reg_targets, reg_mask = [], []
        txt_tgt, txt_mask = [], []

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

        #TARGET 2: Series name = 'asdf'
        if series['name'] is not None:
          series_name_tgt.append(series['name'])
          series_name_mask.append(1)
        else:
          series_name_tgt.append('')
          series_name_mask.append(0)
        
        #TARGET 3: Sequence of points ( node type)
        prev_pt = None
        for pidx, point in enumerate(series['data']):
           
          #TARGET 4: Regression data
          if data['chart_type'] == 'vertical box': 
            # Original: [min, first-quartile, median, third-quartile, max]
            # Prediction: [min_val, first_to_min, median_to_min, third_to_first, max_to_third]
            # Prediction head must contain relu final layer

            min_val        = point['min']
            first_to_min   = point['first_quartile'] - point['min']
            median_to_min  = point['median'] - point['min']
            third_to_first = point['third_quartile'] - point['first_quartile']
            max_to_third   = point['max'] - point['third_quartile']
            reg_tgt = [min_val, first_to_min, median_to_min, third_to_first, max_to_third]

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
          #print("point", point, prev_pt, reg_tgt)
          prev_pt = point

          node_type.append(self.node_map['point'])
          node_mask.append(1)

          #TARGET 5: TEXT data
          if data['chart_type'] not in ['line', 'scatter'] and \
            point.get('x') is not None and \
            len(point.get('x')) and \
            isinstance(point.get('x'), str):

            txt_tgt.append(point['x'])
            txt_mask.append(1)
          else:
            txt_tgt.append('')
            txt_mask.append(0)

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

        series_txt_tgt += [txt_tgt]
        series_txt_mask += [txt_mask]

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
      batch_text_targets += [series_txt_tgt] 
      batch_text_mask += [series_txt_mask]

      batch_name_targets += [series_name_tgt] 
      batch_name_mask += [series_name_mask]

      #Stacks by series
      if self.norm_mode == 'minmax':
        batch_scale_tgt += [torch.stack(series_scale_tgt, dim=0)]
      elif self.norm_mode == 'offset':
        batch_scale_tgt += [series_scale_tgt[0]]



    # print("batch_scale_tgt")
    # for p in batch_scale_tgt:
    #   print(p.shape)

    padded_batch_node_type = []
    padded_batch_node_mask = []
    padded_batch_reg_tgt = []
    padded_batch_reg_mask = []
    padded_batch_txt_tgt = []
    padded_batch_scale_tgt = []
    padded_batch_scale_mask = []

    #Padding to ensure whole batch is same length
    for series_node_type, series_node_mask, series_reg_tgt, series_reg_mask, series_txt_tgt, series_txt_mask, series_scale_tgt in zip(
      batch_node_type, batch_node_mask, batch_reg_targets, batch_reg_mask , batch_text_targets, batch_text_mask, batch_scale_tgt
      ):
      padded_series_node_type = []
      padded_series_node_mask = []

      padded_series_reg_tgt = []
      padded_series_reg_mask = []
      padded_series_txt_tgt = []
      padded_series_txt_mask = []

      #max_token_len = 0 
      ### Pad by column
      for idx, (node_type, node_mask, reg_tgt, reg_mask, txt_tgt, txt_mask) in enumerate(zip(
        series_node_type, series_node_mask, series_reg_tgt, \
        series_reg_mask, series_txt_tgt, series_txt_mask
        )):

        pad_node_len = max_node_len - len(node_type)

        assert sum(reg_mask) > 0, "must be more than zero"

        if pad_node_len > 0:
          node_type += [self.node_map['pad']] * pad_node_len
          node_mask += [0] * pad_node_len

          mask_pad = [0] * pad_node_len
          reg_mask += mask_pad
          txt_mask += mask_pad
          
          reg_len = len(reg_tgt[-1])
          reg_pad = [[0.] * reg_len for _ in range(pad_node_len)]
          reg_tgt += reg_pad

        padded_series_node_type += [node_type]    
        padded_series_node_mask += [node_mask]    

        padded_series_reg_tgt  += [reg_tgt]            
        padded_series_reg_mask  += [reg_mask] 

        if idx == 0:
          padded_series_txt_tgt += [txt_tgt]
          padded_series_txt_mask += [txt_mask]  

        #categorical text data is always the same, except the series name
        # txt_tgt = ['text1', 'text2', '', '']
        # txt_tgt_ids = [[2,3,4,0], [1,2,0,0], [1,0,0,0], [1,0,0,0]]
        #max_token_len = max(max_token_len, len(txt_tgt_ids[0]))

      pad_series_len = max_series_len - len(padded_series_node_type)
      pad_node_len = max(len(p) for p in padded_series_node_type)
      reg_len = len(padded_series_reg_tgt[0][0])

      #print("padded_series_node_type", len(padded_series_node_type))
      #asdf = [len(p) for p in padded_series_node_type]
      #print("asdf'", asdf)

      #padded_series_txt_mask = torch.tensor(padded_series_txt_mask, dtype=torch.long)
      padded_series_reg_mask = torch.tensor(padded_series_reg_mask, dtype=torch.int32)
      padded_series_node_mask = torch.tensor(padded_series_node_mask, dtype=torch.int32)

      assert padded_series_reg_mask.sum() > 0, "mask is zero"
      for _ in range(pad_series_len):
        
        padded_series_node_type += [[self.node_map['eos']] + [self.node_map['pad']] * (pad_node_len-1)]
        reg_pad = [[0.] * reg_len for _ in range(pad_node_len)]
        padded_series_reg_tgt += [reg_pad]

      if pad_series_len > 0 and pad_node_len > 0:
        pad_mask = torch.zeros((pad_series_len, pad_node_len), dtype=torch.long)
        #print("pad_mask", pad_mask.shape)
        #padded_series_txt_mask = torch.cat([padded_series_txt_mask, pad_mask], dim=1)
        padded_series_reg_mask = torch.cat([padded_series_reg_mask, pad_mask], dim=0)
        padded_series_node_mask = torch.cat([padded_series_node_mask, pad_mask], dim=0)
        #print("padded_series_reg_mask", padded_series_reg_mask.shape)

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
      padded_batch_txt_tgt += [padded_series_txt_tgt]
      #padded_batch_txt_mask += [padded_series_txt_mask]

      padded_batch_scale_tgt += [series_scale_tgt]
      padded_batch_scale_mask += [series_scale_mask]
    
    #Shift right all sequence to sequence problems (text, name)
    #print(padded_batch_txt_tgt)
    
    if self.tokenizer is not None:
      txt_inputs, txt_labels = self.tokenize_text_batch(padded_batch_txt_tgt)
      name_inputs, name_labels = self.tokenize_text_batch(batch_name_targets, depth=2)
    #for k in list(name_inputs.keys()):
    #  name_inputs[k] = torch.stack(name_inputs[k], dim=0)
      
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
    attn_mask = torch.stack(padded_batch_reg_mask, dim=0)
    assert attn_mask.sum() > 0, "must be atleast reg :{}".format(attn_mask)

    #print("dataloader >>> attn_mask", attn_mask.shape, attn_mask[:,:10])
    inputs['continuous']['attention_mask']  = attn_mask

    inputs['continuous']['decoder_inputs_embeds'] = [shift_right_vec(inp, start_values=0.0) for inp in inputs['continuous']['inputs_embeds']]
    inputs['continuous']['decoder_attention_mask']  = [shift_right_vec(inp, start_values=0.0) for inp in inputs['continuous']['attention_mask']]

    #### chart_text data
    if self.tokenizer is not None:
      inputs['categorical'] = txt_inputs
      inputs['categorical']['label'] = txt_labels
      inputs['categorical']['raw'] = padded_batch_txt_tgt

      inputs['series_name'] = name_inputs
      inputs['series_name']['label'] = name_labels
      inputs['series_name']['raw'] = batch_name_targets
      
    inputs['node'] = {}
    inputs['node']['input_ids'] = torch.tensor(padded_batch_node_type, dtype=torch.int32)
    inputs['node']['attention_mask'] = torch.stack(padded_batch_node_mask, dim=0)  #torch.tensor(padded_batch_node_mask, dtype=torch.long)
    #inputs['node']['decoder_input_ids'] = torch.stack([shift_tokens_right(inp, self.node_map['pad'], dim=-1) for inp in inputs['node']['input_ids']], dim=0)   
    
    inputs['labels'] = {}
    inputs['labels']['col'] = inputs['node']['input_ids'][:,0,:]
    inputs['labels']['row'] = inputs['node']['input_ids'][:,:,0]
    return inputs
