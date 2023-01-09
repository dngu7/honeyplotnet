# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import os
import json
import numpy as np 

import torch
from torch.nn import functional as F
from utils.recon_plots import create_boxplot, create_scatter

from transformers.trainer_pt_utils import (
  find_batch_size, 
  nested_concat, 
  nested_numpify,
  nested_truncate,
)

from transformers import DataCollatorForSeq2Seq


from .caption import CaptionRunner
from dataset import PmcCaptionInferenceDataset
from torch.utils.data import DataLoader, SequentialSampler

from utils import (
  TASK2PREPEND, 
  prepare_mpl,
  create_bar_chart, 
  create_scatter,
  create_boxplot
)


class GenRunner(CaptionRunner):
  def __init__(self, stage, cfg):
    super(GenRunner, self).__init__(stage, cfg)
    self.stage = stage
    self.chart_text_input = cfg.data.dataset.chart_data.chart_text_input
    self.gen_temperature = cfg.eval.gen_temperature
    self.gen_hypo_count = cfg.eval.hypo_count
    self.gen_hypo_bsz = cfg.eval.hypo_bsz

    self.tok_cb_map = None

    self.cb1_len = cfg.model.continuous_data.vq.emb_len1
    self.cb2_len = cfg.model.continuous_data.vq.emb_len2

  def generate_continuous(self, cb_ind1, cb_ind2, ct_idx, models, tokenizers):
    for m in models.values():
      if m is not None:
        m.eval()
    
    with torch.no_grad():
      with self.autocast_smart_context_manager():

        if models['continuous'].__class__.__name__ == 'DistributedDataParallel':
          cont_module = models['continuous'].module
        else:
          cont_module = models['continuous']

        samples = cont_module.reconstruct_from_indices(
          cb_ind1=cb_ind1, 
          cb_ind2=cb_ind2, 
          ct_idx=ct_idx,
          temp=self.gen_temperature,
          hypo_count=self.gen_hypo_count, 
          hypo_bsz=self.gen_hypo_bsz, 
          )

    return samples

  def sample_indices(self, logits, temp=1.0):
    bsz = logits.size(0)
    probs = F.softmax(logits / temp, dim=-1).data
    probs = torch.flatten(probs, start_dim=0, end_dim=1)
    cb_indices =  torch.multinomial(probs, 1)
    cb_indices = cb_indices.reshape([bsz, -1])
    return cb_indices

  def tokenize(self, text, tokenizer):
      inputs = tokenizer(
        text, max_length=self.cfg.model.caption.hf_model.max_source_len, 
          padding="max_length", truncation=True, return_tensors="pt")
      return inputs
    
  def generate_caption(self, loader, models, tokenizers):
    for model in models.values():
      model.eval()
    
    iterator = loader.__iter__()

    all_tasks =  ['caption', 'context']

    # Initialize containers
    preds_host = {t: None for t in all_tasks}

    # losses/preds/labels on CPU (final containers)
    all_preds = {t: None for t in all_tasks}

    observed_num_examples = 0
    batch_size = self.bsz

    #Only run captions
    task = 'caption'
    task_str = TASK2PREPEND[task]

    for step, contexts in enumerate(iterator):

      if isinstance(self.cfg.eval.max_steps, int) and step > self.cfg.eval.max_steps:
        break
      
      preds_host['context'] = contexts if preds_host['context'] is None else preds_host['context'] + contexts
      
      #Append task to context
      task_contexts = [task_str + c for c in contexts]

      inputs = self.tokenize(task_contexts, tokenizers['caption'])

      # Update the observed num examples
      observed_batch_size = find_batch_size(inputs)
      if observed_batch_size is not None:
          observed_num_examples += observed_batch_size
          # For batch samplers, batch_size is not known by the dataloader in advance.
          if batch_size is None:
              batch_size = observed_batch_size
              
      _, logits, _ = self.prediction_step(
        models['caption'], tokenizer=tokenizers['caption'], 
        inputs=inputs, prediction_loss_only=False)
      

      logits = self._pad_across_processes(logits)
      logits = self._nested_gather(logits)
      preds_host[task] = logits if preds_host[task] is None else nested_concat(preds_host[task], logits, padding_index=-100)
    
      #Move collection to CPU
      #for task in TASK2PREPEND.keys():
      logits = nested_numpify(preds_host[task])
      all_preds[task] = logits if all_preds[task] is None else nested_concat(all_preds[task], logits, padding_index=-100)

      all_preds['context'] = preds_host['context'] if all_preds['context'] is None else all_preds['context'] + preds_host['context']
      
      #Reset containers
      preds_host = {t: None for t in all_tasks}

      
    # Gather all remaining tensors and put them back on the CPU
    num_samples = self.cfg.eval.max_steps

    #for task in TASK2PREPEND.keys():
    logits = nested_numpify(preds_host[task]) if preds_host[task] is not None else None
    if logits is not None:
      all_preds[task] = logits if all_preds[task] is None else nested_concat(all_preds[task], logits, padding_index=-100)
    
    all_preds[task] = nested_truncate(all_preds[task], num_samples)
    
    return all_preds

  def generate_chart_text(self, all_captions, models, tokenizers):
    for model in models.values():
      model.eval()
    

    all_tasks = ['series_name', 'chart_text', 'axis']

    # Initialize containers
    preds_host = {t: None for t in all_tasks}

    # losses/preds/labels on CPU (final containers)
    all_preds = {t: None for t in all_tasks}

    observed_num_examples = 0
    batch_size = self.bsz
    total_steps = int(np.ceil(len(all_captions) / batch_size))

    for step in range(total_steps):
      captions = all_captions[step * batch_size:step * batch_size + batch_size]

      if isinstance(self.cfg.eval.max_steps, int) and step > self.cfg.eval.max_steps:
        break
      
      #preds_host['captions'] = captions if preds_host['captions'] is None else preds_host['captions'] + captions
      #Loop through each task and generate
      for task in all_tasks:
        task_str = TASK2PREPEND[task]
        task_contexts = [task_str + c[0] for c in captions]

        inputs = self.tokenize(task_contexts, tokenizers['chart_text'])

        # Update the observed num examples
        observed_batch_size = find_batch_size(inputs)
        if observed_batch_size is not None:
            observed_num_examples += observed_batch_size
            # For batch samplers, batch_size is not known by the dataloader in advance.
            if batch_size is None:
                batch_size = observed_batch_size
                
        _, logits, _ = self.prediction_step(
          models['chart_text'], tokenizer=tokenizers['chart_text'], 
          inputs=inputs, prediction_loss_only=False)
        
        logits = self._pad_across_processes(logits)
        logits = self._nested_gather(logits)
        preds_host[task] = logits if preds_host[task] is None else nested_concat(preds_host[task], logits, padding_index=-100)
      
      #Move collection to CPU
      for task in all_tasks:
        logits = nested_numpify(preds_host[task])
        all_preds[task] = logits if all_preds[task] is None else nested_concat(all_preds[task], logits, padding_index=-100)

      #Reset containers
      preds_host = {t: None for t in all_tasks}
      
    # Gather all remaining tensors and put them back on the CPU
    num_samples = self.cfg.eval.max_steps

    for task in all_tasks:
      logits = nested_numpify(preds_host[task]) if preds_host[task] is not None else None
      if logits is not None:
        all_preds[task] = logits if all_preds[task] is None else nested_concat(all_preds[task], logits, padding_index=-100)
      
      all_preds[task] = nested_truncate(all_preds[task], num_samples)
    
    
    return all_preds

  def batch_decode(self, tokens, tokenizer, skip_special_tokens=True):
    return tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True)

  def seperate_text_to_list(self, text, seperator='<SEP>'):
    #Convert to list of text
    #processed_dtext = []
    # for dtext in text:
    for remv in ['<pad>','<unk>','<UNK>','</s>']:
      text = text.replace(remv,'')

    split_text = text.split(seperator)

    split_text = list(set([t.strip() for t in split_text]))

    #processed_dtext.append(split_text)
    return split_text

  def detokenize(self, tokens, tokenizer, seperator='<SEP>'):
    decoded = {}
    for task in tokens.keys():
      if task not in ['context']:
        skip_special_tokens = True if task == 'caption' else False
        decoded[task] = self.batch_decode(tokens[task], tokenizer, skip_special_tokens=skip_special_tokens)
        #print(f"detokenize >>> [{task}] >>>> tokens={len(tokens[task])}  >>>>> decoded={len(decoded[task])}")
      
    for task in decoded.keys():
      decoded[task] = [self.seperate_text_to_list(b, seperator=seperator) for b in decoded[task]]
    
    #print("detokenized.decoded tasks: {}".format(decoded.keys()))
    return decoded

  def generate_codebook(self, loader, models):
    for m in models.values():
      if m is not None:
        m.eval()

    iterator = loader.__iter__()
    #container = {'cb1': [], 'cb2': [], 'ct_idx': []}
    container = {}
    ct_idxs = []
    for _, inputs in enumerate(iterator):
      
      context_idx = inputs['input_ids']
      attn_mask = inputs['attention_mask']

      with torch.no_grad():
        with self.autocast_smart_context_manager():
          cb1_logits, cb2_logits, ct_idx, _ = models['seq'](
            context_idx, attn_mask=attn_mask)

          ct_idxs.append(ct_idx.detach().cpu())

          cb_ind1 = self.sample_indices(cb1_logits, temp=self.gen_temperature)

          cb_ind2 = None
          if cb2_logits is not None:
            cb_ind2 = self.sample_indices(cb2_logits, temp=self.gen_temperature)

          if models['continuous'].__class__.__name__ == 'DistributedDataParallel':
            cont_module = models['continuous'].module
          else:
            cont_module = models['continuous']

          samples = cont_module.reconstruct_from_indices(
            cb_ind1=cb_ind1, 
            cb_ind2=cb_ind2, 
            ct_idx=ct_idx,
            temp=self.gen_temperature,
            hypo_count=self.gen_hypo_count, 
            hypo_bsz=self.gen_hypo_bsz, 
            )
          
          #Storing outputs cleanly
          for k, v in samples.items():
            if k in ['ct_idx', 'chart_type_dict']: continue
            if k not in container:
              container[k] = {} 
            for kk, vv in v.items():
              if isinstance(vv, list):
                if kk not in container[k]:
                  container[k][kk] = []
                container[k][kk] += [vvv.detach().cpu() for vvv in vv]
              elif isinstance(vv, dict):
                if kk not in container[k]:
                  container[k][kk] = {}
                for kkk, vvv in vv.items():
                  if kkk not in container[k][kk]:
                    container[k][kk][kkk] = []
                  container[k][kk][kkk].append(vvv.detach().cpu())    
              else:
                raise ValueError(f"Key doesnt exist: {k} {kk}")            

    #Loop through container and concat
    for k in ['shape']:
      for kk in ['counts', 'embeds']:
        for kkk in ['row', 'col']:
          container[k][kk][kkk] = torch.cat(container[k][kk][kkk], dim=0)

    ct_idxs = torch.cat(ct_idxs, dim=0)
    container['chart_idx'] = ct_idxs.view(-1).cpu().tolist()

    return container

  def create_loader(self, gen_captions, model, tokenizer):

    dataset = PmcCaptionInferenceDataset(
      gen_captions, tokenizer, 
      max_source_len=self.cfg.model.caption.hf_model.max_source_len)

    data_collator = DataCollatorForSeq2Seq(
          tokenizer,
          model=model,
          label_pad_token_id=-100,
          pad_to_multiple_of=None,
      )
    
    loader = DataLoader(
        dataset,
        batch_size=self.cfg.eval.batch_size,
        num_workers=self.cfg.num_workers,
        pin_memory=self.cfg.gpu.use,
        collate_fn=data_collator,
        sampler=SequentialSampler(dataset),
    )
    return loader

  def eval(self, eval_loader, models, tokenizers, **kwargs):
    
    #Generate captions
    caption_tokens = self.generate_caption(eval_loader, models, tokenizers)
    caption_text = self.detokenize(caption_tokens, tokenizers['caption'])
    
    contexts = caption_tokens['context']
    captions = caption_text['caption']

    caption_loader = self.create_loader(captions, 
      models['chart_text'], tokenizers['chart_text']
      )
    
    chart_text_tokens = self.generate_chart_text(captions, models, tokenizers)
    chart_text_dict = self.detokenize(chart_text_tokens, tokenizers['chart_text'])

    chart_data = self.generate_codebook(caption_loader, models)

    self.logger.info("Completed generation. Starting save process.")

    chart_text_dict['caption'] = [c[0] for c in captions]
    chart_text_dict['axis_titles'] = chart_text_dict.pop('axis')

    chart_text_dict['categorical'] = chart_text_dict.pop('chart_text')

    chart_text_dict['categorical'] = [s for s in chart_text_dict['categorical'] if 'unnamed' not in s]
    chart_text_dict['series_name'] = [s for s in chart_text_dict['series_name'] if 'unnamed' not in s]

    self.create_mpl_plots(x=chart_data, text_data=chart_text_dict, save_dir=self.cfg.sample_dirs['generate'])

  def create_mpl_plots(self, x, text_data,save_dir):

    if 'chart_data' in x:
      x = x['chart_data']
    
    x_data_mpl = prepare_mpl(x)

    for idx, x_data in enumerate(x_data_mpl):
      f_name = os.path.join(save_dir, f"{idx}-{x_data['head_type']}.png")

      text = {}
      for k, v in text_data.items():
        text[k] = v[idx]

      #print("text", text)
      if x_data['head_type'] == 'categorical':
          create_bar_chart(x_data, text, f_name)
      elif x_data['head_type'] == 'point':
          create_scatter(x_data, text, f_name)
      elif x_data['head_type'] == 'boxplot':
          create_boxplot(x_data, text, f_name)
      else:
        raise ValueError(f"Invalid chart type given: {x_data['head_type']}")
        

  def save_raw_json(self, data, text, save_dir, idx):    
    output_fn = os.path.join(save_dir,  f"{idx}.json")
    with open(output_fn, 'w') as f:
      json.dump(data, f)

  def create_vega_json(self, chart_data, save_dir, idx):
    ''' chart_data: ['chart_type','row','col','continuous' '''
    chart_type = chart_data['chart_type']

    assert chart_type in ['point', 'categorical','boxplot']
    if chart_type == 'categorical':
      json_file = self.build_categorical_json(chart_data)
    elif chart_type == 'point':
      json_file = self.build_point_json(chart_data)
    else:
      return
    output_fn = os.path.join(save_dir,  f"{idx}_{chart_type}.json")
    
    with open(output_fn, 'w') as f:
      json.dump(json_file, f)

  def build_point_json(self,  chart_data):

    chart_type = chart_data['chart_type']
    continuous_data = chart_data['continuous']
    json_file = get_vega_template(chart_type)

    data = []
    values = []
    d = {"name": "table"}

    cols = min(chart_data['col'], len(continuous_data[0]))
    rows = min(chart_data['row'], len(continuous_data))

    for cidx, row_idx in enumerate(range(rows)): #By series name
      for col_idx in range(cols): #Right to left
        v = {}
        v['x'] = continuous_data[row_idx][col_idx][0]
        v['y'] = continuous_data[row_idx][col_idx][1]
        v['c'] = cidx
        values.append(v)    
    d['values'] = values #Add a list of dicts

    data.append(d)
    json_file['data'] = data
    return json_file

  def build_categorical_json(self,  chart_data):

    chart_type = chart_data['chart_type']
    continuous_data = chart_data['continuous']
    chart_text = chart_data['categorical']

    json_file = get_vega_template(chart_type)

    data = []
    values = []
    d = {"name": "table"}

    cols = min(chart_data['col'], len(chart_text), len(continuous_data[0]))
    rows = min(chart_data['row'], len(continuous_data))
    
    text_idx = 0
    for cidx, row_idx in enumerate(range(rows)): #By series name
      for col_idx in range(cols): #Right to left
        v = {}
        v['x'] = chart_text[col_idx]
        v['y'] = continuous_data[row_idx][col_idx][0]
        v['c'] = cidx

        text_idx += 1
        #Classes
        if rows > 1:
          v['c'] = cidx

        values.append(v)

    d['values'] = values #Add a list of dicts

    data.append(d)
    json_file['data'] = data
    return json_file

  def sample_indices(self, logits, temp=1.0):
    bsz = logits.size(0)
    probs = F.softmax(logits / temp, dim=-1).data
    probs = torch.flatten(probs, start_dim=0, end_dim=1)
    cb_indices =  torch.multinomial(probs, 1)
    cb_indices = cb_indices.reshape([bsz, -1])
    return cb_indices