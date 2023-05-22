# ---------------------------------------------------------------
# Copyright (c) _______ .
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import os
import json
import numpy as np 

import torch
from torch.nn import functional as F

from transformers.trainer_pt_utils import (
  find_batch_size, 
  nested_concat, 
  nested_numpify,
)

from .text import ChartTextRunner

from models.constant import UNIQ_CHART_HEADS


from utils import (
  TASK2PREPEND, 
  prepare_mpl,
)


class GenRunner(ChartTextRunner):
  def __init__(self, stage, cfg):
    super(GenRunner, self).__init__(stage, cfg)
    self.stage = stage
    self.discrete_input = cfg.data.dataset.chart_data.discrete_input
    self.gen_temperature = cfg.eval.gen_temperature
    self.gen_hypo_count = cfg.eval.hypo_count
    self.gen_hypo_bsz = cfg.eval.hypo_bsz

    self.tok_cb_map = None

    self.cb1_len = cfg.model.continuous_data.vq.emb_len1
    self.cb2_len = cfg.model.continuous_data.vq.emb_len2

  def set_model_output(self, models, mode):
    '''
    Swaps between text and data generation mode.
    '''
    
    if hasattr(models['seq'], 'module'):
      models['seq'].module.set_output(mode)
    else:
      models['seq'].set_output(mode) 

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
        text, max_length=self.cfg.model.seq.hf_model.max_source_len, 
          padding="max_length", truncation=True, return_tensors="pt")
      return inputs
  

  def _generate_caption(self, contexts, models, tokenizers):
    for model in models.values():
      model.eval()
    
    #Only run captions
    task = 'caption'
    task_str = TASK2PREPEND[task]

    #Check model name
    self.set_model_output(models, 'text')
    
    model = models['seq']
    tok = tokenizers['seq']

    #Append task to context
    task_contexts = [task_str + c for c in contexts]
    inputs = self.tokenize(task_contexts, tok)

    _, tokens, _, _ = self.prediction_step(
      model, tokenizer=tok, 
      inputs=inputs, prediction_loss_only=False)
    
    tokens = self._pad_across_processes(tokens)
    tokens = self._nested_gather(tokens)
    tokens = nested_numpify(tokens)

    return tokens
  
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

    #Check model name
    self.set_model_output(models, 'text')
    
    model = models['seq']
    tok = tokenizers['seq']

    for step, contexts in enumerate(iterator):
      
      if self.debug and step > 1:
        break

      if isinstance(self.cfg.eval.max_steps, int) and step > self.cfg.eval.max_steps:
        break
      
      preds_host['context'] = contexts if preds_host['context'] is None else preds_host['context'] + contexts
      #Append task to context
      task_contexts = [task_str + c for c in contexts]
      inputs = self.tokenize(task_contexts, tok)

      # Update the observed num examples
      observed_batch_size = find_batch_size(inputs)
      if observed_batch_size is not None:
          observed_num_examples += observed_batch_size
          # For batch samplers, batch_size is not known by the dataloader in advance.
          if batch_size is None:
              batch_size = observed_batch_size
              
      _, logits, _, _ = self.prediction_step(
        model, tokenizer=tok, 
        inputs=inputs, prediction_loss_only=False)
      

      logits = self._pad_across_processes(logits)
      logits = self._nested_gather(logits)
      preds_host[task] = logits if preds_host[task] is None else nested_concat(preds_host[task], logits, padding_index=-100)
      
    
      #Move collection to CPU
      #for task in TASK2PREPEND.keys():
      preds_host[task] = nested_numpify(preds_host[task])
      all_preds[task] = preds_host[task] if all_preds[task] is None else nested_concat(all_preds[task], preds_host[task], padding_index=-100)

      all_preds['context'] = preds_host['context'] if all_preds['context'] is None else all_preds['context'] + preds_host['context']
      
      #Reset containers
      preds_host = {t: None for t in all_tasks}

      
    # Gather all remaining tensors and put them back on the CPU

    #for task in TASK2PREPEND.keys():
    logits = nested_numpify(preds_host[task]) if preds_host[task] is not None else None
    if logits is not None:
      all_preds[task] = logits if all_preds[task] is None else nested_concat(all_preds[task], logits, padding_index=-100)
    
    return all_preds

  def caption_conditional_generation(self, all_captions, models, tokenizers, all_tasks):

    for model in models.values():
      model.eval()
    
    #Check model name and setup correct settings
    mode = 'text'
    if len(all_tasks) == 1 and all_tasks[0] == 'data':
      mode = 'data'

    model = models['seq']
    self.set_model_output(models, mode)
    tok = tokenizers['seq']

    # Initialize containers
    preds_host = {t: None for t in all_tasks}

    # losses/preds/labels on CPU (final containers)
    all_preds = {t: None for t in all_tasks}

    observed_num_examples = 0
    batch_size = self.bsz
    total_steps = int(np.ceil(len(all_captions) / batch_size))

    for step in range(total_steps):
      captions = all_captions[step * batch_size:step * batch_size + batch_size]

      #preds_host['captions'] = captions if preds_host['captions'] is None else preds_host['captions'] + captions
      #Loop through each task and generate
      for task in all_tasks:
        task_str = TASK2PREPEND[task]
        task_contexts = [task_str + c[0] for c in captions]

        inputs = self.tokenize(task_contexts, tok)

        # Update the observed num examples
        observed_batch_size = find_batch_size(inputs)
        if observed_batch_size is not None:
            observed_num_examples += observed_batch_size
            # For batch samplers, batch_size is not known by the dataloader in advance.
            if batch_size is None:
                batch_size = observed_batch_size
                
        _, logits, _, _ = self.prediction_step(
          model, tokenizer=tok, 
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

    for task in all_tasks:
      logits = nested_numpify(preds_host[task]) if preds_host[task] is not None else None
      if logits is not None:
        all_preds[task] = logits if all_preds[task] is None else nested_concat(all_preds[task], logits, padding_index=-100)
      
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

    return split_text

  def detokenize(self, tokens, tokenizer, seperator='<SEP>'):
    decoded = {}
    for task in tokens.keys():
      if task not in ['context']:
        skip_special_tokens = True if task == 'caption' else False
        decoded[task] = self.batch_decode(tokens[task], tokenizer, skip_special_tokens=skip_special_tokens)
      
    for task in decoded.keys():
      decoded[task] = [self.seperate_text_to_list(b, seperator=seperator) for b in decoded[task]]
    
    return decoded

  def generate_codebook(self, data_tokens, models):
    for m in models.values():
      if m is not None:
        m.eval()

    container = {}
    ct_idxs = []

    observed_num_examples = 0
    batch_size = self.bsz
    total_steps = int(np.ceil(data_tokens.shape[0] / batch_size))

    emb_len1 = self.cfg.model.continuous_data.vq.emb_len1
    emb_len2 = self.cfg.model.continuous_data.vq.emb_len2

    #Move to torch and gpu
    
    for step in range(total_steps):
      data_token = data_tokens[step * batch_size:step * batch_size + batch_size]
      data_token = torch.from_numpy(data_token).to(self.device)

      with torch.no_grad():
        with self.autocast_smart_context_manager():
          
          if models['continuous'].__class__.__name__ == 'DistributedDataParallel':
            cont_module = models['continuous'].module
          else:
            cont_module = models['continuous']
          
          ct_idx = data_token[:,:1]
          cb_ind1 = data_token[:,1:1 + emb_len1]
          cb_ind2 = data_token[:,1 + emb_len1:1 + emb_len1 + emb_len2]

          ct_idx = ct_idx - 2
          cb1    = cb1    - 2 - len(UNIQ_CHART_HEADS)
          if cb2 is not None:
            cb2  = cb2    - 2 - len(UNIQ_CHART_HEADS) #- self.cfg.model.continuous_data.vq.n_emb1

          ct_idxs.append(ct_idx)

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
            if k not in container: container[k] = {} 

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


  def eval(self, eval_loader, models, tokenizers, **kwargs):
    
    #Generate captions
    caption_tokens = self.generate_caption(eval_loader, models, tokenizers)
    caption_text = self.detokenize(caption_tokens, tokenizers['seq'])
    
    contexts = caption_tokens['context']
    captions = caption_text['caption']

    all_tasks = ['series_name', 'categorical', 'axis']
    discrete_tokens = self.caption_conditional_generation(captions, models, tokenizers, all_tasks=all_tasks)
    discrete_text = self.detokenize(discrete_tokens, tokenizers['seq'])

    all_tasks = ['data']
    data_tokens = self.caption_conditional_generation(captions, models, tokenizers, all_tasks=all_tasks)

    chart_data = self.generate_codebook(data_tokens['data'], models)
    if 'chart_data' in chart_data:
      chart_data = chart_data['chart_data']

    chart_data_mpl = prepare_mpl(chart_data)

    self.logger.info("Completed generation. Starting save process.")

    discrete_text['caption'] = [c[0] for c in captions]
    discrete_text['contexts'] = contexts

    self.create_raw_json(data=chart_data_mpl, text_data=discrete_text, save_dir=self.cfg.sample_dirs['generate']['json'])
  
  def create_raw_json(self, data, text_data, save_dir):
    
    for idx, x_data in enumerate(data):
      json_input = {}
      for k, v in text_data.items():
        json_input[k] = v[idx]
      
      json_input['data'] = x_data

      output_fn = os.path.join(save_dir,  f"{idx}.json")
      with open(output_fn, 'w') as f:
        json.dump(json_input, f)

  def sample_indices(self, logits, temp=1.0):
    bsz = logits.size(0)
    probs = F.softmax(logits / temp, dim=-1).data
    probs = torch.flatten(probs, start_dim=0, end_dim=1)
    cb_indices =  torch.multinomial(probs, 1)
    cb_indices = cb_indices.reshape([bsz, -1])
    return cb_indices