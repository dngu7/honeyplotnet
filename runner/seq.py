# ---------------------------------------------------------------
# Copyright (c) Cybersecurity Cooperative Research Centre 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

from filelock import FileLock
from typing import Dict, NamedTuple, Optional, Tuple, Union

import os
import nltk  
import time
import numpy as np
import torch
from torch.distributions.categorical import Categorical

from transformers.trainer_pt_utils import (
  find_batch_size, nested_concat, 
  nested_numpify, nested_truncate,
)

from transformers.trainer_utils import ( 
  EvalPrediction, denumpify_detensorize
)

from models.constant import CHART_TO_HEAD_IDX, HEAD_IDX_TO_CHART, UNIQ_CHART_HEADS
from dataset.base import shift_tokens_right_pad
from runner.text import ChartTextRunner
from fid import calculate_frechet_distance
from utils.constant import TASK2IDX

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

class EvalLoopOutputwInputs(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids_text: Optional[np.ndarray]
    label_ids_code: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    inputs: Optional[np.ndarray]

class SeqRunner(ChartTextRunner):
  def __init__(self, stage, cfg):
    super(SeqRunner, self).__init__(stage, cfg)
    self.loss_fn_ = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

  def set_model_output(self, models, mode):
    '''
    Swaps between text and data generation mode.
    '''
    if hasattr(models['seq'], 'module'):
      models['seq'].module.set_output(mode)
    else:
      models['seq'].set_output(mode) 

  def compute_loss(self, models, inputs, text_lbls=None, data_lbls=None, return_outputs=False, text_tasks=None):

    text_inputs = self._prepare_inputs(inputs['text'])
    
    self.set_model_output(models, 'both')

    text_logits, data_logits = models['seq'](**text_inputs)

    text_mask, data_mask = None, None
    if text_tasks is not None:
      text_mask = torch.tensor(text_tasks, dtype=torch.float32, device=self.device)
      data_mask = (text_mask == 0).to(torch.float32)

    loss = {}
    if text_lbls is not None:
      text_lbls = text_lbls.to(self.device)
      batch_sz, label_len = text_lbls.shape
      text_logits = text_logits[:,:label_len,:]
      flat_text_logits = torch.flatten(text_logits, start_dim=0, end_dim=1)
      text_loss = self.loss_fn_(flat_text_logits, text_lbls.view(-1))
      text_loss = torch.nan_to_num(text_loss, nan=1e-9)
      text_loss = torch.reshape(text_loss, (batch_sz, label_len))
      non_zeros_count = (text_loss > 1e-5).to(torch.float32)
      text_loss = text_loss.sum(-1) / non_zeros_count.sum(-1) if non_zeros_count.sum() > 0 else text_loss.sum()
      if text_mask is not None:
        text_loss = text_loss * text_mask
        text_loss = text_loss.sum() / text_mask.sum()  if text_mask.sum() > 0 else text_loss.sum()
      else:
        text_loss = text_loss.mean()
      
      loss['text'] = text_loss.mean() * self.cfg.train.loss_weights.text

    if data_lbls is not None:
      data_lbls = data_lbls.to(self.device)
      batch_sz, label_len = data_lbls.shape
      data_logits = data_logits[:,:label_len,:]
      flat_data_logits = torch.flatten(data_logits, start_dim=0, end_dim=1)
      data_loss = self.loss_fn_(flat_data_logits, data_lbls.view(-1))
      data_loss = torch.nan_to_num(data_loss, nan=1e-9)
      data_loss = torch.reshape(data_loss, (batch_sz, label_len))
      non_zeros_count = (data_loss > 1e-5).to(torch.float32)
      data_loss = data_loss.sum(-1) / non_zeros_count.sum(-1) if non_zeros_count.sum() > 0 else data_loss.sum()
      if data_mask is not None:
        data_loss = data_loss * data_mask
        data_loss = data_loss.sum() / data_mask.sum() if data_mask.sum() > 0 else data_loss.sum()
      else:
        data_loss = data_loss.mean()
      
      loss['data'] = data_loss.mean() * self.cfg.train.loss_weights.code
    
    outputs = {'text': text_logits, 'data': data_logits}
    return (loss, outputs, ) if return_outputs else loss
  
  def sample_data_labels(self, models, inputs, tokenizers):
    
    models['continuous'].eval()

    with self.autocast_smart_context_manager():
      with torch.no_grad():
        if hasattr(models['continuous'], 'module'):
          cb1 = models['continuous'].module.sample_codebook(inputs['data'])
        else:
          cb1 = models['continuous'].sample_codebook(inputs['data'])

      if len(cb1) == 2:
        cb1, cb2 = cb1
      else:
        cb1, cb2 = cb1[0], None

      ct_idx = [CHART_TO_HEAD_IDX[ct] for ct in inputs['data']['chart_type']]
      ct_idx = torch.tensor(ct_idx, dtype=torch.long, device=self.device).view(-1,1)

      #################################
      # OFFSETS for the data codebook
      # ct_idx: + 2 
      # cb1   : + 2 + 3 (unique charts)
      # cb2   : + 2 + 3 + (cfg.model.continuous_data.vq.n_emb1)
      ct_idx = ct_idx + 2
      cb1    = cb1    + 2 + len(UNIQ_CHART_HEADS)
      if cb2 is not None:
        cb2  = cb2    + 2 + len(UNIQ_CHART_HEADS) #+ self.cfg.model.continuous_data.vq.n_emb1

      code_labels = torch.cat([ct_idx, cb1], dim=-1)
      if cb2 is not None:
        code_labels = torch.cat([code_labels, cb2], dim=-1)

      eos_token_id = tokenizers['seq'].eos_token_id
      pad_token_id = tokenizers['seq'].pad_token_id

      #Add eos token
      padding = torch.ones([code_labels.shape[0], 1], dtype=torch.long, device=self.device) 
      code_labels = torch.cat([code_labels, padding * eos_token_id], dim=-1)
      #Shift all labels up by one because 0 is reserved for start token
      decoder_input_ids = shift_tokens_right_pad(code_labels, pad_token_id=pad_token_id)

    return code_labels, decoder_input_ids

  def training_step(self, models, inputs, tokenizers, text_tasks=None):
    
    models['seq'].train()

    data_lbls, decoder2_input_ids = self.sample_data_labels(models, inputs, tokenizers)
    
    text_lbls = inputs['text'].pop("labels")
    inputs['text']['decoder2_input_ids'] = decoder2_input_ids

    loss_dict = self.compute_loss(models, inputs, 
        text_lbls=text_lbls, data_lbls=data_lbls, text_tasks=text_tasks)
  
    loss_log = {}
    total_loss = 0.0
    for name, loss in loss_dict.items():
      total_loss += loss
      loss_log[name] = loss.detach().cpu()
  
    if self.gradient_accum_steps > 1 :
      total_loss = total_loss / self.gradient_accum_steps
    
    if self.do_grad_scaling:
      self.scaler.scale(total_loss).backward()
    else:
      total_loss.backward()

    return loss_log

  def train(self, train_loader, models, tokenizers, opts, schs):   

    self.tracker.reset_all()

    tr_loss = torch.tensor(0.0).to(self.device_id)

    for stage, m in models.items():
      if stage == self.stage:
        m.train()
        m.zero_grad()
      else:
        m.eval()
    
    for o in opts.values():
      if o is not None:
        o.zero_grad()    

    iterator = train_loader.__iter__()
    steps_in_epoch = len(iterator)
    text_tasks = None
    for step, (_, inputs) in enumerate(iterator):

      # if self.debug and step > 1:
      #    break
      
      if self.cfg.model.seperate_data_task:
        text_tasks = [1 if t != 'data' else 0 for t in  inputs['task']]

      loss_log = self.training_step(models, inputs, tokenizers, text_tasks=text_tasks)

      tr_loss_step = sum(list(loss_log.values()))
      tr_loss += tr_loss_step
      
      self.tracker.add_logs(split='train', log=loss_log, total_loss=tr_loss_step)


      if (step + 1) % self.gradient_accum_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accum_steps
                    steps_in_epoch <= self.gradient_accum_steps
                    and (step + 1) == steps_in_epoch
                ):
        
        if self.do_grad_scaling:
          self.scaler.unscale_(opts[self.stage])
        if self.max_grad_norm > 0:
          torch.nn.utils.clip_grad_norm_(models[self.stage].parameters(), self.max_grad_norm)

        if self.do_grad_scaling:
          self.scaler.step(opts[self.stage])
          self.scaler.update()
        else:
          opts[self.stage].step()

        models[self.stage].zero_grad()
        opts[self.stage].zero_grad()

        self.global_step += 1
        tr_loss = 0
        
      if isinstance(self.display, int) and step % self.display == 0 and step > 0:
        self.logger.info("E{:02d} GS: {:03d} {}".format(
          self.epoch, self.global_step, self.tracker.loss_str('iter')))
        
        self.tracker.reset_interval('iter')

    if schs[self.stage] is not None:
       schs[self.stage].step()
       
    self.logger.info("E{:02d} (train) {}".format(self.epoch, self.tracker.loss_str('epoch')))
    self.update_writer('train')

  def prediction_step(self, models, tokenizers, inputs, prediction_loss_only=False, text_tasks=None, generate_tokens=False):
    
    has_labels = 'labels' in inputs['text']
    
    for m in models.values():
       m.eval()

    text_inputs = self._prepare_inputs(inputs['text'])
    text_tokens = None
    data_tokens = None
    if generate_tokens:
      gen_kwargs = {
              "input_ids": text_inputs["input_ids"],
              "max_length": self._max_length,
              "num_beams": self._num_beams,
              "synced_gpus": False,
              "repetition_penalty": self._repetition_penalty,
              "temperature": self._gen_temperature,
          }

      if "attention_mask" in text_inputs:
        gen_kwargs["attention_mask"] = text_inputs.get("attention_mask", None)
      if "global_attention_mask" in inputs:
        gen_kwargs["global_attention_mask"] = text_inputs.get("global_attention_mask", None)

      tasks = self.cfg.data.dataset.tasks

      #Generate text
      self.set_model_output(models, 'text')
      
      if any(t in tasks for t in ['categorical','series_name','axis','caption']):
        with torch.no_grad():
          if models['seq'].__class__.__name__ == 'DistributedDataParallel':
            text_tokens = models['seq'].module.generate(**gen_kwargs)      
          else:
            text_tokens = models['seq'].generate(**gen_kwargs)

      #Generate data
      for l in ['max_length', 'num_beams', 'synced_gpus', 'repetition_penalty', 'temperature']:
        gen_kwargs.pop(l)

      self.set_model_output(models, 'data')
      
      if 'data' in tasks:
        with torch.no_grad():
          if models['seq'].__class__.__name__ == 'DistributedDataParallel':
            data_tokens = models['seq'].module.generate(**gen_kwargs)      
          else:
            data_tokens = models['seq'].generate(**gen_kwargs)

      # in case the batch is shorter than max length, the output should be padded
      if text_tokens is not None and text_tokens.shape[-1] < self._max_length:
          text_tokens = self._pad_tensors_to_max_len(
            text_tokens, self._max_length, models['seq'], tokenizers['seq'])

    tr_loss = 0.0
    text_lbls = None
    data_lbls = None
    if has_labels:
      with torch.no_grad():
        text_lbls = inputs['text'].pop("labels")

        data_lbls, decoder2_input_ids = self.sample_data_labels(models, inputs, tokenizers=tokenizers)
        inputs['text']['decoder2_input_ids'] = decoder2_input_ids

        losses, outputs = self.compute_loss(
           models, inputs, text_lbls=text_lbls, data_lbls=data_lbls, 
           return_outputs=True, text_tasks=text_tasks)
        
        if not generate_tokens:
          text_logits, data_logits = outputs['text'], outputs['data']
          text_tokens = Categorical(logits=text_logits).sample()
          data_tokens = Categorical(logits=data_logits).sample()

        loss_log = {}
        for k,v in losses.items():
          tr_loss += v.detach()
          loss_log[k] = v.cpu().detach().item()

        self.tracker.add_logs(split='eval', log=loss_log, total_loss=tr_loss)
        
    if prediction_loss_only:
        return (tr_loss, None, None)

    if text_lbls is not None and text_lbls.shape[-1] < self._max_length:
        text_lbls = self._pad_tensors_to_max_len(text_lbls, 
          self._max_length, models['seq'], tokenizers['seq'])

    return (tr_loss, text_tokens, data_tokens, inputs, text_lbls, data_lbls)
  
  def eval_loop(self, cur_stage, loader, models, tokenizers, metric_key_prefix='eval', prediction_loss_only=False, step_count=None):
    
    self.tracker.reset_all()
    iterator = loader.__iter__()
    steps_in_epoch = len(iterator)

    models[cur_stage].eval()

    # Initialize containers
    # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
    losses_host = None
    preds_host_text = None
    preds_host_data = None
    labels_host_text = None
    labels_host_data = None
    fidact_host_data = None
    inputs_host = None
    task_host = None

    # losses/preds/labels on CPU (final containers)
    all_losses = None
    all_text_preds = None
    all_data_preds = None
    all_labels_text = None
    all_labels_data = None
    all_fidact_data = None
    all_inputs = None
    all_tasks = None

    text_tasks = None
    fid_act = None

    observed_num_examples = 0
    batch_size = self.bsz

    start_time = time.time()

    max_step_count = self.cfg.eval.max_steps if step_count is None and isinstance(self.cfg.eval.max_steps, int) else step_count
    max_step_count = min(steps_in_epoch, max_step_count) if max_step_count is not None else steps_in_epoch

    for step, (_, inputs) in enumerate(iterator): 

      if (self.debug and step >= 6) or (max_step_count is not None and step > max_step_count):
         break 

      if isinstance(self.cfg.eval.display_interval, int) and step % self.cfg.eval.display_interval == 0 and step > 0:
        self.logger.info("Eval | {}/{} Time elapsed : {:.2f}s".format(step, max_step_count, time.time() - start_time))
      
      if self.cfg.model.seperate_data_task:
        text_tasks = [1 if t != 'data' else 0 for t in  inputs['task']]
        task_idx = torch.tensor([TASK2IDX[t] for t in inputs['task']], dtype=torch.long, device=self.device)

      # Update the observed num examples
      observed_batch_size = find_batch_size(inputs)
      if observed_batch_size is not None:
          observed_num_examples += observed_batch_size
          # For batch samplers, batch_size is not known by the dataloader in advance.
          if batch_size is None:
              batch_size = observed_batch_size

      start_time = time.time()

      loss, text_tokens, data_tokens, inputs, text_lbls, data_lbls = self.prediction_step(
         models, tokenizers=tokenizers,
        inputs=inputs, prediction_loss_only=prediction_loss_only,
        text_tasks=text_tasks, 
        generate_tokens=True)
      if self.cfg.eval.fid:
        fid_act = self.compute_fid_acts(models, data_tokens)

      inputs_decode = inputs['text']["input_ids"] 

      # Update containers on host
      if loss is not None:
        losses = self._nested_gather(loss.repeat(batch_size))
        losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)

      if text_lbls is not None:
        text_lbls = text_lbls.to(self.device)
        text_lbls = self._pad_across_processes(text_lbls)
        text_lbls = self._nested_gather(text_lbls)
        labels_host_text = text_lbls if labels_host_text is None else nested_concat(labels_host_text, text_lbls, padding_index=-100)

      if data_lbls is not None:
        data_lbls = data_lbls.to(self.device)
        data_lbls = self._pad_across_processes(data_lbls)
        data_lbls = self._nested_gather(data_lbls)
        labels_host_data = data_lbls if labels_host_data is None else nested_concat(labels_host_data, data_lbls, padding_index=-100)

      if fid_act is not None:
        fid_act = fid_act.to(self.device)
        fid_act = self._pad_across_processes(fid_act)
        fid_act = self._nested_gather(fid_act)
        fidact_host_data = fid_act if fidact_host_data is None else nested_concat(fidact_host_data, fid_act, padding_index=-100)

      if inputs_decode is not None:
        inputs_decode = inputs_decode.to(self.device)
        inputs_decode = self._pad_across_processes(inputs_decode)
        inputs_decode = self._nested_gather(inputs_decode)
        inputs_host = (
            inputs_decode
            if inputs_host is None
            else nested_concat(inputs_host, inputs_decode, padding_index=-100)
        )
      
      if text_tokens is not None:
        text_tokens = self._pad_across_processes(text_tokens)
        text_tokens = self._nested_gather(text_tokens)
        preds_host_text = text_tokens if preds_host_text is None else nested_concat(preds_host_text, text_tokens, padding_index=-100)

      if data_tokens is not None:
        data_tokens = data_tokens.contiguous()
        data_tokens = self._pad_across_processes(data_tokens)
        data_tokens = self._nested_gather(data_tokens)
        preds_host_data = data_tokens if preds_host_data is None else nested_concat(preds_host_data, data_tokens, padding_index=-100)

      if task_idx is not None:
        task_idx = self._pad_across_processes(task_idx)
        task_idx = self._nested_gather(task_idx)
        task_host = task_idx if task_host is None else nested_concat(task_host, task_idx, padding_index=-100)

      # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
      if self.eval_accumulation_steps is not None and (step + 1) % self.eval_accumulation_steps == 0:
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host_text is not None:
            text_tokens = nested_numpify(preds_host_text)
            all_text_preds = text_tokens if all_text_preds is None else nested_concat(all_text_preds, text_tokens, padding_index=-100)
        
        if preds_host_data is not None:
            data_tokens = nested_numpify(preds_host_data)
            all_data_preds = data_tokens if all_data_preds is None else nested_concat(all_data_preds, data_tokens, padding_index=-100)

        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode
                if all_inputs is None
                else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host_text is not None:
            text_lbls = nested_numpify(labels_host_text)
            all_labels_text = (
                text_lbls if all_labels_text is None else nested_concat(all_labels_text, text_lbls, padding_index=-100)
            )
        if labels_host_data is not None:
            data_lbls = nested_numpify(labels_host_data)
            all_labels_data = (
                data_lbls if all_labels_data is None else nested_concat(all_labels_data, data_lbls, padding_index=-100)
            )
        
        if fidact_host_data is not None:
            fid_act = nested_numpify(fidact_host_data)
            all_fidact_data = (
                fid_act if all_fidact_data is None else nested_concat(all_fidact_data, fid_act, padding_index=-100)
            )

        if task_host is not None:
            task_idx = nested_numpify(task_host)
            all_tasks = task_idx if all_tasks is None else nested_concat(all_tasks, task_idx, padding_index=-100)

        # Set back to None to begin a new accumulation
        losses_host, preds_host_text, preds_host_data, inputs_host, \
          labels_host_text, labels_host_data, task_host, fidact_host_data = \
          None, None, None, None, None, None, None, None


    # Gather all remaining tensors and put them back on the CPU
    if losses_host is not None:
        losses = nested_numpify(losses_host)
        all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    if preds_host_text is not None:
        text_tokens = nested_numpify(preds_host_text)
        all_text_preds = text_tokens if all_text_preds is None else nested_concat(all_text_preds, text_tokens, padding_index=-100)
    if preds_host_data is not None:
        data_tokens = nested_numpify(preds_host_data)
        all_data_preds = data_tokens if all_data_preds is None else nested_concat(all_data_preds, data_tokens, padding_index=-100)
    if inputs_host is not None:
        inputs_decode = nested_numpify(inputs_host)
        all_inputs = inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
    if labels_host_text is not None:
        text_lbls = nested_numpify(labels_host_text)
        all_labels_text = text_lbls if all_labels_text is None else nested_concat(all_labels_text, text_lbls, padding_index=-100)
    if labels_host_data is not None:
        data_lbls = nested_numpify(labels_host_data)
        all_labels_data = data_lbls if all_labels_data is None else nested_concat(all_labels_data, data_lbls, padding_index=-100)
    if fidact_host_data is not None:
        fid_act = nested_numpify(fidact_host_data)
        all_fidact_data = fid_act if all_fidact_data is None else nested_concat(all_fidact_data, fid_act, padding_index=-100)
    if task_host is not None:
        task_idx = nested_numpify(task_host)
        all_tasks = task_idx if all_tasks is None else nested_concat(all_tasks, task_idx, padding_index=-100)

    num_samples = steps_in_epoch
    # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
    # samplers has been rounded to a multiple of batch_size, so we truncate.
    if all_losses is not None:
        all_losses = all_losses[:num_samples]
    if all_text_preds is not None:
        all_text_preds = nested_truncate(all_text_preds, num_samples)
    if all_data_preds is not None:
        all_data_preds = nested_truncate(all_data_preds, num_samples)
    if all_labels_text is not None:
        all_labels_text = nested_truncate(all_labels_text, num_samples)
    if all_labels_data is not None:
        all_labels_data = nested_truncate(all_labels_data, num_samples)
    if all_fidact_data is not None:
        all_fidact_data = nested_truncate(all_fidact_data, num_samples)
    if all_inputs is not None:
        all_inputs = nested_truncate(all_inputs, num_samples)
    if all_tasks is not None:
       all_tasks = all_tasks[:num_samples]
  
    # Metrics for text
    metrics = {}    
    fid_train, fid_test = None, None
    text_preds = all_text_preds
    text_labels = all_labels_text

    if self.compute_fid is not None and all_fidact_data is not None and self.cfg.eval.fid:
      data_indices = np.where(all_tasks == TASK2IDX['data'])[0]
      data_acts   = all_fidact_data[data_indices]
      fid_train, fid_test = self.compute_fid(data_acts)

    if self.compute_metrics is not None and text_preds is not None and text_labels is not None:
      # if step_count is not None:
      text_indices = np.where(all_tasks != TASK2IDX['data'])[0]
      text_preds   = all_text_preds[text_indices]
      text_labels  = all_labels_text[text_indices]

      if text_preds.shape[0] > 0:
        metrics = self.compute_metrics(EvalPrediction(predictions=text_preds, 
                          label_ids=text_labels),
                          tokenizers[cur_stage])
    
    metrics = denumpify_detensorize(metrics)

    if all_losses is not None:
        metrics["loss"] = all_losses.mean().item()
    
    if fid_train is not None:
       metrics['fid_train'] = fid_train
       metrics['fid_test'] = fid_test

    return EvalLoopOutputwInputs(predictions=all_text_preds, label_ids_text=all_labels_text,  label_ids_code=all_labels_data, 
      metrics=metrics, num_samples=num_samples, inputs=all_inputs), all_inputs


  def eval(self, val_loader, models, tokenizers, metric_key_prefix='eval', epoch=0, prediction_loss_only=False, step_count=None, **kwargs):
    
    for m in models.values():
      if m is not None:
        m.eval()

    predict_results, all_inputs = self.eval_loop(self.stage, val_loader, models, tokenizers, metric_key_prefix=metric_key_prefix, prediction_loss_only=prediction_loss_only, step_count=step_count)

    if self.rank() == 0 and step_count is None and predict_results.predictions is not None:
      predictions = tokenizers[self.stage].batch_decode(
          predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
      )

      if all_inputs is not None:
        contexts = tokenizers[self.stage].batch_decode(
            all_inputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
      else:
        contexts = [''] * len(predictions)

      ksm_scores = self.compute_ksm_scores(
          predict_results.predictions, predict_results.label_ids_text, 
          contexts, models, tokenizers, seperator='<SEP>')
      
      self.tracker.add_metrics(ksm_scores, metric_key_prefix, 'ksm')

      if self.cfg.eval.write_samples:
        print(f"Writing samples to output: {self.cfg.sample_dirs[self.stage]}")
        for pidx, (pred, context) in enumerate(zip(predictions, contexts)):
          task = context.split(':')[0].lower().strip()
          if 'data' not in task:
            task_directory = os.path.join(self.cfg.sample_dirs[self.stage], task)
            os.makedirs(task_directory, exist_ok=True)

            output_prediction_file = os.path.join(
                task_directory, "e{}_{}.txt".format(epoch, pidx))
            
            text = "OUTPUT: {} \n\n INPUT: {}".format(pred, context)
            with open(output_prediction_file, "w") as writer:
              writer.write(text)

    metrics = predict_results.metrics
    self.tracker.add_metrics(metrics, metric_key_prefix, 'rouge')

    self.logger.info("E{:02d} (eval) {} {}".format(self.epoch, self.tracker.loss_str('epoch'), self.tracker.metric_str('epoch', stage=self.stage)))

    #return predict_results
    opt_mode = self.cfg.model.seq.opt_mode
    outputs= {}
    if opt_mode == 0:
      outputs = {'score': metrics.get('loss')}
    elif opt_mode == 1:
      outputs = {'score': metrics.get('rouge2')}
    elif opt_mode == 2:
      outputs = {'score': metrics.get('fid_test')}
    return outputs

  def compute_fid_acts(self, models, data_tokens):
    start_time = time.time()
    assert 'fid' in models and 'continuous' in models
    for m in models.values():
      if m is not None:
        m.eval()
    
    emb_len1 = self.cfg.model.continuous_data.vq.emb_len1
    emb_len2 = self.cfg.model.continuous_data.vq.emb_len2
    n_emb1 = self.cfg.model.continuous_data.vq.n_emb1
    n_emb2 = self.cfg.model.continuous_data.vq.n_emb2

    ct_idx = data_tokens[:,:1]
    cb_ind1 = data_tokens[:,1:1 + emb_len1]
    cb_ind2 = data_tokens[:,1 + emb_len1:1 + emb_len1 + emb_len2]
    
    ct_idx = ct_idx - 2
    cb_ind1    = cb_ind1 - 2 - len(UNIQ_CHART_HEADS)
    if cb_ind2 is not None:
      cb_ind2  = cb_ind2 - 2 - len(UNIQ_CHART_HEADS) 

    ct_idx = torch.clamp(ct_idx, min=0, max=len(UNIQ_CHART_HEADS)-1)
    cb_ind1 = torch.clamp(cb_ind1, min=0, max=n_emb1-1)
    cb_ind2 = torch.clamp(cb_ind2, min=0, max=n_emb2-1)

    kwargs = {
      'cb_ind1': cb_ind1,
      'cb_ind2': cb_ind2,
      'ct_idx': ct_idx,
      'temp': self.cfg.eval.gen_temperature,
      'hypo_count': self.cfg.eval.hypo_count,
      'hypo_bsz': self.cfg.eval.hypo_bsz
    }


    with torch.no_grad():
        with self.autocast_smart_context_manager():
          if models['continuous'].__class__.__name__ == 'DistributedDataParallel':
            x_hat = models['continuous'].module.reconstruct_from_indices(**kwargs)
          else:
            x_hat = models['continuous'].reconstruct_from_indices(**kwargs)


          x_hat['chart_type'] = [HEAD_IDX_TO_CHART[m] for m in ct_idx.view(-1).detach().cpu().numpy()]
          activations, _, _ = models['fid'](x_hat)
          
          #activations = torch.from_numpy(activations)

    return activations

  def compute_fid(self, act_container):
    
    #act = np.concatenate(act_container, axis=0)
    mu = np.mean(act_container, axis=0)
    sigma = np.cov(act_container, rowvar=False)
    
    #Load existing fid scores
    m1, s1, _ = self.fid_stats['train']
    m2, s2, _ = self.fid_stats['test']

    train_fid = calculate_frechet_distance(mu, sigma, m1, s1)
    test_fid = calculate_frechet_distance(mu, sigma, m2, s2)

    return train_fid, test_fid
