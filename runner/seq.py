# ---------------------------------------------------------------
# Copyright (c) ___ Limited 2023.
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

from transformers.trainer_pt_utils import (
  find_batch_size, nested_concat, 
  nested_numpify, nested_truncate,
)


from transformers.trainer_utils import ( 
  EvalPrediction, denumpify_detensorize
)

from models.constant import CHART_TO_HEAD_IDX

from .text import ChartTextRunner

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

  def compute_loss(self, models, inputs, text_lbls=None, code_lbls=None, return_outputs=False):

    text_inputs = self._prepare_inputs(inputs['text'])

    models['seq'].set_output('both')
    text_logits, code_logits = models['seq'](**text_inputs)

    loss = {}
    if text_lbls is not None:
      
      label_len = text_lbls.shape[-1]
      text_logits = text_logits[:,:label_len,:]
      flat_text_logits = torch.flatten(text_logits, start_dim=0, end_dim=1)
      text_loss = self.loss_fn(flat_text_logits, text_lbls.view(-1))

      if not text_loss.isnan().any():
         loss['text'] = text_loss
      
    if code_lbls is not None:
      label_len = code_lbls.shape[-1]
      code_logits = code_logits[:,:label_len,:]
      flat_code_logits = torch.flatten(code_logits, start_dim=0, end_dim=1)
      code_loss = self.loss_fn(flat_code_logits, code_lbls.view(-1))

      if not code_loss.isnan().any():
         loss['code'] = code_loss
    
    outputs = {'text': text_logits, 'code': code_logits}
    return (loss, outputs, ) if return_outputs else loss
  
  def sample_code_labels(self, models, inputs):

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
      code_labels = torch.cat([ct_idx, cb1], dim=-1)
      if cb2 is not None:
        code_labels = torch.cat([code_labels, cb2], dim=-1)
         

      code_mask = torch.tensor([1 if c == 'data' else 0 for c in inputs['task']], dtype=torch.long, device=self.device).unsqueeze(-1)
      inv_code_mask = torch.tensor([0 if c == 'data' else -100 for c in inputs['task']],dtype=torch.long, device=self.device).unsqueeze(-1)
      code_labels = code_labels * code_mask + inv_code_mask
      
      #Create padding
      pad_len = models['seq'].config.max_length - code_labels.shape[-1]
      padding = torch.ones([code_labels.shape[0], pad_len], dtype=torch.long, device=self.device) * -100
      code_labels = torch.cat([code_labels, padding], dim=-1)

    return code_labels

  def training_step(self, models, inputs):
    
    models['seq'].train()

    with self.autocast_smart_context_manager():
      code_lbls = self.sample_code_labels(models, inputs)
      
      _ = inputs['text'].pop('text')
      text_lbls = inputs['text'].pop("labels")

      loss_dict = self.compute_loss(models, inputs, text_lbls=text_lbls, code_lbls=code_lbls)
  
    loss_log = {}
    total_loss = 0.0
    for name, loss in loss_dict.items():
      total_loss += loss
      loss_log[name] = loss.detach().cpu()
    
    if self.gradient_accum_steps > 1:
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

    for step, (_, inputs) in enumerate(iterator):


      loss_log = self.training_step(models, inputs)

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

    self.logger.info("E{:02d} (train) {}".format(self.epoch, self.tracker.loss_str('epoch')))
    self.update_writer('train')

  def prediction_step(self, models, tokenizer, inputs, prediction_loss_only=False, ignore_keys=[]):
    has_labels = 'labels' in inputs['text']

    if 'text' in inputs['text']:
      _ = inputs['text'].pop('text')

    text_inputs = self._prepare_inputs(inputs['text'])

    gen_kwargs = {
            "max_length": self._max_length,
            "num_beams": self._num_beams,
            "synced_gpus": False,
            "repetition_penalty": self._repetition_penalty,
            "temperature": self._gen_temperature,
            "return_text_only": True
        }

    if "attention_mask" in text_inputs:
            gen_kwargs["attention_mask"] = text_inputs.get("attention_mask", None)
    if "global_attention_mask" in inputs:
        gen_kwargs["global_attention_mask"] = text_inputs.get("global_attention_mask", None)
    
    # prepare generation inputs
    generation_inputs = text_inputs["input_ids"]
    models['seq'].set_output('text')

    if models['seq'].__class__.__name__ == 'DistributedDataParallel':
      generated_tokens = models['seq'].module.generate(
          generation_inputs,
          **gen_kwargs,
      )      
    else:

      generated_tokens = models['seq'].generate(
          generation_inputs,
          **gen_kwargs,
      )
    # in case the batch is shorter than max length, the output should be padded
    if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
        generated_tokens = self._pad_tensors_to_max_len(
          generated_tokens, gen_kwargs["max_length"], models['seq'], tokenizer)

    tr_loss = 0.0
    text_lbls = None
    code_lbls = None
    if has_labels:
      with torch.no_grad():
        with self.autocast_smart_context_manager():
          text_lbls = inputs['text'].pop("labels")
          code_lbls = self.sample_code_labels(models, inputs)
          losses = self.compute_loss(models, inputs, text_lbls=text_lbls, code_lbls=code_lbls, return_outputs=False)

          loss_log = {}
          for k,v in losses.items():
            if not v.isnan().any():
              loss_log[k] = v.mean().cpu().detach().item()
              tr_loss += v.mean().detach()

          self.tracker.add_logs(split='eval', log=loss_log, total_loss=tr_loss)
        
    if prediction_loss_only:
        return (tr_loss, None, None)

    if text_lbls is not None:
        if text_lbls.shape[-1] < gen_kwargs["max_length"]:
            text_lbls = self._pad_tensors_to_max_len(text_lbls, gen_kwargs["max_length"], models['seq'], tokenizer)

    return (tr_loss, generated_tokens, inputs, text_lbls, code_lbls)
  
  def eval_loop(self, cur_stage, loader, models, tokenizers, metric_key_prefix='eval', prediction_loss_only=False, test_count=None):
    
    self.tracker.reset_all()

    iterator = loader.__iter__()
    steps_in_epoch = len(iterator)

    # Initialize containers
    # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
    losses_host = None
    preds_host = None
    labels_host_text = None
    labels_host_code = None
    inputs_host = None

    # losses/preds/labels on CPU (final containers)
    all_losses = None
    all_preds = None
    all_labels_text = None
    all_labels_code = None
    all_inputs = None
    observed_num_examples = 0
    batch_size = self.bsz

    start_time = time.time()

    for step, (_, inputs) in enumerate(iterator):

      if test_count is not None and step > test_count:
         break
      else:
         self.logger.info("Eval | {}/{} Time elapsed : {:.2f}s".format(step, test_count, time.time() - start_time))
      
      if isinstance(self.display, int) and step % self.display == 0 and step > 0:
        self.logger.info("Eval | E{:02d} Step {:04d}/{:04d} ".format(self.epoch, step, self.cfg.eval.max_steps))

      if isinstance(self.cfg.eval.max_steps, int) and step > self.cfg.eval.max_steps:
        break

      # Update the observed num examples
      observed_batch_size = find_batch_size(inputs)
      if observed_batch_size is not None:
          observed_num_examples += observed_batch_size
          # For batch samplers, batch_size is not known by the dataloader in advance.
          if batch_size is None:
              batch_size = observed_batch_size
      
      loss, logits, inputs, text_lbls, code_lbls = self.prediction_step(
        models, tokenizer=tokenizers[cur_stage], 
        inputs=inputs, prediction_loss_only=prediction_loss_only)

      inputs_decode = inputs['text']["input_ids"] 

      # Update containers on host
      if loss is not None:
        losses = self._nested_gather(loss.repeat(batch_size))
        losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)

      if text_lbls is not None:
        text_lbls = self._pad_across_processes(text_lbls)
        text_lbls = self._nested_gather(text_lbls)
        labels_host_text = text_lbls if labels_host_text is None else nested_concat(labels_host_text, text_lbls, padding_index=-100)
        labels_host_code = code_lbls if labels_host_code is None else nested_concat(labels_host_code, code_lbls, padding_index=-100)

      if inputs_decode is not None:
        inputs_decode = self._pad_across_processes(inputs_decode)
        inputs_decode = self._nested_gather(inputs_decode)
        inputs_host = (
            inputs_decode
            if inputs_host is None
            else nested_concat(inputs_host, inputs_decode, padding_index=-100)
        )
      
      if logits is not None:
        logits = self._pad_across_processes(logits)
        logits = self._nested_gather(logits)

        preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
      
      # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
      if self.eval_accumulation_steps is not None and (step + 1) % self.eval_accumulation_steps == 0:
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
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
        if labels_host_code is not None:
            code_lbls = nested_numpify(labels_host_code)
            all_labels_code = (
                code_lbls if all_labels_code is None else nested_concat(all_labels_code, code_lbls, padding_index=-100)
            )

        # Set back to None to begin a new accumulation
        losses_host, preds_host, inputs_host, labels_host_text, labels_host_code = None, None, None, None, None


    # Gather all remaining tensors and put them back on the CPU
    if losses_host is not None:
        losses = nested_numpify(losses_host)
        all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    if preds_host is not None:
        logits = nested_numpify(preds_host)
        all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    if inputs_host is not None:
        inputs_decode = nested_numpify(inputs_host)
        all_inputs = (
            inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
        )
    if labels_host_text is not None:
        text_lbls = nested_numpify(labels_host_text)
        all_labels_text = text_lbls if all_labels_text is None else nested_concat(all_labels_text, text_lbls, padding_index=-100)
    if labels_host_code is not None:
        code_lbls = nested_numpify(labels_host_code)
        all_labels_code = code_lbls if all_labels_code is None else nested_concat(all_labels_code, code_lbls, padding_index=-100)

    num_samples = steps_in_epoch
    # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
    # samplers has been rounded to a multiple of batch_size, so we truncate.
    if all_losses is not None:
        all_losses = all_losses[:num_samples]
    if all_preds is not None:
        all_preds = nested_truncate(all_preds, num_samples)
    if all_labels_text is not None:
        all_labels_text = nested_truncate(all_labels_text, num_samples)
    if all_labels_code is not None:
        all_labels_code = nested_truncate(all_labels_code, num_samples)
    if all_inputs is not None:
        all_inputs = nested_truncate(all_inputs, num_samples)
  
    # Metrics!
    if self.compute_metrics is not None and all_preds is not None and all_labels_text is not None:
      if self.include_inputs_for_metrics:
          metrics = self.compute_metrics(
              EvalPrediction(predictions=all_preds, label_ids=all_labels_text, inputs=all_inputs),
              tokenizers[cur_stage]
          )
      else:
          metrics = self.compute_metrics(
            EvalPrediction(predictions=all_preds, label_ids=all_labels_text),
            tokenizers[cur_stage])
    else:
        metrics = {}
    
    metrics = denumpify_detensorize(metrics)

    if all_losses is not None:
        metrics["loss"] = all_losses.mean().item()

    
    return EvalLoopOutputwInputs(predictions=all_preds, label_ids_text=all_labels_text,  label_ids_code=all_labels_code, 
                                 metrics=metrics, num_samples=num_samples, inputs=all_inputs), all_inputs


  def eval(self, val_loader, models, tokenizers, metric_key_prefix='eval', epoch=0, temp=1.0, create_sample=False, prediction_loss_only=False, test_count=None, **kwargs):
    
    for m in models.values():
      if m is not None:
        m.eval()

    predict_results, all_inputs = self.eval_loop(self.stage, val_loader, models, tokenizers, metric_key_prefix=metric_key_prefix, prediction_loss_only=prediction_loss_only, test_count=test_count)

    if self.rank() == 0 and test_count is None:
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
          predict_results.predictions, predict_results.label_ids_text, contexts, models, tokenizers, seperator='<SEP>')
      
      self.tracker.add_metrics(ksm_scores, metric_key_prefix, 'ksm')

      print(f"Writing samples to output: {self.cfg.sample_dirs[self.stage]}")
      for pidx, (pred, context) in enumerate(zip(predictions, contexts)):
        if context != 'data':
          output_prediction_file = os.path.join(self.cfg.sample_dirs[self.stage], "generated_predictions_e{}_{}.txt".format(epoch, pidx))

          text = "OUTPUT: {} \n\n INPUT: {}".format(pred, context)
          with open(output_prediction_file, "w") as writer:
            writer.write(text)

    metrics = predict_results.metrics
    self.tracker.add_metrics(metrics, metric_key_prefix, 'rouge')
    self.logger.info("E{:02d} (eval) {} {}".format(self.epoch, 
    self.tracker.loss_str('epoch'), 
    self.tracker.metric_str('epoch', stage=self.stage)))
    outputs = {'score': metrics['loss']}
    return outputs

  