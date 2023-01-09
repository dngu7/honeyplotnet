# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------
from filelock import FileLock
import numpy as np
import nltk  
import os
import torch
import torch.distributed as dist
from torch.nn import functional as F
from evaluate import load as load_metric

from .base import BaseRunner

from transformers.trainer_pt_utils import (
  find_batch_size, 
  nested_concat,
  nested_numpify,
  nested_truncate,
)

from transformers.trainer_utils import ( 
  EvalPrediction, denumpify_detensorize
)


from utils.ksm_scores import (
  yake_text, embed_and_encode, calc_similarity
)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


from typing import Dict, NamedTuple, Optional, Tuple, Union

from utils.constant import TASK2PREPEND

class EvalLoopOutputwInputs(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    inputs: Optional[np.ndarray]

class CaptionRunner(BaseRunner):
  def __init__(self, stage, cfg):
    super(CaptionRunner, self).__init__(cfg)
    self.stage = stage

    self.ignore_pad_token_for_loss = self.cfg.model.caption.hf_model.ignore_pad_token_for_loss
    self.include_inputs_for_metrics = self.cfg.eval.include_inputs_for_metrics
    self.eval_accumulation_steps = self.cfg.eval.eval_accumulation_steps

    self._max_length = cfg.eval.max_length
    self._num_beams = cfg.eval.num_beams
    self._repetition_penalty = cfg.eval.repetition_penalty
    self._gen_temperature = cfg.eval.gen_temperature

    self.rouge = load_metric("rouge")

    self.loss_fn =torch.nn.CrossEntropyLoss(ignore_index=-100)

  def postprocess_text(self, preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

  def compute_metrics(self, eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if self.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

    result = self.rouge.compute(
      predictions=decoded_preds, 
      references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

  def compute_loss(self, model, inputs, return_outputs=False):

    if "labels" in inputs and self.cfg.train.label_smoothing_factor != 0:
        labels = inputs.pop("labels")
    else:
        labels = None

    outputs = model(**inputs)

    if labels is not None:
        flat_logits = torch.flatten(outputs.logits, start_dim=0, end_dim=1)
        labels = labels.view(-1)
        loss = self.loss_fn(flat_logits, labels)
    else:
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    return (loss, outputs) if return_outputs else loss

  def training_step(self, model, inputs):
    model.train()
    inputs = self._prepare_inputs(inputs)

    #print("training step: {}".format(inputs['labels']))

    with self.autocast_smart_context_manager():
      loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

    if self.use_torch_dist:
      loss = loss.mean() 
    
    if self.gradient_accum_steps > 1 and not self.use_deepspeed:
      loss = loss / self.gradient_accum_steps
    
    if not torch.isnan(loss).any():
      if self.do_grad_scaling:
        self.scaler.scale(loss).backward()
      elif self.use_deepspeed:
        model.backward(loss)
      else:
        loss.backward()
    else:
      print("NaN found in loss")
    

    return loss.detach()

  def train(self, train_loader, models, tokenizers, opts, schs):

    self.tracker.reset_all()

    tr_loss = torch.tensor(0.0).to(self.device_id)
    
    models[self.stage].zero_grad()
    opts[self.stage].zero_grad()
    iterator = train_loader.__iter__()

    steps_in_epoch = len(iterator)

    for step, model_inputs in enumerate(iterator):
      
      tr_loss_step = self.training_step(models[self.stage], model_inputs)
      
      tr_loss += tr_loss_step

      # Optimizer step for deepspeed called each step
      if self.use_deepspeed:
        models[self.stage].step()

      if (step + 1) % self.gradient_accum_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accum_steps
                    steps_in_epoch <= self.gradient_accum_steps
                    and (step + 1) == steps_in_epoch
                ):
        
        if not self.use_deepspeed:
          if self.do_grad_scaling:
            self.scaler.unscale_(opts[self.stage])
          if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(models[self.stage].parameters(), self.max_grad_norm)

        if self.use_deepspeed:
          pass
        elif self.do_grad_scaling:
          self.scaler.step(opts[self.stage])
          self.scaler.update()
        else:
          opts[self.stage].step()

        # if optimizer_was_run and not self.use_deepspeed:
        #   self.lr_scheduler.step()

        models[self.stage].zero_grad()
        opts[self.stage].zero_grad()

        self.global_step += 1
          
        self.tracker.add_logs(split='train', total_loss=tr_loss)
        tr_loss = 0

      if isinstance(self.display, int) and step % self.display == 0 and step > 0:
        self.logger.info("E{:02d} GS: {:03d} {}".format(
          self.epoch, self.global_step, self.tracker.loss_str('iter')))
        
        self.tracker.reset_interval('iter')

    self.logger.info("E{:02d} (train) {} {}".format(self.epoch, 
      self.tracker.loss_str('epoch'), 
      self.tracker.metric_str('epoch', stage=self.stage)))

    self.update_writer('train')

  def eval(self, eval_loader, models, tokenizers, metric_key_prefix='eval', prediction_loss_only=False, epoch=0, **kwargs):
    
    predict_results, all_inputs = self.eval_loop(self.stage, eval_loader, models, tokenizers, metric_key_prefix=metric_key_prefix, prediction_loss_only=prediction_loss_only)
    

    if self.rank() == 0:
      
      ###### Create readable files below
      predictions = tokenizers[self.stage].batch_decode(
          predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
      )

      if all_inputs is not None:
        contexts = tokenizers[self.stage].batch_decode(
            all_inputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
      else:
        contexts = [''] * len(predictions)

      #### Do KSM here
      ksm_scores = self.compute_ksm_scores(predict_results.predictions, predict_results.label_ids, contexts, models, tokenizers, seperator='<SEP>')
      self.tracker.add_metrics(ksm_scores, metric_key_prefix, 'ksm')

      for pidx, (pred, context) in enumerate(zip(predictions, contexts)):
        output_prediction_file = os.path.join(self.cfg.sample_dirs[self.stage], "generated_predictions_e{}_{}.txt".format(epoch, pidx))

        text = "OUTPUT: {} \n\n INPUT: {}".format(pred, context)
        with open(output_prediction_file, "w") as writer:
          writer.write(text)

      metrics = predict_results.metrics
      self.tracker.add_metrics(metrics, metric_key_prefix, 'rouge')

      # self.logger.info("E{:02d} (eval) {} {}".format(self.epoch, 
      #   self.tracker.metric_str('epoch', metric_key_prefix)))

      self.logger.info("E{:02d} (eval) {} {}".format(self.epoch, 
      self.tracker.loss_str('epoch'), 
      self.tracker.metric_str('epoch', stage=self.stage)))
          
    if self.use_torch_dist:
      dist.barrier()

    return predict_results

  def compute_ksm_scores(self, pred_tokens, labels, contexts, models, tokenizers, seperator='<SEP>'):
    
    sim_scores = {}
    for pidx, (pred_toks, label, context) in enumerate(zip(pred_tokens, labels, contexts)):

      if self.stage == 'chart_text':
        task = None
        for t, prepend_str in TASK2PREPEND.items():
          if context.startswith(prepend_str):
            task = t
            break
        
        if task is None:
          raise ValueError(f"No task found in context:  {context[:100]}")
      elif self.stage == 'caption':
        task = 'caption'
      else:
        raise ValueError(f"Stage not recognised for ksm: {self.stage}")

      #Create key words for context and task

      if task in ['chart_text', 'series_name', 'axis','categorical']:
        task_str = tokenizers[self.stage].decode(
            pred_toks, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        task_str = task_str.replace('<pad>','').replace('</s>','')
        task_keywords = [t.strip() for t in task_str.split(seperator)]

        #Replace -100 with 0
        label[label==-100] = 0
        ref_str = tokenizers[self.stage].decode(
            label, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        ref_str = ref_str.replace('<pad>','').replace('</s>','')
        reference_keywords = [t.strip() for t in ref_str.split(seperator)]

      elif task in ['caption']:
        decoded_str = tokenizers[self.stage].decode(
            pred_toks, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        task_keywords = [kw[0] for kw in yake_text(decoded_str)]
        reference_keywords = [kw[0] for kw in yake_text(context)]

      #Tokenize reference and task
      reference_tok = tokenizers['ksm'](reference_keywords, max_length=128, 
        padding="max_length", truncation=True, return_tensors="pt")
      task_tok = tokenizers['ksm'](task_keywords, max_length=128, 
        padding="max_length", truncation=True, return_tensors="pt")

      reference_emb = embed_and_encode(
        reference_tok, models['ksm'], device=self.device)
      task_emb = embed_and_encode(
        task_tok, models['ksm'], device=self.device)
      
      #Average embeddings through the sequence
      reference_emb = reference_emb.mean(1)
      task_emb = task_emb.mean(1)
      
      sim_score = calc_similarity(reference_emb, task_emb).mean().detach().cpu().item()
      
      if task not in sim_scores:
        sim_scores['ksm_' + task] = []

      sim_scores['ksm_' + task].append(sim_score)
    
    return sim_scores
  
  def eval_loop(self, cur_stage, loader, models, tokenizers, metric_key_prefix='eval', prediction_loss_only=False):

    self.tracker.reset_all()
    iterator = loader.__iter__()
    steps_in_epoch = len(iterator)

    models[cur_stage].eval()

    # Initialize containers
    # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
    losses_host = None
    preds_host = None
    labels_host = None
    inputs_host = None

    # losses/preds/labels on CPU (final containers)
    all_losses = None
    all_preds = None
    all_labels = None
    all_inputs = None
    observed_num_examples = 0
    batch_size = self.bsz

    for step, inputs in enumerate(iterator):
      
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
              
      loss, logits, labels = self.prediction_step(
        models[cur_stage], tokenizer=tokenizers[cur_stage], 
        inputs=inputs, prediction_loss_only=prediction_loss_only)

      inputs_decode = inputs["input_ids"] 

      # Update containers on host
      if loss is not None:
        losses = self._nested_gather(loss.repeat(batch_size))
        losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)

      if labels is not None:
        labels = self._pad_across_processes(labels)
        labels = self._nested_gather(labels)
        labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

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
        #if self.preprocess_logits_for_metrics is not None:
            #logits = self.preprocess_logits_for_metrics(logits, labels)
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
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (
                labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
            )

        # Set back to None to begin a new accumulation
        losses_host, preds_host, inputs_host, labels_host = None, None, None, None


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
    if labels_host is not None:
        labels = nested_numpify(labels_host)
        all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

    num_samples = steps_in_epoch
    # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
    # samplers has been rounded to a multiple of batch_size, so we truncate.
    if all_losses is not None:
        all_losses = all_losses[:num_samples]
    if all_preds is not None:
        all_preds = nested_truncate(all_preds, num_samples)
    if all_labels is not None:
        all_labels = nested_truncate(all_labels, num_samples)
    if all_inputs is not None:
        all_inputs = nested_truncate(all_inputs, num_samples)
  
    # Metrics!
    if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
      if self.include_inputs_for_metrics:
          metrics = self.compute_metrics(
              EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs),
              tokenizers[cur_stage]
          )
      else:
          metrics = self.compute_metrics(
            EvalPrediction(predictions=all_preds, label_ids=all_labels),
            tokenizers[cur_stage])
    else:
        metrics = {}
    
    metrics = denumpify_detensorize(metrics)

    if all_losses is not None:
        metrics["loss"] = all_losses.mean().item()

    
    return EvalLoopOutputwInputs(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples, inputs=all_inputs), all_inputs

  def prediction_step(self, model, tokenizer, inputs, prediction_loss_only=False, ignore_keys=[]):
    has_labels = 'labels' in inputs
    model.eval()
    
    inputs = self._prepare_inputs(inputs)

    gen_kwargs = {
            "max_length": self._max_length,
            "num_beams": self._num_beams,
            "synced_gpus": False,
            "repetition_penalty": self._repetition_penalty,
            "temperature": self._gen_temperature
        }

    if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
    if "global_attention_mask" in inputs:
        gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)
    
    # prepare generation inputs
    generation_inputs = inputs["input_ids"]

    if model.__class__.__name__ == 'DistributedDataParallel':
      generated_tokens = model.module.generate(
          generation_inputs,
          **gen_kwargs,
      )      
    else:
      generated_tokens = model.generate(
          generation_inputs,
          **gen_kwargs,
      )
    # in case the batch is shorter than max length, the output should be padded
    if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
        generated_tokens = self._pad_tensors_to_max_len(
          generated_tokens, gen_kwargs["max_length"], model, tokenizer)

    loss = None
    if has_labels:
      with torch.no_grad():
        with self.autocast_smart_context_manager():
          outputs = model(**inputs)
        
        if self.loss_fn is not None:
          logits = outputs.logits

          if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).to(self.device)
          
          logits = torch.flatten(logits, start_dim=0, end_dim=1)
          labels = torch.flatten(inputs["labels"], start_dim=0, end_dim=1)

          loss = self.loss_fn(logits, labels).mean().detach()
        else:
            loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
          
    if prediction_loss_only:
        return (loss, None, None)

    if has_labels:
        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"], model, tokenizer)
    else:
        labels = None

    return (loss, generated_tokens, labels)
    

  def _pad_tensors_to_max_len(self, tensor, max_length, model, tokenizer):
    if tokenizer is not None and hasattr(tokenizer, "pad_token_id"):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
    else:
        if model.config.pad_token_id is not None:
            pad_token_id = model.config.pad_token_id
        else:
            raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

    padded_tensor = pad_token_id * torch.ones(
        (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, : tensor.shape[-1]] = tensor
    return padded_tensor
      