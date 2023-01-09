# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import os
import numpy as np
import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.nn import functional as F

from utils import (
  load_statistics, calculate_frechet_distance, create_recon_plots, create_single_plot
)

from models.constant import CHART_TO_HEAD_IDX

from .base import BaseRunner

class SeqRunner(BaseRunner):
  def __init__(self, stage, cfg):
    super(SeqRunner, self).__init__(cfg)
    self.stage = stage

  def sample_indices(self, logits, temp=1.0):
    bsz = logits.size(0)
    probs = F.softmax(logits / temp, dim=-1).data
    probs = torch.flatten(probs, start_dim=0, end_dim=1)
    cb_indices =  torch.multinomial(probs, 1)
    cb_indices = cb_indices.reshape([bsz, -1])
    return cb_indices


  def eval_step(self, models, inputs):
    models['continuous'].eval()
    models['seq'].eval()

    with self.autocast_smart_context_manager():
      with torch.no_grad():
        if hasattr(models['continuous'], 'module'):
          cb1 = models['continuous'].module.sample_codebook(inputs)
        else:
          cb1 = models['continuous'].sample_codebook(inputs)

        if len(cb1) == 2:
          cb1, cb2 = cb1
        else:
          cb1, cb2 = cb1[0], None

        context_idx = inputs['captions']['input_ids']
        attn_mask = inputs['captions']['attention_mask']

        ct_idx = [CHART_TO_HEAD_IDX[ct] for ct in inputs['chart_data']['chart_type']]
        ct_idx = torch.tensor(ct_idx, dtype=torch.long, device=self.device).view(-1,1)

        _, _, _, loss_dict = models['seq'](
          context_idx, attn_mask=attn_mask, tgt1=cb1, tgt2=cb2, tgt3=ct_idx)

    loss_log = {}
    total_loss = 0.0
    for name, loss in loss_dict.items():
      total_loss += loss
      loss_log[name] = loss.detach().cpu()
    return loss_log

  def training_step(self, models, inputs):
    models['continuous'].eval()
    models['seq'].train()

    with self.autocast_smart_context_manager():
      with torch.no_grad():
        if hasattr(models['continuous'], 'module'):
          cb1 = models['continuous'].module.sample_codebook(inputs)
        else:
          cb1 = models['continuous'].sample_codebook(inputs)

      if len(cb1) == 2:
        cb1, cb2 = cb1
      else:
        cb1, cb2 = cb1[0], None

      context_idx = inputs['captions']['input_ids']
      attn_mask = inputs['captions']['attention_mask']

      ct_idx = [CHART_TO_HEAD_IDX[ct] for ct in inputs['chart_data']['chart_type']]
      ct_idx = torch.tensor(ct_idx, dtype=torch.long, device=self.device).view(-1,1)

      _, _, _, loss_dict = models['seq'](
        context_idx, attn_mask=attn_mask, tgt1=cb1, tgt2=cb2, tgt3=ct_idx)

    loss_log = {}
    total_loss = 0.0
    for name, loss in loss_dict.items():
      total_loss += loss
      loss_log[name] = loss.detach().cpu()

    if self.use_torch_dist:
      total_loss = total_loss.mean() 
    
    if self.gradient_accum_steps > 1 and not self.use_deepspeed:
      total_loss = total_loss / self.gradient_accum_steps
    
    if self.do_grad_scaling and not self.use_deepspeed:
      self.scaler.scale(total_loss).backward()
    elif self.use_deepspeed:

      models['seq'].backward(total_loss)
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

    if self.use_torch_dist:
      dist.barrier()

  def eval(self, val_loader, models, tokenizers, metric_key_prefix='eval', epoch=0, temp=1.0, create_sample=False):

    if self.use_torch_dist:
      dist.barrier()

    for m in models.values():
      if m is not None:
        m.eval()

    self.tracker.reset_all()
    tr_loss = torch.tensor(0.0).to(self.device_id)
    for _, (_, inputs) in enumerate(val_loader.__iter__()):
      loss_log = self.eval_step(models, inputs)
      tr_loss_step = sum(list(loss_log.values()))
      tr_loss += tr_loss_step
      self.tracker.add_logs(split='eval', log=loss_log, total_loss=tr_loss_step)

    self.logger.info("E{:02d} (eval) {}".format(self.epoch, self.tracker.loss_str('epoch')))
    self.update_writer('eval')

    epoch_dir = os.path.join(self.cfg.sample_dirs[self.stage], "{}".format(epoch))

    if create_sample:
      act_container = []    
      for step, (indices, inputs) in enumerate(val_loader.__iter__()):
        with torch.no_grad():
          with self.autocast_smart_context_manager():

            chart_types = inputs['chart_data']['chart_type']
            context_idx = inputs['captions']['input_ids']
            attn_mask   = inputs['captions']['attention_mask']

            cb1_logits, cb2_logits, ct_idx, _ = models['seq'](
              context_idx, attn_mask=attn_mask)
          
            cb_ind1 = self.sample_indices(cb1_logits, temp=temp)
            cb_ind2 = self.sample_indices(cb2_logits, temp=temp)
            
            if ct_idx is not None:
              ct_idx = [CHART_TO_HEAD_IDX[ct] for ct in chart_types]
              ct_idx = torch.tensor(ct_idx, dtype=torch.long, device=self.device).view(-1,1)

            #print("cb_ind1", cb_ind1.shape, "cb_ind2", cb_ind2.shape)
            if hasattr(models['continuous'], 'module'):
              x_hat = models['continuous'].module.reconstruct_from_indices(
                ct_idx=ct_idx,
                cb_ind1=cb_ind1, 
                cb_ind2=cb_ind2, 
                temp=temp)
            else:
              x_hat = models['continuous'].reconstruct_from_indices(
                ct_idx=ct_idx,
                cb_ind1=cb_ind1, 
                cb_ind2=cb_ind2, 
                temp=temp)
            
            #Required for labels
            x_hat['chart_type'] = chart_types

        if create_sample or ((step + 1) % self.cfg.eval.sample_interval == 0):
          if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir, exist_ok=True)

          #print("create", step, epoch_dir)
          #self.to_vega_json(x_hat, prefix=metric_key_prefix, step=step, epoch=epoch)
          text_data = [val_loader.dataset.get_text_with_idx(ind) for ind in indices]
          #create_recon_plots(inputs, x_hat, text_data, step, epoch_dir)
          create_single_plot(x_hat, text_data, epoch_dir, step)

    #Calculate scores for saving
    score_names = ['loss/eval/total']
    score = sum(self.tracker.get_loss('epoch', s) for s in score_names)

    output = {}
    output['score'] = score
    return output

  def create_generator_seq(self, vae_model, seq_model, batch_size, num_total_samples, sample_count, seq_cond_model=None):

    num_iters = int(np.ceil(num_total_samples / batch_size))
    total_samples = 0
    temp = 1.0
    ind_2 = None

    if self.use_torch_dist:
      vae_module = vae_model.module
    else:
      vae_module = vae_model
    
    p1 = self.seq_shape1
    p2 = self.seq_shape2
    if self.cfg.model.backbone == 'vq2':
      p1, p2 = p2, p1
      
    for _ in range(num_iters):
      if total_samples > num_total_samples:
          break
      
      ind_1 = self.sample_seq_model(seq_model, p1, temp=temp)

      if seq_cond_model is not None:
        ind_2 = self.sample_seq_model(seq_cond_model, 
          p2, temp=temp, condition=Variable(ind_1))

      with torch.no_grad():
        outputs = vae_module.reconstruct_from_indices(
          ind_1=ind_1, ind_2=ind_2, sample_count=sample_count)

      if 'xk_hat' not in outputs:
        out = outputs['xc_hat']
      else:
        out = torch.flatten(outputs['xk_hat'], start_dim=0, end_dim=1) 

      num_total_samples += out.size(0)
      yield out.float()


  def sample_seq_model(self, model, size, temp=1.0, condition=None):
    cache = {}
    ind = torch.zeros(size, dtype=torch.long)
    ind = ind.cuda(self.device_id)
    with torch.no_grad():
      for i in range(size[-1]):
        for j in range(size[-2]):
          out, cache = model(Variable(ind), condition=condition, cache=cache)
          probs = F.softmax(out[:, :, j, i] / temp, dim=1).data
          val =  torch.multinomial(probs, 1).view(-1)
          ind[:, j, i] = val

    return ind