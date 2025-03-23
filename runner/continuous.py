# ---------------------------------------------------------------
# Copyright (c) Cybersecurity Cooperative Research Centre 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import os
import numpy as np

import torch
import torch.distributed as dist

from fid import calculate_frechet_distance

from utils import (
  pickle_save, create_recon_plots
)

from .base import BaseRunner

from models.constant import CHART_TO_HEAD_IDX



class ContinuousRunner(BaseRunner):
  def __init__(self, stage, cfg):
    super(ContinuousRunner, self).__init__(cfg)
    self.stage = stage
    self.loss_weights = cfg.train.loss_weights

  def training_step(self, models, inputs, opt_idx):

    opt_model = 'continuous' if opt_idx == 0 else 'disc'

    with self.autocast_smart_context_manager():
      x_hat, loss_dict, metric_log = models['continuous'](inputs)

      if 'disc' in models:
        disc_loss, disc_log = models['disc']( 
              loss_dict=loss_dict,
              inputs=inputs['chart_data'], 
              reconstructions=x_hat, 
              optimizer_idx=opt_idx, 
              global_step=self.global_step
              )
        loss_dict  = {**loss_dict, **disc_loss}
        metric_log = {**metric_log, **disc_log}

    loss_log = {}
    total_loss = 0.0
    for name, loss in loss_dict.items():
      l_name = name.split('/')[-1].replace('_loss','')

      weight = 1.0
      if hasattr(self.loss_weights, l_name):
        weight = getattr(self.loss_weights, l_name)

      total_loss += loss * weight
      loss_log[name] = loss.detach().cpu()
        

    if self.use_torch_dist:
      total_loss = total_loss.mean() 
    
    if self.gradient_accum_steps > 1:
      total_loss = total_loss / self.gradient_accum_steps
    
    if self.do_grad_scaling:
      self.scaler.scale(total_loss).backward()
    else:
      total_loss.backward()

    return loss_log, metric_log

  def train(self, train_loader, models, tokenizers, opts, schs):

    self.tracker.reset_all()
    tr_loss = torch.tensor(0.0).to(self.device_id)

    for m in models.values():
      m.train()
      m.zero_grad()
    for o in opts.values():
      if o is not None:
        o.zero_grad()

    iterator = train_loader.__iter__()
    steps_in_epoch = len(iterator)

    for step, (_, inputs) in enumerate(iterator):
      opt_count = 2 if 'disc' in models else 1
      for opt_idx in range(opt_count):
        opt_model = 'continuous' if opt_idx == 0 else 'disc'

        loss_log, metric_log = self.training_step(models, inputs, opt_idx=opt_idx)

        tr_loss_step = sum(list(loss_log.values()))
        tr_loss += tr_loss_step
        
        self.tracker.add_logs(split='train', log=loss_log, total_loss=tr_loss_step)
        self.tracker.add_metrics(split='train', metrics=metric_log, metric_name='continuous')

        if (step + 1) % self.gradient_accum_steps == 0 or (
                      # last step in epoch but step is always smaller than gradient_accum_steps
                      steps_in_epoch <= self.gradient_accum_steps
                      and (step + 1) == steps_in_epoch
                  ):
          
          if self.do_grad_scaling:
            self.scaler.unscale_(opts[opt_model])
          if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(models[opt_model].parameters(), self.max_grad_norm)

          if self.do_grad_scaling:
            self.scaler.step(opts[opt_model])
            self.scaler.update()
          else:
            opts[opt_model].step()

          models[opt_model].zero_grad()
          opts[opt_model].zero_grad()

          self.global_step += 1
          tr_loss = 0
      
      if isinstance(self.display, int) and step % self.display == 0 and step > 0:
        self.logger.info("E{:02d} GS: {:03d} {} {}".format(
          self.epoch, self.global_step, self.tracker.loss_str('iter'), 
          self.tracker.metric_str('iter')))
        
        self.tracker.reset_interval('iter')

    self.logger.info("E{:02d} (train) {} {}".format(self.epoch, 
    self.tracker.loss_str('epoch'), self.tracker.metric_str('epoch')))
    self.update_writer('train')

    if self.use_torch_dist:
      dist.barrier()

  def eval(self, val_loader, models, metric_key_prefix='eval', epoch=0, **kwargs):

    self.tracker.reset_all()
    iterator = val_loader.__iter__()
    steps_in_epoch = len(iterator)

    if self.use_torch_dist:
      dist.barrier()

    for m in models.values():
      if m is not None:
        m.eval()
    
    fid_log = {}
    if self.use_fid:
      fid_log['train_fid'], fid_log['test_fid'] = self.compute_fid(val_loader, models)
      self.tracker.add_metrics(split=metric_key_prefix, metrics=fid_log, metric_name='fid')

    if (epoch + 1) % self.cfg.eval.sample_epoch == 0:
      epoch_dir = os.path.join(self.cfg.sample_dirs[self.stage], "{}".format(epoch))
      if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir, exist_ok=True)

    tr_loss = 0.0
    for step, (indices, inputs) in enumerate(iterator):

      with torch.no_grad():
        with self.autocast_smart_context_manager():
          x_hat, loss_dict, metric_log = models['continuous'](inputs, is_train=False, split=metric_key_prefix)

          if 'disc' in models:
            disc_loss, disc_log = models['disc']( 
                  loss_dict=loss_dict,
                  inputs=inputs, 
                  reconstructions=x_hat, 
                  optimizer_idx=0, 
                  global_step=self.global_step
                  )
            loss_dict  = {**loss_dict, **disc_loss}
            metric_log = {**metric_log, **disc_log}

        loss_log = {}
        tr_loss_step = 0.0
        for name, loss in loss_dict.items():
          tr_loss_step += loss.detach().cpu()
          loss_log[name] = loss.detach().cpu()

        tr_loss += tr_loss_step

      self.tracker.add_logs(split=metric_key_prefix, log=loss_log, total_loss=tr_loss_step)
      self.tracker.add_metrics(split=metric_key_prefix, metrics=metric_log, metric_name='continuous')

      if (step + 1) % self.gradient_accum_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accum_steps
                    steps_in_epoch <= self.gradient_accum_steps
                    and (step + 1) == steps_in_epoch
                ):
        tr_loss = 0
      
      if (step + 1) % self.cfg.eval.sample_interval == 0 and  (epoch + 1) % self.cfg.eval.sample_epoch == 0:
        #self.to_vega_json(x_hat, prefix=metric_key_prefix, step=step, epoch=epoch)
        text_data = [val_loader.dataset.get_text_with_idx(ind) for ind in indices]
        create_recon_plots(inputs, x_hat, text_data, step, epoch_dir)
          

      if isinstance(self.display, int) and step % self.display == 0 and step > 0:
        self.logger.info("E{:02d} GS: {:04d} {} {}".format(
          self.epoch, self.global_step, self.tracker.loss_str('iter'),
          self.tracker.metric_str('iter')))
        
        self.tracker.reset_interval('iter')

    self.logger.info("E{:02d} (eval)  {} {}".format(
      self.epoch, self.tracker.loss_str('epoch'),
      self.tracker.metric_str('epoch')))

    self.update_writer('eval')

    if self.use_torch_dist:
      dist.barrier()
    
    #Calculate scores for saving
    if self.use_fid:
      score = fid_log['test_fid']
    else:
      score_names = ['metric/eval/row/total','metric/eval/col/total','metric/eval/scale/total','metric/eval/cont/total']
      score = sum(self.tracker.get_loss('epoch', s) for s in score_names)

    return {'score': score}

  def generate(self, val_loader, models, tokenizers, metric_key_prefix='generate', epoch=0):
  
    self.tracker.reset_all()
    iterator = val_loader.__iter__()
    steps_in_epoch = len(iterator)
    if self.use_torch_dist:
      dist.barrier()

    for m in models.values():
      if m is not None:
        m.eval()

    batch_size = self.bsz
    for step, (idx, inputs) in enumerate(iterator):

      if isinstance(self.display, int) and step % self.display == 0 and step > 0:
        self.logger.info("Eval | E{:02d} Step {:04d}/{:04d} ".format(self.epoch, step, self.cfg.eval.max_steps))

      if isinstance(self.cfg.eval.gen_steps, int) and step > self.cfg.eval.gen_steps:
        break
      
      #Sample the indices
      with torch.no_grad():
        with self.autocast_smart_context_manager():
          cb_ind1 = models['continuous'].sample_codebook(inputs)

      cb_ind2 = None
      if len(cb_ind1) == 2:
        cb_ind1, cb_ind2 = cb_ind1
      else:
        cb_ind1 = cb_ind1[0]

      ct_idx = [CHART_TO_HEAD_IDX[ct] for ct in inputs['chart_data']['chart_type']]
      ct_idx = torch.tensor(ct_idx, dtype=torch.long, device=self.device).view(-1,1)
    
      with torch.no_grad():
        with self.autocast_smart_context_manager():
          samples = models['continuous'].reconstruct_from_indices(
              ct_idx=ct_idx, 
              cb_ind1=cb_ind1, 
              cb_ind2=cb_ind2,
              hypo_count=self.cfg.eval.hypo_count, 
              hypo_bsz=self.cfg.eval.hypo_bsz
              )

    if self.use_torch_dist:
      dist.barrier()

    self.logger.info("Epoch {:02d} ({}) | {} ".format(epoch, metric_key_prefix, self.tracker.loss_str('epoch')))
  
  def save(self, loader, models, prefix):

    data_path = os.path.join(self.cfg.data_path, 'processed')
    save_dir = os.path.join(data_path, self.cfg.exp_name)

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{prefix}.pkl')
    self.logger.info(f"Saving data @ {file_path}")

    iterator = loader.__iter__()
    if self.use_torch_dist:
      dist.barrier()

    for m in models.values():
      if m is not None:
        m.eval()
    
    container = []
    for _, (idx, inputs) in enumerate(iterator):
      
      #Sample the indices
      with torch.no_grad():
        with self.autocast_smart_context_manager():
          cb_ind1 = models['continuous'].sample_codebook(inputs)

      cb_ind2 = None
      if len(cb_ind1) == 2:
        cb_ind1, cb_ind2 = cb_ind1
        cb_ind2 = cb_ind2.detach().cpu().tolist()
      else:
        cb_ind1 = cb_ind1[0]

      cb_ind1 = cb_ind1.detach().cpu().tolist()

      #Get data from dataset
      data = loader.dataset.get_data_with_idx(idx)

      for d, ind1, ind2 in zip(data, cb_ind1, cb_ind2):

        d['codebook'] = {}
        d['codebook'][0] = ind1
        d['codebook'][1] = ind2
        container.append(d)
      
    pickle_save(container, file_path)
      

  def compute_fid(self, loader, models):
    assert 'fid' in models and 'continuous' in models
    iterator = loader.__iter__()

    if self.use_torch_dist:
      dist.barrier()

    for m in models.values():
      if m is not None:
        m.eval()

    act_container = []
    
    for (_, inputs) in iterator:
      with torch.no_grad():
        with self.autocast_smart_context_manager():
          x_hat, _, _ = models['continuous'](inputs, is_train=False, temp=1.0)

          activations, _, _ = models['fid'](x_hat)
          act_container.append(activations)

    act = np.concatenate(act_container, axis=0)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    
    #Load existing fid scores
    m1, s1, _ = self.fid_stats['train']
    m2, s2, _ = self.fid_stats['test']

    train_fid = calculate_frechet_distance(mu, sigma, m1, s1)
    test_fid = calculate_frechet_distance(mu, sigma, m2, s2)

    return train_fid, test_fid
