# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
# Forked from:
# https://github.com/pytorch/elastic/blob/master/examples/imagenet/main.py
# ---------------------------------------------------------------

import os
import glob
import random

import torch
import numpy as np


class State:
  """
  Container for objects that we want to checkpoint. Represents the
  current "state" of the worker. This object is mutable.
  """

  def __init__(self, models, tokenizers, opts, schs, rank=0, mode='train', stage='caption'):
    self.epoch = -1
    self.global_step = 0
    self.best_score = {'chart_text': 0.0, 'continuous': float('inf'), 'seq':float('inf')}
    self.models = models
    self.tokenizers = tokenizers
    self.opts = opts
    self.schs = schs
    self.scaler = {}
    self.metrics = []
    self.snapshot_keys = ['chart_text', 'continuous', 'seq_base', 'seq_cond']
    self.mode = mode
    self.cur_stage = stage
    self.rank = rank

  def _capture_snapshot(self, stage):
    snap = {}
    if self.models.get(stage) is not None:
      snap['{}_model'.format(stage)] = self.models[stage].state_dict()

    if self.opts.get(stage) is not None:
      snap['{}_opt'.format(stage)] = self.opts[stage].state_dict()

    if self.schs.get(stage) is not None:
      snap['{}_schs'.format(stage)] = self.schs[stage].state_dict()
    
    if self.scaler.get(stage) is not None:
      snap['{}_scaler'.format(stage)] = self.scaler[stage].state_dict()
    return snap

  def capture_snapshot(self, stage=None):
    if stage is None:
      stages = self.snapshot_keys
    else:
      stages = [stage]

    snapshot = {
      'epoch': self.epoch,
      'global_step': self.global_step,
      'best_score': self.best_score, 
      'metrics': self.metrics,
      'rng': self.get_rng()
      }

    for stage in stages:
      snap = self._capture_snapshot(stage)
      snapshot = {**snapshot, **snap}

    return snapshot

  def _apply_snapshot(self, obj, stage):

    obj_name = '{}_model'.format(stage)
    if obj.get(obj_name) is not None and self.models.get(stage) is not None:
      ddp_ckpt = 'module.' in list(obj[obj_name].keys())[0]
      ddp_model = self.models[stage].__class__.__name__ == 'DistributedDataParallel'
      
      if ddp_ckpt and not ddp_model:
        obj[obj_name] = self.from_ddp_ckpt(obj[obj_name])
      elif not ddp_ckpt and ddp_model:
        obj[obj_name] = self.to_ddp_ckpt(obj[obj_name])

      self.models[stage].load_state_dict(obj[obj_name])

    obj_name = '{}_tokenizer'.format(stage)
    if obj.get(obj_name) is not None and self.tokenizers.get(stage) is not None:
      try:
        self.tokenizers[stage].load_state_dict(obj[obj_name])
      except:
        print("Failed loading tokenizer for {} ".format(stage))

    if stage == self.cur_stage:
      obj_name = '{}_opt'.format(stage)
      if obj.get(obj_name) is not None and self.opts.get(stage) is not None:
        try:
          self.opts[stage].load_state_dict(obj[obj_name])
        except:
          print("Failed loading opt for {} ".format(stage))

      obj_name = '{}_schs'.format(stage)
      if obj.get(obj_name) is not None and self.opts.get(stage) is not None:
        try:
          self.schs[stage].load_state_dict(obj[obj_name])
        except:
          print("Failed loading schs for {} ".format(stage))

      obj_name = '{}_scaler'.format(stage)
      if obj.get(obj_name) is not None and self.opts.get(stage) is not None:
        try:
          self.scaler[stage].load_state_dict(obj[obj_name])
        except:
          print("Failed loading schs for {} ".format(stage))

  def apply_snapshot(self, obj, stage=None):
    if stage is None:
      stages = self.snapshot_keys
    else:
      stages = [stage]

    for stage in stages:
      self._apply_snapshot(obj, stage)

  def get_rng(self):
    rng_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }

    if torch.cuda.is_available():
        rng_states["cuda"] = torch.cuda.random.get_rng_state('cuda').cpu()
    
    return rng_states
  
  def set_rng(self, rng):
    if rng is not None:
      random.setstate(rng["python"])
      np.random.set_state(rng["numpy"])
      try:
        torch.random.set_rng_state(rng["cpu"].cpu())
        if torch.cuda.is_available():
          torch.cuda.random.set_rng_state(rng["cuda"].cpu())
      except:
        print("Failed to set random seed for torch")
        pass

  def load(self, ckpt_dirs, device_id):
    device = 'cpu' if device_id == 'cpu' else f"cuda:{device_id}"
    
    for stage, ckpt_dir in ckpt_dirs.items():
      if self.models.get(stage) is None: continue

      client_sd = None
      ckpt_files = glob.glob(ckpt_dir + '/*')
      if self.rank == 0:
        print(f"Checkpoint Dir: Files=[{len(ckpt_files)}] Path={ckpt_dir}")

      if len(ckpt_files) > 0:
        zero2f32_ckpts = [f for f in ckpt_files if f.endswith('.bin')]
        torch_ckpts = sorted([f for f in ckpt_files if f.endswith('snapshot.pth')])

        if len(zero2f32_ckpts) > 0:
          f = zero2f32_ckpts[0]
          print("[stage={}] Loading zero2f32 ckpt: {}".format(stage, f))
          snapshot = torch.load(f, map_location=device)
          self.models[stage].load_state_dict(snapshot, strict=False)

        elif len(torch_ckpts) > 0:
          f = torch_ckpts[-1]

          print("[stage={}] Loading torch ckpt: {}".format(stage, f))
          client_sd = torch.load(f, map_location=device)
          self.apply_snapshot(client_sd, stage=stage)

        if client_sd is not None and client_sd.get('epoch') is not None and (client_sd.get('epoch') > self.epoch or stage == self.cur_stage):
          try:
            self.set_rng(client_sd.get('rng'))
            self.global_step = client_sd['global_step']
            self.epoch = client_sd['epoch']
            self.best_score = client_sd['best_score']
            self.metrics = client_sd['metrics']
          except:
            print("Failed to update client sd")
            pass
        else:
          print("client_sd is None! state has not been updated")
  

  def save(self, ckpt_dirs, tag=None):

    for stage, ckpt_dir in ckpt_dirs.items():
      if stage != self.cur_stage: continue

      if tag is not None:
        ckpt_fn = '{}_snapshot.pth'.format(tag)
      else:
        ckpt_fn = '{}_snapshot.pth'.format(self.global_step)

      ckpt_path = os.path.join(ckpt_dir, ckpt_fn)
      torch.save(self.capture_snapshot(stage), ckpt_path)

  def from_ddp_ckpt(self, ddp_snapshot):
    # Convert from ddp ckpt to nonddp model.
    snapshot = {}
    for k,v in ddp_snapshot.items():
      new_k = k.replace('module.','')
      snapshot[new_k] = v
    
    return snapshot
  
  def to_ddp_ckpt(self, snapshot):
    ddp_snapshot = {}
    for k,v in snapshot.items():
      new_k = 'module.' + k
      ddp_snapshot[new_k] = v
    return ddp_snapshot