# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import os
import json
import sys
import torch
import contextlib
from packaging import version
from numpy import ceil
from collections.abc import Mapping

import torch
import torch.distributed as dist
from transformers import get_scheduler

from utils import (
  Logger, Writer, ResultTracker,
)

from transformers.trainer_pt_utils import (
   distributed_concat
)

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

class BaseRunner(object):
  def __init__(self, cfg):
    self.epoch = 0
    self.global_step = 1
    self.work_env = cfg.work_env

    self.cfg = cfg
    self.debug = cfg.debug
    self.device_id = self.local_rank()
    self.device = f'cuda:{self.local_rank()}'

    self.use_fid = cfg.eval.fid
    self.fid_stats = None

    self.use_torch_dist = cfg.torch_dist.use
    self.display = cfg.train.intervals.display
    self.bsz = self.cfg.batch_size

    self.logger = Logger(self.rank(), cfg.save_dir) 
    self.writer = Writer(self.rank(), cfg.save_dir) 
    
    self.logger.info("Runner Initialized - Rank=[{}/{}]".format(self.local_rank(), self.rank()))

    self.metrics = []
    self.metric_names = ['scale', 'continuous', 'categorical','series_name', 'cb1', 'cb2', 'wta','ct','row','col'] #TODO
    self.print_names  = ['scale', 'continuous', 'categorical','series_name', 'cb1', 'cb2', 'wta','ct','row','col'] #TODO
    
    self.tracker = ResultTracker(['epoch', 'iter'], print_names=self.print_names)
  
    self.lr_scheduler = None
    self.scaler = None
    self.gradient_accum_steps = cfg.train.gradient_accum_steps
    self.max_grad_norm = cfg.train.max_grad_norm
    self.use_deepspeed = self.cfg.torch_dist.deepspeed

    self.use_amp = False
    self.do_grad_scaling = False

    if self.cfg.fp16.use and not self.use_deepspeed: 
      self.use_amp = True
      self.amp_dtype = torch.float16 if self.cfg.fp16.use else torch.bfloat16
      self.do_grad_scaling = True
      self.scaler = torch.cuda.amp.GradScaler()
      
  def local_rank(self):
    r = os.environ.get("LOCAL_RANK")
    r = 0 if r is None else int(r)
    return r

  def rank(self):
    r = os.environ.get("RANK")
    r = 0 if r is None else int(r)
    return r

  def update_writer(self, split, interval='epoch'):
    for n in sorted(list(set(self.tracker.metric_names))):
      if self.tracker.get_loss(interval, n):
          self.writer.add_scalar(n, self.tracker.get_loss(interval, n), self.epoch)
  
  def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    passed as an argument.
    Args:
        num_training_steps (int): The number of training steps to do.
    """
    
    if self.lr_scheduler is None:
        self.lr_scheduler = get_scheduler(
            'linear', #self.cfg.train.scheduler.type,
            optimizer=optimizer,
            num_warmup_steps=int(ceil(num_training_steps * self.cfg.train.scheduler.warmup_ratio)),
            num_training_steps=num_training_steps,
        )

    return self.lr_scheduler

  def autocast_smart_context_manager(self):
    """
    A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
    arguments, depending on the situation.
    """
    if self.use_amp:
        if version.parse(torch.__version__) >= version.parse("1.10"):
            ctx_manager = autocast(dtype=self.amp_dtype)
        else:
            ctx_manager = autocast()
    else:
        ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

    return ctx_manager

  def _prepare_input(self, data):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: self._prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(self._prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=self.device_id)
        return data.to(**kwargs)
    return data

  def _prepare_inputs(self, inputs):
    """
    Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    inputs = self._prepare_input(inputs)
    if len(inputs) == 0:
        raise ValueError(
            "The batch received was empty, your model won't be able to train on it. Double-check that your "
            f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
        )

    return inputs

  def _nested_gather(self, tensors, name=None):
    """
    Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
    concatenating them to `gathered`
    """
    if tensors is None:
        return
    elif self.use_torch_dist:
      tensors = distributed_concat(tensors)
    return tensors

  # Copied from Accelerate.
  def _pad_across_processes(self, tensor, pad_index=-100):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
    they can safely be gathered.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(self._pad_across_processes(t, pad_index=pad_index) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: self._pad_across_processes(v, pad_index=pad_index) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
        )

    if len(tensor.shape) < 2:
        return tensor
    # Gather all sizes
    size = torch.tensor(tensor.shape, device=tensor.device)[None]
    sizes = self._nested_gather(size).cpu()

    max_size = max(s[1] for s in sizes)
    if tensor.shape[1] == max_size:
        return tensor

    # Then pad to the maximum size
    old_size = tensor.shape
    new_size = list(old_size)
    new_size[1] = max_size
    new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
    new_tensor[:, : old_size[1]] = tensor
    return new_tensor
  
  def to_json(self, samples, prefix, step=0, epoch=0):
    # Save list of dicts
    assert isinstance(samples, list), "must be list"
    if len(samples):
      assert isinstance(samples[0], dict), "must be list of dicts"

      epoch_dir = os.path.join(self.cfg.sample_dirs[self.stage], "{}".format(epoch))
      if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir, exist_ok=True)

      for idx, sample in enumerate(samples):
        filename = os.path.join(epoch_dir, "{}-{}-{}.json".format(prefix, step, idx))

        with open(filename, "w") as f:
          json.dump(sample, f)

  def to_vega_json(self, samples, prefix, step=0, epoch=0):
    # Save list of dicts
    assert isinstance(samples, list), "must be list"
    if len(samples):
      assert isinstance(samples[0], dict), "must be list of dicts"

      epoch_dir = os.path.join(self.cfg.sample_dirs[self.stage], "{}".format(epoch))
      if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir, exist_ok=True)

      for idx, sample in enumerate(samples):
        chart_type = sample['chart_type']
        assert chart_type in ['point','categorical','boxplot'], chart_type


        
        if chart_type == 'categorical':
          json_file = self.build_categorical_json(sample)
        elif chart_type == 'point':
          json_file = self.build_point_json(sample)
        else:
          continue

        filename = os.path.join(epoch_dir, f"{step}-{idx}-{chart_type}.json")

        with open(filename, 'w') as f:
          json.dump(json_file, f)


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
    chart_text_text = chart_data.get('chart_text')
    if chart_text_text is None:
      chart_text_text = [str(i) for i in range(10000)]

    json_file = get_vega_template(chart_type)

    data = []
    values = []
    d = {"name": "table"}

    cols = min(chart_data['col'], len(chart_text_text), len(continuous_data[0]))
    rows = min(chart_data['row'], len(continuous_data))
    
    text_idx = 0
    for cidx, row_idx in enumerate(range(rows)): #By series name
      for col_idx in range(cols): #Right to left
        v = {}
        v['x'] = chart_text_text[col_idx]
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
    