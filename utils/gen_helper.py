# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import os
import pickle
import numpy as np
import requests
import zipfile
import random
import torch
from easydict import EasyDict as edict
import yaml

def start_debug_mode(cfg):
  print(">>>> Debug mode activated.")
  os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
  os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
  os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
  cfg.timeout = 600
  cfg.debug = True
  cfg.num_workers = 0
  return cfg

def set_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def unzip_file(zip_path, extract_path):
  try:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_path)
        print("Extracted zip file to: {}".format(extract_path))
  except:
    raise FileExistsError("Invalid file: {}".format(zip_path))

def download_file(url, save_path):
  if not os.path.exists(save_path):
    print("Downloading from {}\nto {}".format(url, save_path))
    r = requests.get(url, stream = True)
    with open(save_path, "wb") as f:
      for chunk in r.iter_content(chunk_size = 1024):
        if chunk:
          f.write(chunk)


def create_eval_dir(base_dir, tag, epoch=0):
  epoch_pred_dir = os.path.join(base_dir, '{:02d}_{}'.format(epoch, tag))
  if not os.path.exists(epoch_pred_dir):
    try:
        os.mkdir(epoch_pred_dir)
    except FileExistsError:
        return epoch_pred_dir
  return epoch_pred_dir

def pickle_open(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data

def pickle_save(item, fname):
  with open(fname, "wb") as f:
    pickle.dump(item, f)

def mkdir_safe(path):
  if not os.path.exists(path):
    try:
      os.mkdir(path)
    except:
      pass

def load_cfg(cfg_name, cfg_dir, default_name='default.yaml'):
  default_path = os.path.join(cfg_dir, default_name)
  cfg_path = os.path.join(cfg_dir, cfg_name)
  assert os.path.exists(default_path) and os.path.exists(cfg_path), f"{default_path}, {cfg_path}"
  return merge_a_into_b(cfg_path, default_path)

def merge_a_into_b(cfg_path, default_path):
  default = edict(yaml.load(open(default_path, 'r'), Loader=yaml.FullLoader))
  cfg = edict(yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader))
  return _merge_a_into_b(cfg, default)
  
def _merge_a_into_b(a, b):

  assert type(a) is edict and type(b) is edict

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v 
  
  return b

def setup_gpu_cfg(cfg):
  n_gpus = torch.cuda.device_count()
  dist_avail = torch.distributed.is_available()
  if not dist_avail:
    raise SystemError("Torch Distributed Package Unavailable")

  if n_gpus == 0 or not cfg.use_gpu:
    cfg.use_gpu = False
    cfg.torch_dist.use = False
    cfg.rank = 'cpu'
    cfg.device_id = 'cpu'

  elif cfg.torch_dist.use:
    cfg.rank = int(os.environ["RANK"])
    cfg.device_id = int(os.environ["LOCAL_RANK"])
    cfg.device_ids = [cfg.device_id]

    if cfg.torch_dist.gpus_per_model <= 2:
      cfg.device_ids.append(cfg.device_id + 1)
    elif cfg.torch_dist.gpus_per_model > 2:
      raise NotImplementedError()

    torch.cuda.set_device(cfg.device_id)
  else:
    cfg.rank = 0
    cfg.device_id = 0
    torch.cuda.set_device(cfg.device_id)
  return cfg