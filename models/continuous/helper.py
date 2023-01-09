# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------


import torch

from ..constant import CHART_TO_HEAD_MAP, UNIQ_CHART_HEADS

def pad_vector(x, pad_len, pad_dim, device, dtype):
  x_pad_len = pad_len - x.size(pad_dim)
  if x_pad_len > 0:
    x_shape = list(x.shape)
    x_shape[pad_dim] = x_pad_len
    padding = torch.zeros(x_shape, device=device, dtype=dtype)
    x = torch.cat([x, padding], dim=pad_dim)
  return x

def unflat_tensor(x, hypo_count, hypo_dim=1):
  x_shape = list(x.shape)
  x_shape.insert(hypo_dim, hypo_count)
  x_shape[0] = -1

  return x.reshape(x_shape)

def get_topk_batch(topk_idx, x):
  x_batch = []
  for idx, _x  in zip(topk_idx, x):
    x_batch.append(_x[idx])
  return torch.stack(x_batch, dim=0)

def get_chart_type_dict(chart_types, hypo_bsz=1):
  chart_type_dict = {}
  
  for idx, ct in enumerate(chart_types): 
    hidx = idx * hypo_bsz

    head_map = CHART_TO_HEAD_MAP[ct]

    if head_map not in chart_type_dict:
      chart_type_dict[head_map] = []
    chart_type_dict[head_map] += [i for i in range(hidx, hidx + hypo_bsz)]
    
  return chart_type_dict

def get_chart_type_from_idx(indices):
  chart_type_dict = {}
  for i, idx in enumerate(indices):
    head_map = UNIQ_CHART_HEADS[idx]
    if head_map not in chart_type_dict:
      chart_type_dict[head_map] = []
    chart_type_dict[head_map] += [i]
  return chart_type_dict

def make_repeat_frame(x, hypo_bsz, only_frame=True, hypo_dim=1):
  x_shape = list(x.shape)
  repeat_frame = [1] * (len(x_shape) + 1)
  repeat_frame[hypo_dim] = hypo_bsz
  if only_frame:
    return repeat_frame
  else:
    x_repeated = x.unsqueeze(hypo_dim).repeat(repeat_frame)
    return repeat_frame, x_repeated

def get_win_hypo(hypos, wta_idx):
    batch_list = torch.arange(hypos.size(0))
    winner_hypo = hypos[batch_list, wta_idx, :]
    return winner_hypo