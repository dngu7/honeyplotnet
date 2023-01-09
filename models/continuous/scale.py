# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F

from .base import BaseDataModel

class ScaleModel(BaseDataModel):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def unscale_scales(self, scale_logits, tab_shape):
    '''
    Returns to original values
    '''

    row_counts = tab_shape['row']

    scale_values = []
    for bidx, logits in enumerate(scale_logits):
      rows = int(row_counts[bidx])

      logits = logits[:rows, :]

      base = torch.ones_like(logits) * self.scale_exponent
      scale_pred = torch.pow(base, logits)

      dims = scale_pred.size(-1)
      mean_dims = [i for i in range(dims) if (i % 2) == 0]
      std_dims  = [i for i in range(dims) if (i % 2) == 1]

      scale_pred[:,mean_dims] -= self.scale_eps[0]
      scale_pred[:,std_dims]  -= self.scale_eps[1]
      
      scale_values.append(scale_pred)

    return scale_values
  
  def embed_scale(self, scale_logits, tab_shape, tab_samples):
    '''
    Converts ouput of decoders into input embeds that can be used by GANs
    '''

    row_sample = tab_samples['row']
    row_counts = tab_shape['row']
    max_rows = max(row_counts)

    scale_embds = []
    scale_masks = []
    for bidx, logits in enumerate(scale_logits):
      row_preds = row_sample[bidx].unsqueeze(-1)
      embed = logits * row_preds
      embed = embed[:max_rows,:]
      rows = row_counts[bidx]
      mask = [1] * rows + [0] * (max_rows - rows)
      mask = torch.tensor(mask, device=self.device)
      scale_embds.append(embed)
      scale_masks.append(mask)
    return scale_embds, scale_masks
  
