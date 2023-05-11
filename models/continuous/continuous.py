# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------


import torch
from .scale import ScaleModel

class ContinuousModel(ScaleModel):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def embed_continuous(self, cont_logits, tab_shape, tab_samples):
    '''
    Converts ouput of decoders into input embeds that can be used by GANs
    '''
    
    row_sample = tab_samples['row']
    col_sample = tab_samples['col']

    row_counts = tab_shape['row']
    col_counts = tab_shape['col']

    max_rows = min(max(row_counts), max([l.size(0) for l in cont_logits]))
    max_cols = max(col_counts)

    cont_embds = []
    cont_masks = []
    for bidx, logits in enumerate(cont_logits):
      row_preds = row_sample[bidx].unsqueeze(-1).unsqueeze(-1)
      col_preds = col_sample[bidx].unsqueeze(0).unsqueeze(-1)

      embed = logits * row_preds[:logits.size(0), :, :]
      embed = embed * col_preds[:, :logits.size(1), :]
      embed = embed[:max_rows,:max_cols, :]

      rows = min(row_counts[bidx], max_rows)
      cols = col_counts[bidx]
      mask = [[1] * cols + [0] * (max_cols - cols)] * rows + [[0] * max_cols] * (max_rows - rows)
      mask = torch.tensor(mask, device=self.device)
      cont_embds.append(embed)
      cont_masks.append(mask)
    
    return cont_embds, cont_masks

  def unscale_continuous(self, cont_preds, scale_preds, tab_shape=None):
    
    # (bsz)
    if tab_shape is not None:
      row_counts = tab_shape['row']
      col_counts = tab_shape['col']
    
    cont_values = []
    for bidx, (all_cont_pred) in enumerate(cont_preds):

      #Can only generate as far as we have predictions available.
      if tab_shape is not None:
        rows = min(row_counts[bidx], all_cont_pred.size(0))
        cols = min(col_counts[bidx], all_cont_pred.size(1))
      else:
        rows = all_cont_pred.size(0)
        cols = all_cont_pred.size(1)

      #cut continuous shape to match tab shape
      cont_pred = all_cont_pred[:rows, :cols, :]
  
      #rows, (min, range) for each dimension
      scale_pred = scale_preds[bidx]
      if len(scale_pred.shape) == 1:
        scale_pred = scale_pred.view(1, -1)

      cont_dims = cont_pred.size(-1)
      
      #Reverse offseting. Increment from first point
      if cont_dims * 2 == scale_pred.size(-1):

        for dim in range(cont_dims):
          mindim = dim * 2
          rngdim = mindim + 1
          
          min_scale = scale_pred[:rows, mindim]
          rng_scale = scale_pred[:rows, rngdim]


          #scale_min_rng = (bsz, 2)
          min_scale = min_scale[:,None]
          rng_scale = rng_scale[:,None]

          #Perform scaling
          cont_pred[:,:,dim] = cont_pred[:,:,dim] * rng_scale + min_scale
      else:

        for cidx in range(1, cols):
          cont_pred[:, cidx, :] += cont_pred[:, cidx - 1, :]

        min_scale = scale_pred[:rows, 0]
        rng_scale = scale_pred[:rows, 1]

        #scale_min_rng = (bsz, 2)
        min_scale = min_scale.view(-1,1,1)
        rng_scale = rng_scale.view(-1,1,1)

        cont_pred = cont_pred * rng_scale + min_scale

      #print("cont_pred", bidx, cont_pred.shape)
      cont_values.append(cont_pred)

    return cont_values

