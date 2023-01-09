# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class Discriminator(nn.Module):
  def __init__(self, 
    enc_conv, enc_tf, disc_loss='hinge', emb_dim1=8,  use_pos_embs=True,
    max_series_blocks=5, max_cont_blocks=32, norm_mode='minmax',
    disc_factor=1.0, disc_weight=1.0, disc_conditional=False, disc_start=100,
    device='cpu', fp16=False, debug=False):
    super().__init__()
    assert disc_loss in ["hinge", "vanilla"]
    
    self.debug        = debug
    self.device       = device 
    self.use_fp16     = fp16
    self.dtype_float  = torch.float32 
    self.dtype_int    = torch.int32 

    self.norm_mode    = norm_mode

    self.discriminator = Encoder(
      enc_conv=enc_conv,
      enc_tf=enc_tf,
      out_dim=emb_dim1,
      max_series_blocks=max_series_blocks,
      max_cont_blocks=max_cont_blocks,
      use_pos_embs=use_pos_embs,
      norm_mode=norm_mode,
      chart_type_conditional=disc_conditional,
      device=device,
      debug=debug
    )

    self.discriminator_iter_start = disc_start

    if disc_loss == "hinge":
        self.disc_loss = hinge_d_loss
    elif disc_loss == "vanilla":
        self.disc_loss = vanilla_d_loss
    else:
        raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
    
    self.disc_factor = disc_factor
    self.discriminator_weight = disc_weight
    self.disc_conditional = disc_conditional    
      
  def forward(self, loss_dict, inputs, reconstructions, optimizer_idx, global_step):

    if optimizer_idx == 0:
      # generator update
      logits_fake, na_loss = self.discriminator(reconstructions)
      g_loss = -torch.mean(logits_fake)

      disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
      loss = disc_factor * g_loss + na_loss
      return {'gen': loss}, {}
    
    else:
      
      logits_real, na1_loss = self.discriminator(self.preprocess_for_disc(inputs))
      logits_fake, na2_loss = self.discriminator(self.preprocess_for_disc(reconstructions))

      disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
      #zero_loss = 0.0 * sum([v for v in loss_dict.values()])
      d_loss = disc_factor * self.disc_loss(logits_real, logits_fake) + na1_loss + na2_loss

      return {'disc': d_loss}, {}
    
  def preprocess_for_disc(self, x):
    output = {'scale': {}, 'continuous': {}}
    
    for k, v in x.items():

      if k in ['scale','continuous']:
        for kk, vv in v.items():
          if isinstance(vv, list):
            output[k][kk] = [vvv.contiguous().detach() for vvv in vv]
          elif isinstance(vv, torch.Tensor):
            output[k][kk] = vv.contiguous().detach()

      else:
        output[k] = v

    return output
