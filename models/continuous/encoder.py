# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import random

import torch
import torch.nn as nn

from .helper import (
  get_chart_type_dict,
  pad_vector
)

from ..constant import SCALE_DIMS, REG_DIMS, UNIQ_CHART_HEADS, CHART_TO_HEAD_IDX

from .coder import Coder

class Encoder(Coder):
  def __init__(self,
    enc_conv,
    enc_tf,
    out_dim, 
    max_series_blocks,
    max_cont_blocks,
    chart_type_conditional=False,
    use_pos_embs=True, 
    norm_mode='minmax',
    device='cuda:0', 
    debug=False,
    ):
    super().__init__()

    self.chart_type_conditional = chart_type_conditional
    self.max_cont_blocks = max_cont_blocks
    self.max_series_blocks = max_series_blocks
    self.out_dim = out_dim
    self.norm_mode = norm_mode
    self.use_pos_embs = use_pos_embs
    self.device = device
    self.debug = debug
    self.dtype_float = torch.float32

    self.enc_conv  = enc_conv
    self.enc_tf    = enc_tf

    self.cont_encoder = nn.ModuleDict()
    for name in ['scale', 'continuous']:
      self.cont_encoder[name] = nn.ModuleDict()

    for head_name, head_dim in SCALE_DIMS[self.norm_mode].items():
      self.cont_encoder['scale'][head_name] = nn.Linear(head_dim, self.out_dim)

    for head_name, head_dim in REG_DIMS.items():
      self.cont_encoder['continuous'][head_name] = nn.Linear(head_dim, self.out_dim)

    self.pos_emb_cont = None
    self.pos_emb_scale = None
    if self.use_pos_embs:
      self.pos_emb_cont = nn.Parameter(
        torch.zeros(1, max_series_blocks, max_cont_blocks, self.out_dim))

      self.pos_emb_scale = nn.Parameter(
        torch.zeros(1, max_series_blocks, self.out_dim))

    if self.chart_type_conditional:
      self.ct_emb = nn.Embedding(len(UNIQ_CHART_HEADS), self.out_dim)
    
    self.apply(self._init_weights)


  def forward(self,inputs):
        
    ### Shape into sequence
    encoder_input, _, loss = self.prepare_encoder_input(inputs)
    
    ### Input through transformer (t5 or gpt)
    if self.enc_tf is not None:
      encoder_output = self.enc_tf(encoder_input)

    ### Downsample encoder inputs
    encoder_output = self.enc_conv(encoder_output)

    return encoder_output, loss

  def get_chart_type_emb(self, ct_idx=None, chart_type=None, enc_ver=False): 
    if ct_idx is None and chart_type is None:
      raise ValueError("Need to provide either indinces or string")

    if chart_type is not None:
      ct_idx = [CHART_TO_HEAD_IDX[ct] for ct in chart_type] 
      ct_idx = torch.tensor(ct_idx, dtype=torch.long, device=self.device).view(-1,1)

    return self.ct_emb(ct_idx)
  

  def prepare_encoder_input(self, inputs):

    x = inputs['chart_data'] if 'chart_data' in inputs else inputs

    chart_type = x['chart_type']
    chart_type_dict = x.get('chart_type_dict')

    if chart_type_dict is None:
      chart_type_dict = get_chart_type_dict(chart_type)

    scale_embd, scale_mask, s_loss = self.preencode_scale(
      inputs_embeds=x['scale']['inputs_embeds'],
      attention_mask=x['scale']['attention_mask'],
      chart_type_dict=chart_type_dict
      )

    cont_embd, cont_mask, c_loss  = self.preencode_continuous(
      inputs_embeds=x['continuous']['inputs_embeds'],
      attention_mask=x['continuous']['attention_mask'],
      chart_type_dict=chart_type_dict,
      flatten_series=True
    )

    loss = s_loss + c_loss

    encoder_input = [scale_embd, cont_embd]
    encoder_mask  = [scale_mask, cont_mask]

    #Prepend chart type before encoding
    if self.chart_type_conditional:
      ct_vec = self.get_chart_type_emb(chart_type=chart_type, enc_ver=True)
      ct_mask = torch.ones_like(ct_vec).mean(-1) #.to(self.device)
      encoder_input = [ct_vec] + encoder_input
      encoder_mask = [ct_mask] + encoder_mask

    encoder_input = torch.cat(encoder_input, dim=1)
    encoder_mask  = torch.cat(encoder_mask, dim=1)

    #Pad or crop to fit inside sequence length
    bsz, seq_len, dim = encoder_input.shape
    if encoder_input.size(1) >= self.max_cont_blocks:
      encoder_input = encoder_input[:, :self.max_cont_blocks, :]
      encoder_mask  = encoder_mask[:, :self.max_cont_blocks]
    else:
      pad_len = self.max_cont_blocks - seq_len
      
      inp_pad  = torch.zeros((bsz, pad_len, dim), device=self.device, dtype=self.dtype_float)
      mask_pad = torch.zeros((bsz, pad_len), device=self.device, dtype=self.dtype_float)

      encoder_input = torch.cat([encoder_input, inp_pad], dim=1)
      encoder_mask  = torch.cat([encoder_mask, mask_pad], dim=1)

    return encoder_input, encoder_mask, loss

  def preencode_continuous(self, inputs_embeds, attention_mask, chart_type_dict, pad_len=None, pad_dim=1, flatten_series=True):
    if isinstance(attention_mask, list):
      attention_mask = torch.stack(attention_mask, dim=0)
    
    assert isinstance(attention_mask, torch.Tensor)
    cont_mask = attention_mask.to(self.device)
    bsz, row_len, col_len = cont_mask.shape

    #Declare continuous embeddings
    cont_embd = torch.zeros((bsz, row_len, col_len, self.out_dim), dtype=self.dtype_float, device=self.device)

    #Loop through each chart type dict containing information on what indices contains what chart type
    for head_name, ind in chart_type_dict.items():

      #Prepare inputs by obtaining rows pertaining to a particular chart type
      cont_x = torch.stack([l for idx, l in enumerate(inputs_embeds) if idx in ind], dim=0).to(self.device)
      mask = cont_mask[ind,:] 
        
      #Loop through each row and encode values
      for s_idx in range(row_len):

        data_x     = cont_x[:, s_idx, :]
        data_mask  = mask[:, s_idx, :].unsqueeze(-1)
        cont_embd[ind, s_idx, :] = self.cont_encoder['continuous'][head_name](data_x) * data_mask
    
    #Ensure all heads are used for torchrun
    total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype_float)
    for head_name, head in self.cont_encoder['continuous'].items():
      if head_name not in chart_type_dict:
        blank = torch.zeros( [1,1,REG_DIMS[head_name]], device=self.device)
        total_loss += (head(blank) * 0).sum()

    ### Add position embds here
    series_count, pt_count = cont_embd.size(1), cont_embd.size(2)

    if series_count > self.max_series_blocks:
      randints = torch.tensor(
        sorted(random.sample(range(series_count), self.max_series_blocks)), dtype=torch.long, device=self.device)
      cont_embd = torch.index_select(cont_embd, dim=1, index=randints)
      cont_mask = torch.index_select(cont_mask, dim=1, index=randints)
    
    if pt_count > self.max_cont_blocks:
      randints = torch.tensor(
        sorted(random.sample(range(pt_count), self.max_cont_blocks)), dtype=torch.long, device=self.device)
      cont_embd = torch.index_select(cont_embd, dim=2, index=randints)
      cont_mask = torch.index_select(cont_mask, dim=2, index=randints)

    cont_embd = cont_embd + (self.pos_emb_cont[:,:series_count,:pt_count,:] if self.pos_emb_cont is not None else 0.0)

    if flatten_series:
      cont_embd = torch.flatten(cont_embd, start_dim=1, end_dim=2)
      cont_mask = torch.flatten(cont_mask, start_dim=1, end_dim=2)

    if pad_len is not None and isinstance(pad_len, int):
      cont_embd = pad_vector(cont_embd, pad_len, pad_dim, self.device, self.dtype_float)
      cont_mask = pad_vector(cont_mask, pad_len, pad_dim, self.device, self.dtype_float)
    
    return  cont_embd, cont_mask, total_loss


  def preencode_scale(self, inputs_embeds, chart_type_dict, attention_mask=None, pad_len=None, pad_dim=1):

    bsz = len(inputs_embeds)
    #Create attention mask if it doesnt exist
    if attention_mask is not None and attention_mask[0] is not None:
      scale_mask = torch.stack(attention_mask, dim=0).to(self.device)
    else:
      scale_mask = torch.ones((bsz), dtype=self.dtype_float, device=self.device)

    emb_len = min(scale_mask.size(1), self.max_series_blocks)
    scale_embd = torch.zeros((bsz, emb_len, self.out_dim), dtype=self.dtype_float, device=self.device)

    for head_name, ind in chart_type_dict.items():
      scale_x     = torch.stack([v for idx, v in enumerate(inputs_embeds) if idx in ind], dim=0).to(self.device)

      scale_enc = self.cont_encoder['scale'][head_name](scale_x) 
      series_count = scale_enc.size(1)


      scale_enc    = scale_enc + (self.pos_emb_scale[:,:series_count,:] if self.pos_emb_scale is not None else 0.0)

      scale_m    = scale_mask[ind, :].unsqueeze(-1)
      scale_enc *= scale_m
      scale_embd[ind, :, :]  = scale_enc
    
    #Ensure all heads are used for torchrun
    total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype_float)
    for head_name, head in self.cont_encoder['scale'].items():
      if head_name not in chart_type_dict:
        blank = torch.zeros([1,1,SCALE_DIMS['minmax'][head_name]], device=self.device)
        total_loss += (head(blank) * 0).sum()

    ### Padding to the right
    if pad_len is not None and isinstance(pad_len, int):
      scale_embd = pad_vector(scale_embd, pad_len, pad_dim, self.device, self.dtype_float)
      scale_mask = pad_vector(scale_mask, pad_len, pad_dim, self.device, self.dtype_float)


    return scale_embd, scale_mask, total_loss