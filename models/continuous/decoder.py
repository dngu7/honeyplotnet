# ---------------------------------------------------------------
# Copyright (c) Cybersecurity Cooperative Research Centre 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .helper import (
  make_repeat_frame, 
  unflat_tensor,
  get_win_hypo
)

from utils import PAD_IDX
from ..constant import SCALE_DIMS, REG_DIMS

from .coder import Coder

class Decoder(Coder):
  def __init__(self,
    dec_conv,
    dec_proj_col,
    dec_proj_row,
    dec_tf_col,
    dec_tf_row,
    emb_dim1,
    cont_loss_fn,
    scale_loss_fn,
    max_series_blocks,
    max_cont_blocks,
    norm_mode,
    scale_floor,
    use_mhd,
    hypothese_bsz,
    hypothese_dim=1,
    device='cuda:0', 
    debug=False,
    ):
    super().__init__()

    self.dec_tf_col = dec_tf_col
    self.dec_tf_row = dec_tf_row 

    self.dec_conv = dec_conv
    self.dec_proj_col = dec_proj_col
    self.dec_proj_row = dec_proj_row

    self.dec_col_head    = nn.Linear(emb_dim1, 2)
    self.dec_row_head    = nn.Linear(emb_dim1, 2)

    #continuous embeddings
    self.cont_decoder = nn.ModuleDict()
    for name in ['scale', 'continuous']:
      self.cont_decoder[name] = nn.ModuleDict()

    for head_name, head_dim in SCALE_DIMS[norm_mode].items():
      self.cont_decoder['scale'][head_name] = nn.ModuleList([nn.Linear(emb_dim1, head_dim) for _ in range(1)])

    for head_name, head_dim in REG_DIMS.items():
      self.cont_decoder['continuous'][head_name] = nn.Linear(emb_dim1, head_dim)

    self.emb_dim1 = emb_dim1

    self.use_mhd = use_mhd
    self.hypothese_bsz = hypothese_bsz
    self.hypothese_dim = hypothese_dim
    self.max_cont_blocks = max_cont_blocks
    self.max_series_blocks = max_series_blocks
    self.scale_floor = scale_floor
    
    self.device = device
    self.debug = debug
    self.dtype_float = torch.float32
    self.cont_loss_fn  = F.smooth_l1_loss if cont_loss_fn == 'l1' else F.mse_loss
    self.scale_loss_fn = F.smooth_l1_loss if scale_loss_fn == 'l1' else F.mse_loss
    self.ce_loss_fn    = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='none')

    self.apply(self._init_weights)


  def forward(self, y_hat, chart_type_dict, cond, labels, wta_idx, 
    scale_embd, scale_mask, cont_embd, cont_mask):
    
    if len(y_hat.shape) == 4 and self.use_mhd and wta_idx is not None:
      if cond is not None:
          repeat_frame = make_repeat_frame(cond, hypo_bsz=self.hypothese_bsz)
          cond = cond.unsqueeze(self.hypothese_dim).repeat(repeat_frame)
          cond = torch.flatten(cond, start_dim=0, end_dim=1)

      y_hat = torch.flatten(y_hat, start_dim=0, end_dim=1)

    if cond is not None:
      y_hat = torch.cat([cond, y_hat], dim=1)

    dec_hidden_col, dec_hidden_row = self.decode_y_hat(y_hat)

    if self.use_mhd and wta_idx is not None:
      repeat_frame = make_repeat_frame(scale_embd, hypo_bsz=self.hypothese_bsz)
      scale_embd = scale_embd.unsqueeze(self.hypothese_dim).repeat(repeat_frame)
      scale_embd = torch.flatten(scale_embd, start_dim=0, end_dim=1)

      repeat_frame = make_repeat_frame(scale_mask, hypo_bsz=self.hypothese_bsz)
      scale_mask = scale_mask.unsqueeze(self.hypothese_dim).repeat(repeat_frame)
      scale_mask = torch.flatten(scale_mask, start_dim=0, end_dim=1)

      repeat_frame = make_repeat_frame(cont_embd, hypo_bsz=self.hypothese_bsz)
      cont_embd = cont_embd.unsqueeze(self.hypothese_dim).repeat(repeat_frame)
      cont_embd = torch.flatten(cont_embd, start_dim=0, end_dim=1)

      repeat_frame = make_repeat_frame(cont_mask, hypo_bsz=self.hypothese_bsz)
      cont_mask = cont_mask.unsqueeze(self.hypothese_dim).repeat(repeat_frame)
      cont_mask = torch.flatten(cont_mask, start_dim=0, end_dim=1)

    hidden_row = self.dec_tf_row(
      inputs_embeds=scale_embd,
      attention_mask=scale_mask,
      encoder_hidden_states=dec_hidden_row
    ).last_hidden_state

    row_len = cont_embd.size(1)
    hidden_col = []
    for ridx in range(row_len):
      row_cont_embd = cont_embd[:, ridx,:]
      row_cont_mask = cont_mask[:, ridx,:]

      h = self.dec_tf_col(
        inputs_embeds=row_cont_embd,
        attention_mask=row_cont_mask,
        encoder_hidden_states=dec_hidden_col
      ).last_hidden_state
      hidden_col += [h]
    
    hidden_col = torch.stack(hidden_col, dim=1)

    col_logit, row_logit, tab_loss, tab_logs = self.decode_tab_shape(
      hidden_col, hidden_row, labels, wta_idx=wta_idx, chart_type_dict=chart_type_dict)

    loss = {**tab_loss}
    logs = {**tab_logs} #logs are split by chart data. This is used for metrics

    outputs = {}
    outputs['row']        = row_logit
    outputs['col']        = col_logit

    scale_pred, scale_loss, scale_logs = self.decode_scale(
      hidden_row, chart_type_dict, labels, wta_idx=wta_idx)

    outputs['scale']     = scale_pred
    loss['scale']      = scale_loss
    logs['scale']      = scale_logs

    cont_pred, cont_loss, cont_logs = self.decode_continuous(
      hidden_col, hidden_row, chart_type_dict, labels, wta_idx=wta_idx)

    outputs['continuous'] = cont_pred
    loss['cont']  = cont_loss
    logs['cont']  = cont_logs

    if len(y_hat.shape) == 4 and self.use_mhd and wta_idx is not None:
      keys = list(outputs.keys())
      for k in keys:
        outputs[k] = unflat_tensor(outputs[k], self.hypothese_bsz) 
    
    return outputs, loss, logs

  def decode_y_hat(self, y_hat):
    h = self.dec_conv(y_hat)
    dec_hidden_col  = self.dec_proj_col(h)
    dec_hidden_row  = self.dec_proj_row(h)
    return dec_hidden_col, dec_hidden_row

  def decode_tab_shape(self, hidden_col, hidden_row, labels=None, wta_idx=None, chart_type_dict=None):
    '''
    Predict the number of rows and columns
    '''
    ## Check if column is flattened. If not just average through it
    if len(hidden_col.shape) == 4:
      hidden_col = hidden_col.mean(1)

    col_logit = self.dec_col_head(hidden_col)
    row_logit = self.dec_row_head(hidden_row)

    loss_dict = {}
    logs = {'row': {}, 'col': {}}
    if labels is not None:
      
      if self.use_mhd and wta_idx is not None: #, "Must provide winner idx for this"
          unflat_col_logit = unflat_tensor(col_logit, self.hypothese_bsz)
          unflat_row_logit = unflat_tensor(row_logit, self.hypothese_bsz)

          col_logit = get_win_hypo(unflat_col_logit, wta_idx)
          row_logit = get_win_hypo(unflat_row_logit, wta_idx)

      labels = labels['chart_data'].get('labels')
      col_label = labels['col'].to(self.device)
      row_label = labels['row'].to(self.device)
      bsz = col_logit.size(0)
      
      pad_col_len = col_logit.size(1) - col_label.size(1)
      pad_row_len = row_logit.size(1) - row_label.size(1)

      assert pad_col_len >= 0 and pad_row_len >= 0, "column or row label exceed max size. Limit: C={} R={} Labels: C={} R={}".format(
        col_logit.size(1), row_logit.size(1), labels['col'].size(1), labels['row'].size(1)
        )

      pad_col = torch.ones((bsz, pad_col_len), device=self.device, dtype=torch.long) * PAD_IDX
      pad_row = torch.ones((bsz, pad_row_len), device=self.device, dtype=torch.long) * PAD_IDX

      col_label = torch.cat([col_label, pad_col], dim=1)
      row_label = torch.cat([row_label, pad_row], dim=1)

      col_label_flat = torch.flatten(col_label, start_dim=0, end_dim=1)
      row_label_flat = torch.flatten(row_label, start_dim=0, end_dim=1)

      col_logit_flat = torch.flatten(col_logit, start_dim=0, end_dim=1)
      row_logit_flat = torch.flatten(row_logit, start_dim=0, end_dim=1)

      col_nll = self.ce_loss_fn(col_logit_flat, col_label_flat)
      row_nll = self.ce_loss_fn(row_logit_flat, row_label_flat)

      col_counts = torch.where(col_label == PAD_IDX, 
        torch.tensor(0.0, device=self.device, dtype=self.dtype_float), 
        torch.tensor(1.0, device=self.device, dtype=self.dtype_float))
      
      row_counts = torch.where(row_label == PAD_IDX, 
        torch.tensor(0.0, device=self.device, dtype=self.dtype_float), 
        torch.tensor(1.0, device=self.device, dtype=self.dtype_float))

      loss_dict['col'] = col_nll.sum() / col_counts.sum()
      loss_dict['row'] = row_nll.sum() / row_counts.sum()
      
      if chart_type_dict is not None:

        # Metrics remove the zeros
        col_nll_unflat = col_nll.view(bsz, -1).sum(-1)
        row_nll_unflat = row_nll.view(bsz, -1).sum(-1)

        col_count_unflat = col_counts.view(bsz, -1).sum(-1)
        row_count_unflat = row_counts.view(bsz, -1).sum(-1)

        col_nll_unflat = torch.nan_to_num(col_nll_unflat / col_count_unflat, nan=0.0)
        row_nll_unflat = row_nll_unflat / row_count_unflat
        
        for chart_type, indices in chart_type_dict.items():
          logs['col'][chart_type] = col_nll_unflat[indices].detach().cpu()
          logs['row'][chart_type] = row_nll_unflat[indices].detach().cpu()      

    return col_logit, row_logit, loss_dict, logs


  def decode_continuous(self, hidden_col, hidden_row, chart_type_dict, labels=None, wta_idx=None):
    '''
    Predict continuous values for each row and column
    Prepend each column prediction with the particular row
    '''

    
    is_unflat = len(hidden_col.shape) == 4 

    if self.use_mhd and wta_idx is not None:
      hidden_row = unflat_tensor(hidden_row, self.hypothese_bsz)
      hidden_col = unflat_tensor(hidden_col, self.hypothese_bsz)

    total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype_float)
    all_predictions = [[] for _ in range(hidden_col.size(0))]

    max_rows = hidden_row.size(1)
    if labels is not None:
      cd_label = labels['chart_data']
      max_rows = max(emb.size(0) for emb in cd_label['continuous']['inputs_embeds'])


    #Ensure all heads are used for torchrun
    for head_name, head in self.cont_decoder['continuous'].items():
      if head_name not in chart_type_dict:
        total_loss += (head(hidden_col) * 0).sum()

    logs = {}
    #Loops through each head and creates an embedding based on the number of continous variables and scale variables
    for head_name, ind in chart_type_dict.items():
      logs[head_name] = []

      for row_idx in range(max_rows):
        if is_unflat:
          hid_col = hidden_col[ind, row_idx, : ]
        else:
          hid_col = hidden_col[ind, :]
        
        if self.use_mhd and wta_idx is not None:
          hid_col = torch.flatten(hid_col, start_dim=0, end_dim=1)
        
        cont_logits = self.cont_decoder['continuous'][head_name](hid_col)
        cont_pred   = torch.sigmoid(cont_logits)

        if labels is not None:

          if self.use_mhd and wta_idx is not None: # "Must provide winner idx for this"
              unflat_cont_logit = unflat_tensor(cont_pred, self.hypothese_bsz)
              cont_pred = self.mhd_layer.get_win_hypo(unflat_cont_logit, wta_idx[ind])

          label = torch.stack([l[row_idx,:] for idx, l in enumerate(cd_label['continuous']['inputs_embeds']) if idx in ind], dim=0).to(self.device)
          mask = torch.stack([l[row_idx,:] for idx, l in enumerate(cd_label['continuous']['attention_mask']) if idx in ind], dim=0).to(self.device)

          #Pad labels to match max len
          pad_label_len = cont_pred.size(1) - label.size(1)
          assert pad_label_len >= 0, "Targets contain length={} larger than max={} | Shapes label={} pred={}".format(
            label.size(1), cont_pred.size(1), label.shape, cont_pred.shape
          )
          pad_label = torch.zeros((label.size(0), pad_label_len, label.size(-1)), device=self.device, dtype=self.dtype_float)
          pad_mask  = torch.zeros((label.size(0), pad_label_len), device=self.device, dtype=self.dtype_float)


          label = torch.cat([label, pad_label], dim=1)
          mask = torch.cat([mask, pad_mask], dim=1)

          loss = self.cont_loss_fn(cont_pred, label, reduction='none').mean(-1)
          rec_error = F.mse_loss(cont_pred.detach(), label, reduction='none').mean(-1).detach()

          #Reduce this way to ignore the zeros. Average across batch
          counts = mask.sum(-1)

          #Protect against divide by zero
          counts = torch.where(counts == 0, torch.tensor(0.0, device=self.device, dtype=self.dtype_float), 1 / counts)

          loss = (loss * mask * counts.unsqueeze(-1)).sum(-1)
          rec_error = (rec_error * mask * counts.unsqueeze(-1)).sum(-1)

          c = counts.nonzero()
          logs[head_name].append(rec_error[c])
          total_loss += loss.mean(-1)
          
        for pidx, pred in zip(ind, cont_pred):
          all_predictions[pidx].append(pred)

      logs[head_name] = torch.cat(logs[head_name], dim=0).view(-1).detach().cpu()

    # Stack all predictions. Each chart type has different dimension
    for pidx in range(len(all_predictions)):
      p = all_predictions[pidx]
      assert len(p) > 0, "One of the continuous predictions not recorded."
      all_predictions[pidx] = torch.stack(p, dim=0)

    return all_predictions, total_loss, logs

  def decode_scale(self, hidden_row, chart_type_dict, labels=None, wta_idx=None, greedy=False):
    if labels is not None:
      cd_label = labels['chart_data']

    if self.use_mhd and wta_idx is not None:
      hidden_row = unflat_tensor(hidden_row, self.hypothese_bsz)

    logs = {}
    total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype_float)
    bsz = hidden_row.size(0) 
    all_predictions = [None for _ in range(bsz)] 

    #Ensure all heads are used for torchrun
    for head_name, head in self.cont_decoder['scale'].items():
      if head_name not in chart_type_dict:
        total_loss += (head[0](hidden_row)* 0).sum()
          
    for head_name, ind in chart_type_dict.items():

      hid_row = hidden_row[ind, :, :]

      if self.use_mhd and wta_idx is not None:
        hid_row = torch.flatten(hid_row, start_dim=0, end_dim=1)

      scale_logit  = self.cont_decoder['scale'][head_name][0](hid_row)
      scale_logit  = self.dim_wise_relu(scale_logit, head_name=head_name, decoder_name='scale')

      if labels is not None:
        ## Compute loss

        # Collect labels and mask
        label = torch.stack([l for idx, l in enumerate(cd_label['scale']['inputs_embeds']) if idx in ind], dim=0).to(self.device)
        
        mask = torch.stack([l for idx, l in enumerate(cd_label['scale']['attention_mask']) if idx in ind], dim=0).to(self.device)
        pad_label_len = scale_logit.size(1) - label.size(1)
        assert pad_label_len >= 0, "Targets contain length={} larger than max={} | tensor shape = label: {} logits: {}".format(
          label.size(1), scale_logit.size(1), label.shape, scale_logit.shape
        )
        pad_label = torch.zeros((label.size(0), pad_label_len, label.size(-1)), device=self.device, dtype=self.dtype_float)
        pad_mask  = torch.zeros((label.size(0), pad_label_len), device=self.device, dtype=self.dtype_float)

        label = torch.cat([label, pad_label], dim=1)
        mask  = torch.cat([mask, pad_mask], dim=1)
        counts = mask.sum(-1)

        #Prevents div by zero
        counts = torch.where(counts == 0, torch.tensor(0.0, device=self.device, dtype=self.dtype_float), 1/counts)

        if self.use_mhd and wta_idx is not None: #, "Must provide winner idx for this"
          unflat_scale_logit = unflat_tensor(scale_logit, self.hypothese_bsz)
          scale_logit = get_win_hypo(unflat_scale_logit, wta_idx[ind])

        #Compute distance loss
        dist_loss = self.scale_loss_fn(scale_logit, label, reduction='none')
        dist_loss = torch.clip(dist_loss, min=-1e2, max=1e2)
        dist_loss = dist_loss.mean(-1) * mask


        loss = (dist_loss.sum(-1) * counts).mean(-1)
        total_loss += loss.mean()

        logs[head_name] = loss.detach().cpu()

      #Cannot stack because last dimension can vary by chart type
      for idx, pred in zip(ind, scale_logit):
        all_predictions[idx] = pred
    assert all(p is not None for p in all_predictions), "One of scale predictions not recorded."
    return all_predictions, total_loss, logs
    
  def dim_wise_relu(self, x, head_name=None, decoder_name=None):
    dims = x.size(-1)

    if head_name == 'boxplot' and decoder_name == 'continuous':
      incrent_dim = [i for i in range(dims) if i > 0]
      x[:,:,incrent_dim] = torch.relu(x[:,:,incrent_dim])

    elif decoder_name == 'scale':
      assert len(x.shape) == 3, "unsupported shape"
      
      min_dims = [i for i in range(dims) if (i % 2) == 0]
      rng_dims = [i for i in range(dims) if (i % 2) == 1]

      x[:,:,min_dims] = torch.relu(x[:,:,min_dims] + self.scale_floor[0]) - self.scale_floor[0]
      x[:,:,rng_dims] = torch.relu(x[:,:,rng_dims] + self.scale_floor[1]) - self.scale_floor[1]
    else:
      raise NotImplementedError()

    return x