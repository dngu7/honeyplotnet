# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------


from ..constant import UNIQ_CHART_HEADS, CHART_TO_HEAD_IDX

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .encoder import Encoder
from .decoder import Decoder

import numpy as np
import math

from .helper import (
  unflat_tensor,
  get_chart_type_dict,
  get_chart_type_from_idx,
  make_repeat_frame
)


class BaseDataModel(nn.Module):
  def __init__(self, 
    enc_conv, enc_proj1, vq_layer1,
    dec_conv, dec_proj_col, dec_proj_row, 
    dec_tf_col, dec_tf_row, enc_tf=None, 
    use_mhd=False, mhd_layer=None, enc_proj2=None, enc_proj3=None, 
    vq_layer2=None, emb_dim1=8,  emb_len1=4, emb_len2=None, 
    max_series_blocks=5, max_cont_blocks=32, 
    conditional_encoder=False, conditional_decoder=True, 
    scale_mode='log10', scale_eps=1.00001,
    scale_floor=-1.0, norm_mode='minmax',
    cont_loss_fn='l1', scale_loss_fn='l1', 
    use_pos_embs=True, hypothese_bsz=2048, hypothese_count=2048,
    device='cpu', fp16=False, debug=False):
    super().__init__()
    
    if use_mhd:
      assert enc_proj2 is not None and vq_layer2 is not None and mhd_layer is not None

    self.debug = debug

    self.device       = device 
    self.use_fp16     = fp16
    self.dtype_float  = torch.float32 
    self.dtype_int    = torch.int32 
    self.use_mhd      = use_mhd

    self.norm_mode    = norm_mode

    self.encoder = Encoder(
      enc_conv=enc_conv,
      enc_tf=enc_tf,
      out_dim=emb_dim1,
      max_series_blocks=max_series_blocks,
      max_cont_blocks=max_cont_blocks,
      use_pos_embs=use_pos_embs,
      norm_mode=norm_mode,
      chart_type_conditional=conditional_encoder,
      device=device,
      debug=debug
    )

    self.enc_proj1 = enc_proj1
    self.vq_layer1   = vq_layer1

    if self.use_mhd:
      self.mhd_layer         = mhd_layer
      self.vq_layer2         = vq_layer2
      self.up_sample_d       = nn.ConvTranspose1d(emb_len2, emb_len1, 1)
      self.enc_proj2         = enc_proj2
      self.enc_proj3         = enc_proj3

      self.hypothese_bsz     = hypothese_bsz
      self.hypothese_count   = hypothese_count


    self.decoder = Decoder(
      dec_conv=dec_conv,
      dec_proj_col=dec_proj_col,
      dec_proj_row=dec_proj_row,
      dec_tf_col=dec_tf_col,
      dec_tf_row=dec_tf_row,
      emb_dim1=emb_dim1,
      cont_loss_fn=cont_loss_fn,
      scale_loss_fn=scale_loss_fn,
      max_series_blocks=max_series_blocks,
      max_cont_blocks=max_cont_blocks,
      use_mhd=use_mhd,
      hypothese_bsz=hypothese_bsz,
      scale_floor=scale_floor,
      norm_mode=norm_mode,
      device=device,
      debug=debug
    )

    #Chart type embeddings
    self.ct_emb      = nn.Embedding(len(UNIQ_CHART_HEADS), emb_dim1)

    #2. Continuous data -> Scale/Data settings (self.data_embed)
    self.max_series_blocks = max_series_blocks
    self.max_cont_blocks   = max_cont_blocks

    self.conditional_decoder = conditional_decoder

    #positional embeddings
    self.use_pos_embs = use_pos_embs
    self.pos_emb_scale = None

    self.scale_exponent = 10 if scale_mode == 'log10' else math.e
    self.scale_eps   = scale_eps
    self.scale_floor = scale_floor

    self.emb_dim1 = emb_dim1

  def get_chart_type_emb(self, ct_idx=None, chart_type=None): 
    if ct_idx is None and chart_type is None:
      raise ValueError("Need to provide either indinces or string")

    if chart_type is not None:
      ct_idx = [CHART_TO_HEAD_IDX[ct] for ct in chart_type] 
      ct_idx = torch.tensor(ct_idx, dtype=torch.long, device=self.device).view(-1,1)

    return self.ct_emb(ct_idx)
  
  def run_encoder(self, inputs):
    return self.encoder(inputs)

  def quantize(self, encoder_output):
    outputs = {}
    loss = {}

    y1 = self.enc_proj1(encoder_output)
    c_base, _, cb_loss1, _ = self.vq_layer1(y1)

    loss['cb1'] = cb_loss1

    if self.use_mhd:
      y2 = self.enc_proj2(encoder_output)
      y2 = self.enc_proj3(y2)
      c_i, _, cb_loss2, _ = self.vq_layer2(y2)
       
      loss['cb2']                  = cb_loss2
      
      outputs['c_base']            = c_base
      outputs['secondary_latents'] = c_i
      outputs['latent_tgt']        = y1
      outputs['hypothese_count']   = self.hypothese_count
    else:
      outputs['y_hat'] = c_base

    return outputs, loss

  def get_topk_batch(self, topk_idx, x):
    x_batch = []
    for idx, _x  in zip(topk_idx, x):
        x_batch.append(_x[idx])
    return torch.stack(x_batch, dim=0)

  def run_mhd_layer(self, c_base, hypothese_count, latent_tgt=None, secondary_latents=None, split='train', keep_len_limit=128):

    #Reshape for mhd layer
    if secondary_latents is None:
        inp_latents = c_base
    else:
        inp_latents = secondary_latents

    if latent_tgt is not None:
        bsz = self.hypothese_bsz
    else:
        bsz = hypothese_count
        
    num_batches = int(np.ceil(hypothese_count / bsz))
    keep_length = int(np.ceil(bsz / num_batches))

    repeat_frame = make_repeat_frame(c_base, hypo_bsz=bsz)
    expanded_c = c_base.unsqueeze(self.mhd_layer.hypothese_dim).repeat(repeat_frame)
  
    batch_hypos = []
    diff_vec_container = []

    for _ in range(num_batches):
      diff_vector = self.mhd_layer(inp_latents, hypothese_count=bsz) 
      
      #Flatten along batch
      if self.up_sample_d is not None:
        diff_vector = torch.flatten(diff_vector, start_dim=0, end_dim=1)
        diff_vector = self.up_sample_d(diff_vector)
        diff_vector = unflat_tensor(diff_vector, bsz)
 
      y_hat = expanded_c + diff_vector

      if latent_tgt is not None:

          dist_loss, topk_idx = self.mhd_layer.get_dist_loss(y_hat, latent_tgt, topk=keep_length)

          ### Only keep the top x based on meta loss
          keep_idx = topk_idx[:,:keep_length]
          topk_hypos = self.get_topk_batch(keep_idx, y_hat)
          batch_hypos.append(topk_hypos)

          diff_vector = self.get_topk_batch(keep_idx, diff_vector)
          diff_vec_container.append(diff_vector)

      wta_loss = None
      mhd_dict = {}
      mhd_dict['bsz'] = bsz
      mhd_dict['keep_length'] = keep_length
      if latent_tgt is not None:

          batch_hypos  = torch.cat(batch_hypos, dim=1)[:,:bsz,:]
          diff_vector  = torch.cat(diff_vec_container, dim=1)[:,:bsz,:]

          # Recombine into y_hat (final list)
          y_hat = batch_hypos

          #Get the winner out of the batched winners
          dist_loss, topk_idx = self.mhd_layer.get_dist_loss(y_hat, latent_tgt, topk=keep_length)
          wta_loss, wta_idx   = self.mhd_layer.get_wta_loss(dist_loss, topk_idx)

          mhd_dict['yw_hat']   = self.mhd_layer.get_win_hypo(y_hat, wta_idx)
          #mhd_dict['yk_hat'] = self.mhd_layer.get_topk_hypo(y_hat, topk_idx)
          
          mhd_dict['wta_idx']  = wta_idx
          mhd_dict['topk_idx'] = topk_idx

          wta_loss = wta_loss.mean()

      mhd_dict['c_base']      = expanded_c
      mhd_dict['diff_vector'] = diff_vector
      mhd_dict['y_hat']       = y_hat 

      return mhd_dict, wta_loss

  def run_decoder(self, y_hat, chart_type_dict, cond=None, labels=None, wta_idx=None):
    
    # "decoder_inputs_embeds" are the right shifted inputs. 
    # See dataset.continuous.py (lines 599)
    scale_embd, scale_mask, s_loss = self.encoder.preencode_scale(
          inputs_embeds=labels['chart_data']['scale']['decoder_inputs_embeds'],
          attention_mask=labels['chart_data']['scale']['decoder_attention_mask'],
          chart_type_dict=chart_type_dict,
          pad_len=self.max_series_blocks, pad_dim=1)

    cont_embd, cont_mask, c_loss = self.encoder.preencode_continuous( 
        inputs_embeds=labels['chart_data']['continuous']['decoder_inputs_embeds'],
        attention_mask=labels['chart_data']['continuous']['decoder_attention_mask'],
        chart_type_dict=chart_type_dict,
        pad_len=self.max_cont_blocks,
        pad_dim=2,
        flatten_series=False)

    loss = s_loss + c_loss

    decoder_output, dec_loss, logs = self.decoder(
      y_hat, chart_type_dict, cond, labels, wta_idx, 
      scale_embd, scale_mask, cont_embd, cont_mask
    )

    dec_loss['na2'] = loss

    return decoder_output, dec_loss, logs

  def reconstruct_tab_shape(self, row_logits, col_logits, temp=1.0, eval=False):

    row_probs = F.softmax(row_logits / temp, dim=-1)
    col_probs = F.softmax(col_logits / temp, dim=-1)

    row_sample = Categorical(probs=row_probs).sample()
    col_sample = Categorical(probs=col_probs).sample()

    bsz = row_sample.size(0)

    #Force first series to always be active for stability
    row_sample[:, 0] = 1
    col_sample[:, 0] = 1

    #Count from left until 0 is reached. 
    row_count = torch.zeros(bsz, device=self.device, dtype=torch.long)
    row_stop  = torch.ones(bsz, device=self.device, dtype=torch.long)

    col_count = torch.zeros(bsz, device=self.device, dtype=torch.long)
    col_stop  = torch.ones(bsz, device=self.device, dtype=torch.long)

    search_len = max(row_sample.size(1), col_sample.size(1))
    for idx in range(search_len):
      if idx < row_sample.size(1):
        
        row = row_sample[:, idx]
        row_stop  *= row
        row_count += (row_stop * row)

      if idx < col_sample.size(1):

        col = col_sample[:, idx]
        col_stop  *= col
        col_count += (col_stop * col)
      
      if (row_stop.sum(-1) + col_stop.sum(-1)) == 0:
        break
    
    row_count = torch.where(row_count == 0, torch.ones_like(row_count), row_count)
    col_count = torch.where(col_count == 0, torch.ones_like(col_count), col_count)

    samples = {'row': row_sample, 'col': col_sample}
    tab_shape  = {'row': row_count, 'col': col_count}
    return tab_shape, samples

  def eval_loop(self, inputs, temp=1.0, split='eval', return_labels=True):

    assert temp > 0.0 and temp <= 1.0, "invalid temp given"
    samples, loss_dict, metric_logs = self.train_loop(inputs)

    return samples, loss_dict, metric_logs
  
  def batch_outputs(self, outputs, to_np=True):
    #Batches for easy index and converts to numpy
    #batched_outputs = [{} for _ in range(bsz)] #[{}] * bsz
    batched_outputs = []
    for k, v in outputs.items():
      for bidx, vi in enumerate(v):
        if isinstance(vi, torch.Tensor):
          vi = vi.detach().cpu().tolist()
          if to_np:
            vi = vi.numpy()

        if len(batched_outputs) <= bidx:
          batched_outputs.append({})
        batched_outputs[bidx][k] = vi
    return batched_outputs

  def sample_codebook(self, inputs):
    encoder_output, _ = self.run_encoder(inputs)

    y1 = self.enc_proj1(encoder_output)
    cb_ind1, _ = self.vq_layer1.get_code_indices(y1)
    cb_ind1 = cb_ind1.view(y1.size(0), -1)

    if not self.use_mhd:
      return (cb_ind1, )
    else:
      y2 = self.enc_proj2(encoder_output)
      y2 = self.enc_proj3(y2)

      cb_ind2, _ = self.vq_layer2.get_code_indices(y2)
      cb_ind2 = cb_ind2.view(y2.size(0), -1)

      return (cb_ind1, cb_ind2, )

  def reconstruct_from_indices(self, ct_idx, cb_ind1, cb_ind2=None, hypo_count=4, hypo_bsz=4, temp=1.0, greedy=True):

    # Obtain embeddings from vq codebooks
    c_base = self.vq_layer1(cb_ind1, is_indices=True)

      
    c_i = None
    if self.use_mhd:
      c_i = self.vq_layer2(cb_ind2, is_indices=True)      

      decoder_input, _ = self.run_mhd_layer(
        c_base=c_base, 
        secondary_latents=c_i, 
        hypothese_count=hypo_count, 
        )

      y_hat = decoder_input['y_hat']
      y_hat = y_hat[:,0]
    else:
      y_hat = c_base


    # Obtain chart type embedding
    cond = None
    if self.conditional_decoder and ct_idx is not None: 
      cond = self.get_chart_type_emb(ct_idx=ct_idx)

    if len(y_hat.shape) == 4 and self.use_mhd and cond is not None and hypo_bsz > 1:
      repeat_frame = make_repeat_frame(cond, hypo_bsz=hypo_bsz)
      cond = cond.unsqueeze(self.mhd_layer.hypothese_dim).repeat(repeat_frame)
      cond = torch.flatten(cond, start_dim=0, end_dim=1)

    #Prepend chart type embedding    
    if cond is not None:
      y_hat = torch.cat([cond, y_hat], dim=1)

    # For decoder heads
    chart_type_dict = get_chart_type_from_idx(ct_idx)

    
    #Split y_hat into two paths: Bottom and top
    dec_hidden_col, dec_hidden_row = self.decoder.decode_y_hat(y_hat)

    ################################
    scale_preds, hidden_row = self.generate_scale(dec_hidden_row, chart_type_dict)
    
    ################################
    cont_preds, hidden_col  = self.generate_continuous(dec_hidden_col, chart_type_dict)

    col_logit, row_logit, _, _ = self.decoder.decode_tab_shape(hidden_col, hidden_row)

    x_hat = self.reconstruct_x(row_logit, col_logit, scale_preds, cont_preds, eval=True)

    #Repeat chart type for each hypothesis
    x_hat['ct_idx'] = ct_idx
    x_hat['chart_type_dict'] = chart_type_dict

    return x_hat

  def train_loop(self, inputs):

    encoder_output, loss = self.run_encoder(inputs)

    decoder_input, cb_loss = self.quantize(encoder_output)

    all_loss = cb_loss
    all_loss['na1'] = loss

    wta_idx = None
    if self.use_mhd:
      decoder_input, wta_loss = self.run_mhd_layer(**decoder_input)
      wta_idx = decoder_input.get('wta_idx')
      all_loss['wta'] = wta_loss

    ct_cond = None
    chart_type = inputs['chart_data']['chart_type']
    if self.conditional_decoder: 
      ct_cond = self.get_chart_type_emb(chart_type=chart_type)

    y_hat = decoder_input['y_hat']
    if wta_idx is not None:
      y_hat = self.mhd_layer.get_win_hypo(y_hat, wta_idx)

    chart_type_dict = get_chart_type_dict(chart_type)

    decoder_output, dec_loss, logs = self.run_decoder(
      y_hat=y_hat, 
      chart_type_dict=chart_type_dict, 
      cond=ct_cond, 
      labels=inputs,
      wta_idx=None
      )

    #Prepare loss and logs for outputs
    all_loss = {**all_loss, **dec_loss}

    row_logit   = decoder_output['row']
    col_logit   = decoder_output['col']
    scale_preds = decoder_output['scale']
    cont_preds  = decoder_output['continuous']

    #Convert to input shape
    x_hat = self.reconstruct_x(row_logit, col_logit, scale_preds, cont_preds, eval=False)
    x_hat['chart_type'] = inputs['chart_data']['chart_type']
    x_hat['ct_idx'] = [CHART_TO_HEAD_IDX[ct] for ct in inputs['chart_data']['chart_type']]
    x_hat['chart_type_dict'] = chart_type_dict

    return x_hat, all_loss, logs 

  def reconstruct_x(self, row_logit, col_logit, scale_preds, cont_preds, eval=False):

    tab_shape, tab_samples = self.reconstruct_tab_shape(row_logit, col_logit, eval=eval)

    #Convert predictions into values (unscaled)
    #scale_values = self.unscale_scales(scale_preds, tab_shape)
    #cont_values  = self.unscale_continuous(cont_preds, scale_values, tab_shape)

    scale_embeds, scale_mask = self.embed_scale(scale_preds, tab_shape, tab_samples)
    cont_embeds, cont_mask   = self.embed_continuous(cont_preds, tab_shape, tab_samples)

    samples = {}
    samples['shape'] = {}
    samples['shape']['counts'] = tab_shape
    samples['shape']['embeds'] = tab_samples

    samples['scale'] = {}
    #samples['scale']['output'] = scale_values
    samples['scale']['inputs_embeds'] = scale_embeds
    samples['scale']['attention_mask'] = scale_mask

    samples['continuous'] = {}
    #samples['continuous']['output'] = cont_values
    samples['continuous']['inputs_embeds'] = cont_embeds
    samples['continuous']['attention_mask'] = cont_mask

    return samples

  def forward(self, inputs, is_train=True, temp=1.0, **kwargs):
    if is_train:
      return self.train_loop(inputs)
    else:
      return self.eval_loop(inputs, temp)
  

  def generate_scale(self, dec_hidden_row, chart_type_dict):

    bsz = dec_hidden_row.size(0)
    scale_embd = torch.zeros([bsz, 1, self.emb_dim1], device=self.device, dtype=self.dtype_float)
    scale_preds = [[] for _ in range(bsz)]

    for sidx in range(self.max_series_blocks):

      # Generate rows starting with zeros
      hidden_state = self.decoder.dec_tf_row(
        inputs_embeds=scale_embd,
        encoder_hidden_states=dec_hidden_row
      ).last_hidden_state

      # Collect predictions from multi-head group
      scale_pred, _, _ = self.decoder.decode_scale(hidden_state, chart_type_dict)

      #Collect only the last prediction of each output
      for bidx, p in enumerate(scale_pred):
        scale_preds[bidx].append(p[-1:, :])

      #Expand scale embd by one 
      new_embd   = torch.zeros([bsz, 1, self.emb_dim1], device=self.device, dtype=self.dtype_float)
      scale_embd = torch.cat([scale_embd, new_embd], dim=1)

      #Encode scale_pred and assign new values to embedding
      for head_name, ind in chart_type_dict.items():
        scale_x = torch.stack([v[-1] for bidx, v in enumerate(scale_preds) if bidx in ind], dim=0) #[:, -1:, :]
        scale_enc   = self.encoder.cont_encoder['scale'][head_name](scale_x)
        pos_emb      = self.encoder.pos_emb_scale[:,sidx:sidx + 1,:] if self.encoder.pos_emb_scale is not None else 0.0
        scale_enc    = scale_enc + pos_emb
        scale_embd[ind,-1:,:] = scale_enc
    
    #Stack each column
    for bidx in range(bsz):
      scale_preds[bidx] = torch.cat(scale_preds[bidx], dim=0)

    return scale_pred, hidden_state

  def generate_continuous(self, dec_hidden_col, chart_type_dict):
    #1 .Create hidden column state by expanding the continuous  embedding one at a time

    bsz = dec_hidden_col.size(0)

    hidden_col = []
    cont_embds = []
    cont_preds = [[] for _ in range(bsz)]
    for sidx in range(self.max_series_blocks):
      cont_embd = torch.zeros([bsz, 1, self.emb_dim1], device=self.device, dtype=self.dtype_float)
      cont_col_preds = [[] for _ in range(bsz)]
      for pidx in range(self.max_cont_blocks):
        h = self.decoder.dec_tf_col(
            inputs_embeds=cont_embd,
            encoder_hidden_states=dec_hidden_col
          ).last_hidden_state
        
        #Expand continuous embding by one 
        new_embd  = torch.zeros([bsz, 1, self.emb_dim1], device=self.device, dtype=self.dtype_float)
        cont_embd = torch.cat([cont_embd, new_embd], dim=1)
        
        for head_name, ind in chart_type_dict.items():
          hid_col = h[ind, :]
          cont_logits = self.decoder.cont_decoder['continuous'][head_name](hid_col)[:, -1:, :]
          
          cont_pred   = torch.sigmoid(cont_logits)

          cont_enc    = self.encoder.cont_encoder['continuous'][head_name](cont_pred)
          pos_emb     = self.encoder.pos_emb_cont[:,sidx,pidx:pidx+1,:] if self.encoder.pos_emb_cont is not None else 0.0
          
          cont_embd[ind,-1:, :] = cont_enc + pos_emb

          #Collect predictions
          for idx, pred in zip(ind, cont_pred):
            cont_col_preds[idx].append(pred)
        
      #Append only the final one
      hidden_col += [h]
      cont_embds += [cont_embd]

      #Stack each column
      for idx, preds in enumerate(cont_col_preds):
        p = torch.stack(preds, dim=1)
        cont_preds[idx].append(p)

    for bidx in range(bsz):
      cont_preds[bidx] = torch.cat(cont_preds[bidx], dim=0)
    
    hidden_col = torch.stack(hidden_col, dim=1)
    return cont_preds, hidden_col
