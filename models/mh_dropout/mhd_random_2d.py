# ---------------------------------------------------------------
# Copyright (c) ___________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mhd_helper import (
    get_reduce_fn, get_dist_loss, MLP2d
)

class MHDropoutNetRandom2D(nn.Module):
    def __init__(self,
        inp_dim, 
        hidden_dim, 
        out_dim, 
        decoder_cfg,
        dist_reduce='mean', 
        loss_reduce='mean', 
        loss_reduce_dims=[-3,-2,-1], 
        hypothese_dim=1, 
        dist_loss='mse', 
        gamma=0.25, 
        dropout_rate=0.5, 
        topk=64, 
        bottleneck=False,
        norm=False,
        **kwargs):

        super().__init__()
        self.dtype_float = torch.float32 
        self.topk = topk
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hypothese_dim = hypothese_dim
        self.dropout_rate = dropout_rate
        self.gamma = gamma
        self.dist_loss = get_dist_loss(dist_loss)(reduction='none') 
        self.loss_reduce_dims = loss_reduce_dims
        self.loss_reduce = get_reduce_fn(loss_reduce)
        self.dist_reduce = get_reduce_fn(dist_reduce)
        self.bottleneck = bottleneck

        self.norm_1 = nn.GroupNorm(1, inp_dim) if norm and inp_dim > 1 else None

        if self.bottleneck: 
            self.proj = MLP2d(inp_dim, hidden_dim, hidden_dim, act='relu') 
            dec_inp_dim = hidden_dim
        else:
            dec_inp_dim = inp_dim

        self.decoder = MLP2d(dec_inp_dim, hidden_dim, out_dim, **decoder_cfg)
        
    def forward(self, x, y=None, wta_idx=None, wta_loss=None, hypothese_count=1, keep_idx=None, **kwargs):
        
        hypotheses = self.sample(x, hypothese_count=hypothese_count, keep_idx=keep_idx)
        
        if (y is None and wta_idx is None and wta_loss is None):
            return hypotheses
        
        if y is not None:
            wta_loss, topk_idx  = self.get_dist_loss(hypotheses, y)
            wta_idx = topk_idx[:,0]
   
        win_hypo = self.get_win_hypo(hypotheses, wta_idx)
            
        return win_hypo, wta_loss.mean(-1)
    
    def get_win_hypo(self, hypotheses, wta_idx):
        batch_list = torch.arange(hypotheses.size(0))
        winner_hypo = hypotheses[batch_list, wta_idx, :]
        return winner_hypo

    def get_topk_hypo(self, hypotheses, topk_idx):
        topk_hypos = [batch_hypo[idx, :] for batch_hypo, idx in zip(hypotheses, topk_idx)]
        topk_hypos = torch.stack(topk_hypos, dim=0)
        return topk_hypos

    def create_mask(self, x, keep_idx=None):
        
        m_shape = x.shape[:2]
        if keep_idx is None:
            m_prob = torch.ones(m_shape, device=x.device, requires_grad=x.requires_grad, dtype=self.dtype_float) * (1. - self.dropout_rate)
            m = torch.bernoulli(m_prob).unsqueeze(-1).unsqueeze(-1)
        else:
            assert isinstance(keep_idx, int), "keep_idx must be integer"
            m_prob = torch.zeros(m_shape, device=x.device, requires_grad=x.requires_grad, dtype=self.dtype_float)
            m_prob[:,keep_idx] = 1.0
            m = m_prob.unsqueeze(-1).unsqueeze(-1)

        return m

    def sample(self, x, hypothese_count, keep_idx=None):
        
        #Infer the repeat structure
        x_shape = list(x.shape)
        bs, x_dim, y_dim = x_shape[0], x_shape[-1], x_shape[-2]
        repeat_frame = [1] * (len(x_shape) + 1)
        repeat_frame[self.hypothese_dim] = hypothese_count

        #Optional: Apply normalization
        if self.norm_1 is not None:
            x = self.norm_1(x)

        #Repeat for single forward pass sampling
        x_repeat = x.unsqueeze(self.hypothese_dim).repeat(repeat_frame)
        
        #Flatten along batch and sample axis
        x_repeat = torch.flatten(x_repeat, start_dim=0, end_dim=1)

        #Optional: Create bottleneck to reduce space of all possible masks
        if self.bottleneck: 
            hidden = self.proj(x_repeat)
        else:
            hidden = x_repeat

        # Apply dropout here
        distd_hidden = hidden * self.create_mask(hidden, keep_idx)

        # [bs * hypothese_count, out_dim, x, y]
        output = self.decoder(distd_hidden)
        
        output = output.reshape((bs, hypothese_count, self.out_dim, x_dim, y_dim))

        return output

    def get_dist_loss(self, hypotheses, y, topk=None):

        hypothese_count = hypotheses.size(self.hypothese_dim)

        y_shape = [1] * len(hypotheses.shape)
        y_shape[self.hypothese_dim] = hypothese_count

        #Create copies of y to match sample length
        y_expanded = y.unsqueeze(self.hypothese_dim).repeat(y_shape)
        
        dist_loss = self.dist_loss(hypotheses, y_expanded)
        dist_loss = self.loss_reduce(dist_loss, dim=self.loss_reduce_dims)

        #Get the sample with the lowest meta loss
        if topk is None:
            topk = min(self.topk, dist_loss.size(1)) 
        
        topk_idx = torch.topk(dist_loss, dim=-1, largest=False, sorted=True, k=topk)[1]
        
        return dist_loss, topk_idx
        
    def get_wta_loss(self, dist_loss, topk_idx, **kwargs):
        wta_idx = topk_idx[:,0]
        #Create a mask based on the sample count
        wta_mask = F.one_hot(wta_idx, dist_loss.size(-1))
        
        #Get Winner-Takes-All Loss by using the mask
        wta_loss = (dist_loss * wta_mask).sum(-1)
        return wta_loss, wta_idx

        