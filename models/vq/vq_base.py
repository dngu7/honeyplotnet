# ---------------------------------------------------------------
# Copyright (c) ___________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils import dist_adapter as dist

class VectorQuantizer(torch.nn.Module):
    def __init__(self, n_emb, emb_dim, beta=0.25, tiled=True, ema_update=True, random_restart=True, threshold=1.0, **kwargs):
        super().__init__()
        self.name = 'vq'
        self.chn_dim = 1
        self.emb_dim = emb_dim
        self.n_emb = n_emb
        self.beta = beta
        self.dtype_float = torch.float32
        self.threshold = threshold
        self.tiled = tiled
        self.ema_update = ema_update
        self.random_restart = random_restart
        
        self.init = False
        
        # init function called during first pass.
        self.k_sum = None
        self.k_elem = None
        self.register_buffer('embeddings', torch.zeros(self.n_emb, self.emb_dim))

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_emb:
            n_repeats = (self.n_emb + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x
    
    def init_codebook(self, x):
        self.init = True

        if self.embeddings.sum() == 0:
            if self.tiled:
                y = self._tile(x)
                embeds = y[torch.randperm(y.shape[0])][:self.n_emb]
            else:
                embeds = torch.nn.Embedding(self.n_emb, self.emb_dim).weight.to(x.device)
                torch.nn.init.uniform_(embeds)

            dist.broadcast(embeds, 0)
            assert embeds.shape == (self.n_emb, self.emb_dim)
            self.embeddings = embeds

        self.k_sum = self.embeddings.clone()
        self.k_elem = torch.ones(self.n_emb, device=x.device)

    def restore_k(self, num_tokens=None, threshold=1.0):
        emb_width, k_bins = self.emb_width, self.k_bins
        self.init = True
        assert self.embeddings.shape == (k_bins, emb_width)
        self.k_sum = self.embeddings.clone()
        self.k_elem = torch.ones(k_bins, device=self.k.device)
        if num_tokens is not None:
            expected_usage = num_tokens / k_bins
            self.k_elem.data.mul_(expected_usage)
            self.k_sum.data.mul_(expected_usage)
        self.threshold = threshold
        
    def forward(self, x, is_indices=False, **kwargs):
        if is_indices:
            return self.sample_decoder(x)
        else:
            return self._forward(x)

    def sample_decoder(self, encoding_indices):
        bs, x_dim = encoding_indices.shape
        output_shape = [bs, x_dim, self.emb_dim]
        
        flattened = torch.reshape(encoding_indices, [bs, -1])
        quantized = F.embedding(flattened, self.embeddings)
        
        quantized = torch.reshape(quantized, output_shape)
        return quantized

    def preprocess(self, x):
        x = x.transpose(self.chn_dim, -1)
        x = torch.reshape(x, [-1, self.emb_dim])
        if x.shape[-1] == self.emb_dim:
            prenorm = torch.norm(x - torch.mean(x)) / np.sqrt(np.prod(x.shape))
        elif x.shape[-1] == 2 * self.emb_dim:
            x1, x2 = x[...,:self.emb_dim], x[...,self.emb_dim:]
            prenorm = (torch.norm(x1 - torch.mean(x1)) / np.sqrt(np.prod(x1.shape))) + (torch.norm(x2 - torch.mean(x2)) / np.sqrt(np.prod(x2.shape)))

            # Normalise
            x = x1 + x2
        else:
            assert False, f"Expected {x.shape[-1]} to be (1 or 2) * {self.emb_dim}"    

        return x, prenorm    

    def update_codebook(self, x, x_l):
        # Updates centres w random restart and computes usage metrics
        #x_l: encoding indices

        with torch.no_grad():
            # Calculate new centres
            x_l_onehot = torch.zeros(self.n_emb, x.shape[0], device=x_l.device)  # k_bins, N * L
            x_l_onehot.scatter_(0, x_l.view(1, x.shape[0]), 1)
            
            _k_sum = torch.matmul(x_l_onehot, x)  # k_bins, w
            _k_elem = x_l_onehot.sum(dim=-1)  # k_bins
            
            y = self._tile(x)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_emb]

            dist.broadcast(_k_rand, 0)
            dist.all_reduce(_k_sum)
            dist.all_reduce(_k_elem)

            #Update centre
            old_k = self.embeddings
            self.k_sum = self.beta * self.k_sum + (1. - self.beta) * _k_sum  # w, k_bins
            self.k_elem = self.beta * self.k_elem + (1. - self.beta) * _k_elem  # k_bins
            usage = (self.k_elem.view(self.n_emb, 1) >= self.threshold).float()

            #new_k = old_k.clone()
            new_k = (self.k_sum.view(self.n_emb, self.emb_dim) / self.k_elem.view(self.n_emb, 1))
                
            if self.random_restart:
                new_k = usage * new_k + (1-usage) * _k_rand
            
            self.embeddings = new_k
            
            #Compute metrics
            _k_prob = _k_elem / torch.sum(_k_elem)
            entropy = -torch.sum(_k_prob * torch.log(_k_prob + 1e-8)) #How many being used
            used_curr = (_k_elem >= self.threshold).sum() #How many of them being used (raw values)
            usage = torch.sum(usage)
            dk = torch.norm(self.embeddings - old_k) / np.sqrt(np.prod(old_k.shape))
            dk = torch.nan_to_num(dk)

        return dict(entropy=entropy,
                    used_curr=used_curr,
                    usage=usage,
                    dk=dk)

    def _forward(self, x, update_k=True):

        x_shape = list(x.shape)
        x_shape[-1], x_shape[self.chn_dim] = x_shape[self.chn_dim], x_shape[-1]
        
        flat_x, prenorm = self.preprocess(x)

        if update_k and not self.init:
            self.init_codebook(flat_x)

        encoding_indices, fit = self.get_code_indices(flat_x)
        code_metrics = self.update_codebook(flat_x, encoding_indices)

        quantized = F.embedding(encoding_indices, self.embeddings)
        
        quantized = torch.reshape(quantized, x_shape)
        quantized = quantized.transpose(self.chn_dim, -1)

        commit_loss = self.beta * torch.mean(
            (quantized.detach() - x) ** 2
        )

        # Vanilla Codebook Loss.
        codebook_loss = torch.mean((quantized - x.detach()) ** 2) if not self.ema_update else 0

        loss = commit_loss + codebook_loss

        # Straight-through estimator.
        out = x + (quantized - x).detach()
        
        return out, quantized, loss, dict(fit=fit, prenorm=prenorm, **code_metrics)

    def get_code_indices(self, x):

        #Check if flat
        if len(list(x.shape)) >= 3:
            x, _ = self.preprocess(x)

        if not self.init:
            self.init_codebook(x)

        similarity = torch.matmul(x, self.embeddings.t())

        s1 = torch.sum(x ** 2, axis=1, keepdims=True)
        s2 = torch.sum(self.embeddings.t() ** 2, axis=0)
        s3 = - 2 * similarity

        distances = s1 +s2 + s3

        # Derive the indices for minimum distances.
        min_distance, encoding_indices = torch.min(distances, axis=1)
        fit = torch.mean(torch.nan_to_num(min_distance))
        return encoding_indices, fit
    
    def get_embed(self, **kwargs):
        return self.embeddings
    
    def _get_code_indices(self, x):
        x_shape = list(x.shape)
        x_shape[-1], x_shape[self.chn_dim] = x_shape[self.chn_dim], x_shape[-1]
        
        flat_x, _ = self.preprocess(x)
        encoding_indices, _ = self.get_code_indices(flat_x)
        return encoding_indices


class VQMulti(torch.nn.Module):
    def __init__(self, n_embs, emb_dims, betas, levels, tiled, random_restart, ema_update, **kwargs):
        super().__init__()
        self.levels = levels
        self.emb_dims = emb_dims
        self.level_blocks = torch.nn.ModuleList()
        for level in range(self.levels):
            self.level_blocks.append(
                VectorQuantizer(n_embs[level], emb_dims[level], beta=betas[level], 
                                tiled=tiled, random_restart=random_restart, ema_update=ema_update))
    
    def forward(self, xs):
        zs, xs_quantised, commit_losses, metrics = [], [], [], []

        for level in range(self.levels):
            level_block = self.level_blocks[level]
            x = xs[level]
            z, x_quantised, commit_loss, metric = level_block(x)
            zs.append(z)
            if not self.training:
                # Be extra paranoid and make sure the encoder weights can't
                # change from straight-through estimator
                x_quantised = x_quantised.detach()
            xs_quantised.append(x_quantised)
            commit_losses.append(commit_loss)
            if self.training:
                metrics.append(metric)
        
        return zs, xs_quantised, commit_losses, metrics

    def get_code_indices(self, xs):
        code_indices = []
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            x = xs[level]
            codes = level_block._get_code_indices(x)
            code_indices.append(codes)
        return codes

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        xs_quantised = [level_block.sample_decoder(z) for (level_block, z) in zip(self.level_blocks[start_level:end_level], zs)]
        return xs_quantised