# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------


import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.continuous.encoder import Encoder
from models.conv import Conv1dEncoder 
from models.gpt import GPTNoEmbed

from .fid_score import load_statistics
from .fid_utils import download_pretrained, load_fid_snapshot

def init_fid_model(cfg, load_path=None, device_id=0):

    cd_cfg        = cfg.model.continuous_data
    fid_cfg       = cd_cfg.fid
    tf_cfg        = fid_cfg.transformer
    max_blocks    = cd_cfg.encoder.max_blocks
    enc_out_dim   = fid_cfg.enc_out_dim

    enc_conv_kwargs = fid_cfg.conv
    enc_conv_kwargs['channels'] = [max_blocks.points] + enc_conv_kwargs['channels'] 

    enc_conv = Conv1dEncoder(**enc_conv_kwargs)
    
    enc_tf = GPTNoEmbed(
      block_size=max_blocks.points + 1,
      n_layer=tf_cfg.n_layer,
      n_head=tf_cfg.n_head,
      n_embd=enc_out_dim,
    )

    kwargs = {
    'enc_conv': enc_conv,
    'enc_tf': enc_tf,
    'out_dim': enc_out_dim,
    'max_series_blocks': max_blocks.series,
    'max_cont_blocks': max_blocks.points,
    'norm_mode': cfg.data.dataset.chart_data.norm_mode,
    'use_pos_embs': fid_cfg.use_pos_embs,
    'device': f'cuda:{device_id}' if device_id != 'cpu' else 'cpu',
    'debug': cfg.debug,
    }

    model = Classifier(**kwargs)

    #Create temp folder and load checkpoint
    if load_path is None:
        load_path = './.temp'
        os.makedirs(load_path, exist_ok=True)
    
    #Download pretrained and 
    download_pretrained(path=load_path)

    #Load the model with weights.
    snapshot_path = os.path.join(load_path, 'chartFid_snapshot.pth')
    load_fid_snapshot(model, snapshot_path)

    #Get fid statistics
    fid_stats = {}
    fid_stats['train'] = load_statistics(os.path.join(load_path, 'pmc-train.npz'))
    fid_stats['test'] = load_statistics(os.path.join(load_path, 'pmc-test.npz'))

    return model, fid_stats

class Classifier(nn.Module):
    def __init__(self, 
        enc_conv, enc_tf, out_dim=8, use_pos_embs=True,
        max_series_blocks=5, max_cont_blocks=32, num_classes=1, 
        norm_mode='minmax', device='cpu', debug=False):
        super().__init__()
        
        self.debug        = debug
        self.device       = device 
        self.dtype_float  = torch.float32 
        self.dtype_int    = torch.int32 

        self.norm_mode    = norm_mode

        self.discriminator = Encoder(
            enc_conv=enc_conv,
            enc_tf=enc_tf,
            out_dim=out_dim,
            max_series_blocks=max_series_blocks,
            max_cont_blocks=max_cont_blocks,
            use_pos_embs=use_pos_embs,
            norm_mode=norm_mode,
            chart_type_conditional=False,
            device=device,
            debug=debug
            )
        
        self.pre_layer  = nn.Linear(out_dim, out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)
        self.loss_fn = nn.BCELoss()

    def forward(self, inputs, labels=None):
        out, zero_loss = self.discriminator(inputs)

        out = out.mean(1) 

        act = self.pre_layer(out)
        logits = self.classifier(F.relu(act)).view(-1)
        logits = logits.sigmoid() 
        
        #x = inputs['chart_data'] if 'chart_data' in inputs else inputs
        loss_dict, metric_log = {}, {}
        if labels is not None:

            loss = self.get_loss(logits, labels) + zero_loss
            loss_dict = {'nll': loss}

            #Metrics (accuracy)
            pred = torch.where(logits > 0.5, 1, 0)
            acc = (pred == labels).to(torch.int32).detach().cpu().tolist()
            metric_log = {'acc': acc, 'nll': loss.detach().cpu().item()}
        
        activations = act.detach().cpu().numpy()
        return activations, loss_dict, metric_log

    def get_loss(self, pred, labels):
        return self.loss_fn(pred, labels)


