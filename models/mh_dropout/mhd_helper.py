# ---------------------------------------------------------------
# Copyright (c) Cybersecurity Cooperative Research Centre 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import torch
import torch.nn as nn


def get_reduce_fn(name):
    if name == 'mean':
        fn = torch.mean
    elif name == 'sum':
        fn = torch.sum
    else:
        raise
    return fn

def get_act_fn(name):
    if name == 'relu':
        fn = nn.ReLU
    elif name == 'gelu':
        fn = nn.GELU
    elif name == 'prelu':
        fn = nn.PReLU
    elif name == 'tanh':
        fn = nn.Tanh
    elif name == 'leakyrelu':
        fn = nn.LeakyReLU
    else:
        raise

    return fn

def get_dist_loss(name):
    if name == 'mse':
        fn = nn.MSELoss
    elif name == 'smoothl1':
        fn = nn.SmoothL1Loss
    else:
        raise
    return fn

class ResBlock2d(nn.Module):
    def __init__(self, in_channel, channel, kernels, act='relu'):
        super().__init__()

        self.conv = nn.Sequential(
            get_act_fn(act)(),
            nn.Conv2d(in_channel, channel, kernels[0], padding=1),
            nn.BatchNorm2d(channel),
            get_act_fn(act)(),
            nn.Conv2d(channel, in_channel, kernels[1]),
            nn.BatchNorm2d(in_channel),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x

        return out

class MLP2d(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, act, kernels=[3,1], padding=1, res=False, n_res_block=0, 
        res_kernels=[3,1], n_res_channel=64, **kwargs):
        super().__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.res = res

        if self.inp_dim != self.out_dim and self.res:
            self.pre = nn.Conv2d(inp_dim, out_dim, 1)
        
        self.f = nn.Sequential(
            nn.Conv2d(inp_dim, hid_dim, kernels[0], padding=padding),
            get_act_fn(act)(),
            nn.Conv2d(hid_dim, out_dim, kernels[1])
        )
        
        self.n_res_block = n_res_block

        blocks = []
        for _ in range(n_res_block):
            blocks.append(ResBlock2d(out_dim, n_res_channel, res_kernels, act=act))
        
        if n_res_block > 0:
            blocks.append(get_act_fn(act)())
            blocks.append(nn.Conv2d(out_dim, out_dim, 1))
            self.blocks = nn.Sequential(*blocks)

    def forward(self, x):

        out = self.f(x)
        if self.res:
            if self.inp_dim != self.out_dim:
                x = self.pre(x)
            out += x

        if self.n_res_block > 0:
            out = self.blocks(out)
            
        return out 


class ResBlock1d(nn.Module):
    def __init__(self, in_channel, channel, kernels, act='relu'):
        super().__init__()

        self.conv = nn.Sequential(
            get_act_fn(act)(),
            nn.Conv1d(in_channel, channel, kernels[0], padding=1),
            nn.BatchNorm1d(channel),
            get_act_fn(act)(),
            nn.Conv1d(channel, in_channel, kernels[1]),
            nn.BatchNorm1d(in_channel),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        return out

class MLP1d(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, act, res=False, kernels=[3,1], padding=1, n_res_block=0, 
        res_kernels=[3,1], n_res_channel=64, **kwargs):
        super().__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.res = res

        if self.inp_dim != self.out_dim and self.res:
            self.pre = nn.Linear(inp_dim, out_dim)

        self.f = nn.Sequential(
            nn.Conv1d(inp_dim, hid_dim, kernels[0], padding=padding),
            get_act_fn(act)(),
            nn.Conv1d(hid_dim, out_dim, kernels[1])
        )
        
        self.n_res_block = n_res_block

        blocks = []
        for _ in range(n_res_block):
            blocks.append(ResBlock1d(out_dim, n_res_channel, res_kernels, act=act))
        
        if n_res_block > 0:
            blocks.append(get_act_fn(act)())
            blocks.append(nn.Conv1d(out_dim, out_dim, 1))
            self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.f(x)
        if self.res:
            if self.inp_dim != self.out_dim:
                x = self.pre(x)
            out += x

        if self.n_res_block > 0:
            out = self.blocks(out)
            
        return out 
