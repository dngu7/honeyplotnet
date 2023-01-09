# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------


import torch
import torch.nn as nn

def get_act_fn(name):
    if name == 'relu':
        fn = torch.nn.ReLU
    elif name == 'gelu':
        fn = torch.nn.GELU
    elif name == 'prelu':
        fn = torch.nn.PReLU
    elif name == 'tanh':
        fn = torch.nn.Tanh
    elif name == 'leakyrelu':
        fn = torch.nn.LeakyReLU
    else:
        raise
    return fn

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, kernels, padding=0, act='relu'):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, channel, kernels[0], padding=padding),
            nn.BatchNorm1d(channel),
            get_act_fn(act)(),
            nn.Conv1d(channel, in_channel, kernels[1]),
            nn.BatchNorm1d(in_channel),
            get_act_fn(act)(),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out + x
        return out

class Conv1dEncoder(nn.Module):
    def __init__(self, channels, kernels, stride, padding, 
        n_res_block, n_res_channel, res_kernels, res_padding, 
        act='leakyrelu', use_bn=False, use_proj=False):
        super().__init__()
        assert isinstance(channels, list), "provide list for channels"
        assert len(res_kernels) == 2

        self.nblocks = len(channels)
        blocks = []
        prev_chn = channels[0]
        kidx = 0
        if use_proj:
            out_channel    = channels[-1]
            block_channels = channels[1:-1]
        else:
            block_channels = channels[1:]
        for chn in block_channels:
            new_block = [nn.Conv1d(
                prev_chn, chn, kernel_size=kernels[kidx], stride=stride, padding=padding)]
            
            if use_bn:
                new_block += [nn.BatchNorm1d(chn)]

            new_block += [get_act_fn(act)()]
            blocks += new_block
            kidx += 1

            prev_chn = chn

        for _ in range(n_res_block):
            blocks.append(ResBlock(prev_chn, n_res_channel, res_kernels, padding=res_padding, act=act))
        
        if use_proj:
            blocks.append(nn.Conv1d(prev_chn, out_channel, 1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Conv1dDecoder(nn.Module):
    def __init__(self, channels, kernels, stride, padding, 
        n_res_block, n_res_channel, res_kernels, res_padding, 
        act='relu', use_bn=False, use_proj=False):
        super().__init__()
        assert isinstance(channels, list), "provide list for channels"
        assert len(res_kernels) == 2

        self.nblocks = len(channels)
        prev_chn = channels[0]
        blocks = []
        kidx = 0

        for _ in range(n_res_block):
            blocks.append(ResBlock(prev_chn, n_res_channel, res_kernels, padding=res_padding, act=act))
        
        if use_proj:
            out_channel    = channels[-1]
            block_channels = channels[1:-1]
        else:
            block_channels = channels[1:]

        for chn in block_channels:
            new_block = [nn.ConvTranspose1d(
                prev_chn, chn, kernel_size=kernels[kidx], stride=stride, padding=padding)]
            
            if use_bn:
                new_block += [nn.BatchNorm1d(chn)]

            new_block += [get_act_fn(act)()]
            blocks += new_block
            kidx += 1

            prev_chn = chn

        if use_proj:
            blocks.append(nn.Conv1d(prev_chn, out_channel, 1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)