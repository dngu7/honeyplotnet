# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F

from .gpt import GPTConfig, Block


class GPTLight(nn.Module):
    def __init__(
        self,
        n_class,
        block_size,
        n_layer=12,
        n_head=8,
        n_embd=256,
        embd_pdrop=0.1,  
        use_fp16=False
    ):
        super().__init__()

        config = GPTConfig(vocab_size=1, block_size=block_size,
                           embd_pdrop=embd_pdrop, 
                           n_layer=n_layer, n_head=n_head,
                            n_embd=n_embd)

        self.config = config
        self.dtype_float = torch.float32 #torch.float16 if use_fp16 else torch.float32

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.block_size = config.block_size
        self.out_head = nn.Linear(config.n_embd, n_class, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, embeddings, attn_mask=None, condition=None):
        # forward the GPT model
        t = embeddings.size(1)

        if condition is not None:
            t += condition.size(1)
            embeddings = torch.cat((embeddings, condition), dim=1)

        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        attn_mask = self.prep_atten_mask(attn_mask)

        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector

        x = self.drop(embeddings + position_embeddings)
        for idx, block in enumerate(self.blocks):
            if idx == 0:
                x = block(x, attn_mask=attn_mask)
            else:
                x = block(x)

        x = self.ln_f(x)
        out = self.out_head(x)

        return out

    def prep_atten_mask(self, attn_mask):
        if attn_mask is not None:
            batch_size = attn_mask.size(0)
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attn_mask = attn_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attn_mask = attn_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attn_mask = attn_mask.to(dtype=self.dtype_float)  # fp16 compatibility
            attn_mask = (1.0 - attn_mask) * -10000.0
            
        return attn_mask


class CondGPT(nn.Module):
    def __init__(self, shared, device_id, use_fp16, use_cond, emb_len1, emb_len2,
            n_classes, block_sizes, n_layers, n_heads, n_embds, pkeeps=[1.0,1.0]):

        super().__init__()
        self.shared = shared #Shared token embeddings
        self.cond_embd = nn.Embedding(n_classes[0], n_embds[0])
        self.use_cond = use_cond

        self.device_id = device_id
        self.emb_len1 = emb_len1
        self.emb_len2 = emb_len2
        self.dtype_float = torch.float32 #torch.float16 if use_fp16 else torch.float32

        self.model = GPTLight(
            use_fp16=use_fp16,
            n_class=n_classes[0],
            block_size=block_sizes[0],
            n_layer=n_layers[0],
            n_head=n_heads[0],
            n_embd=n_embds[0],
            embd_pdrop=pkeeps[0],
            )

        if use_cond:
            self.cond_model=GPTLight(
                use_fp16=use_fp16,
                n_class=n_classes[1],
                block_size=block_sizes[1],
                n_layer=n_layers[1],
                n_head=n_heads[1],
                n_embd=n_embds[1],
                embd_pdrop=pkeeps[1],
                )


        self.ct_model=GPTLight(
            n_class=3,
            use_fp16=use_fp16,
            block_size=block_sizes[1] if len(block_sizes) == 2 else block_sizes[0],
            n_layer=2,
            n_head=2,
            n_embd=n_embds[1] if len(n_embds) == 2 else n_embds[0],
            embd_pdrop=pkeeps[1] if len(pkeeps) == 2 else pkeeps[0],
            )        

    def forward(self, context_idx, attn_mask=None, 
        tgt1=None, tgt2=None, tgt3=None, temp=1.0, split='train'):
        bsz = context_idx.size(0)
        
        context_idx = context_idx.to(f'cuda:{self.device_id}')
        if attn_mask is not None:
            attn_mask1 = attn_mask.to(f'cuda:{self.device_id}')
            attn_mask2 = torch.ones([bsz, self.emb_len1], device=f'cuda:{self.device_id}', dtype=self.dtype_float)
            attn_mask2 = torch.cat([attn_mask1, attn_mask2], dim=1)

        x = self.shared(context_idx) # each index maps to a (learnable) vector

        logits1 = self.model(x, attn_mask=attn_mask1)
        
        logits1 = logits1[:,:self.emb_len1,:]
        
        #probs1 = F.softmax(logits1 / temp, dim=-1).data
        #probs1 = torch.flatten(probs1, start_dim=0, end_dim=1)
        cb1_idx =  self.get_prediction(logits1, temp)
        cb1_idx = cb1_idx.reshape([bsz, self.emb_len1])
        cb1_embd = self.cond_embd(cb1_idx)

        logits2 = None
        if self.use_cond:
            logits2 = self.cond_model(x, attn_mask=attn_mask2, condition=cb1_embd)
            logits2 = logits2[:,:self.emb_len2,:]

        logits3 = self.ct_model(x, attn_mask=attn_mask2, condition=cb1_embd)
        logits3 = logits3[:,:1,:]

        loss_dict = {}
        if tgt1 is not None:
            loss_dict['cb1'] = self.get_loss(logits1, tgt1)
        if tgt2 is not None and self.use_cond:
            loss_dict['cb2'] = self.get_loss(logits2, tgt2)
        if tgt3 is not None:
            loss_dict['ct'] = self.get_loss(logits3, tgt3)

        ct_pred = self.get_prediction(logits3, temp)
        ct_pred = ct_pred.reshape([bsz, 1])
        
        return logits1, logits2, ct_pred, loss_dict

    def get_loss(self, logits, tgt):
        flatten_logits = torch.reshape(logits, [-1, logits.size(-1)])
        flatten_target = tgt.view(-1)
        return F.cross_entropy(flatten_logits, flatten_target)

    def get_prediction(self, logits, temp):
        probs = F.softmax(logits / temp, dim=-1).data
        probs = torch.flatten(probs, start_dim=0, end_dim=1)
        return torch.multinomial(probs, 1)