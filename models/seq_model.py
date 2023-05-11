# ---------------------------------------------------------------
# Copyright (c) Cyber Security Research Centre Limited 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import torch
import torch.nn as nn

from transformers import (
  BigBirdPegasusForConditionalGeneration,
  BigBirdPegasusConfig,
  PegasusTokenizer
)


def init_seq_model(cfg, device_id, load_opt=True):

    sep_token = cfg.data.dataset.chart_data.sep_token
    codebook_size = max(
        cfg.model.continuous_data.vq.n_emb1, 
        cfg.model.continuous_data.vq.n_emb2
        )

    model_cfg = cfg.model.caption.hf_model
    hf_config = CustomAutoConfig.from_pretrained(model_cfg.name, cache_dir=cfg.cache_dir, codebook_size=codebook_size)
    tokenizer = PegasusTokenizer.from_pretrained(model_cfg.name, cache_dir=cfg.cache_dir)
    num_added_toks = tokenizer.add_tokens([sep_token], special_tokens=True)
    
    model = CustomAutoModelForSeq2SeqLM.from_pretrained(
            model_cfg.name,
            from_tf=False, 
            config=hf_config, 
            cache_dir=cfg.cache_dir
            )

    model.resize_token_embeddings(len(tokenizer))

    if cfg.rank == 0:
        print("HoneyChart Transformer | backbone={} total tokens={} ".format(
        model_cfg.name,  len(tokenizer)))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    opt = None
    if load_opt:
        if device_id is not 'cpu':
            model.cuda(device_id)

        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        lr = cfg.train.optim.learning_rate
        betas = cfg.train.optim.betas
        if cfg.train.optim.type == 'AdamW':
            opt = torch.optim.AdamW(params, lr=lr, betas=betas)
        elif cfg.train.optim.type == 'Adam':
            opt = torch.optim.Adam(params, lr=lr, betas=betas)
        else:
            raise NotImplementedError()

    return model, tokenizer, opt

class CustomAutoConfig(BigBirdPegasusConfig):
    def __init__(
        self,
        codebook_size: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size

class CustomAutoModelForSeq2SeqLM(BigBirdPegasusForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        
        # add a second lm_head
        self.lm_head2 = nn.Linear(config.d_model, config.codebook_size)

        # set the number of output heads to 2
        self.num_labels = 2

        self.loss_fn = nn.CrossEntropyLoss()
        self.output_mode = 'both'
    
    def set_output(self, mode):
        assert mode in ['both', 'text', 'data']
        self.output_mode = mode

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        return_dict=None,
        labels=None,
        return_codes_only=False,
        return_text_only=False,
        return_both=False,
        **kwargs
    ):
        # call the parent class's forward method
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            labels=labels,
        )
        
        #k=['loss', 'logits', 'decoder_hidden_states', 'encoder_last_hidden_state', 'encoder_hidden_states'])
        # compute logits for the second lm_head
        code_logits = self.lm_head2(outputs.decoder_hidden_states[1])

        if self.output_mode == 'both':
            return (outputs.logits, code_logits,) 
        elif self.output_mode == 'text':
            return outputs
        else:
            outputs.logits = code_logits
            return outputs
