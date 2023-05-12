# ---------------------------------------------------------------
# Copyright (c) ________________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import copy

from transformers import (
  T5Config, BigBirdPegasusConfig, PegasusConfig,
  T5ForConditionalGeneration, BigBirdPegasusForConditionalGeneration, PegasusForConditionalGeneration,
  T5PreTrainedModel, BigBirdPegasusPreTrainedModel, PegasusPreTrainedModel,
  AutoTokenizer

)

from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.pegasus.modeling_pegasus import PegasusDecoder
from transformers.models.bigbird_pegasus.modeling_bigbird_pegasus import BigBirdPegasusDecoder

from models.constant import UNIQ_CHART_HEADS

SUPPORTED_MODELS = [
    'google/t5-v1_1-large',
    'google/pegasus-pubmed',
    'google/bigbird-pegasus-large-pubmed'
]

def init_seq_model(cfg, device_id, load_opt=True):

    sep_token = cfg.data.dataset.chart_data.sep_token
    codebook_size = 2 + len(UNIQ_CHART_HEADS) + \
        max(cfg.model.continuous_data.vq.n_emb1, cfg.model.continuous_data.vq.n_emb2)
    
    code_seq_len = 1 + cfg.model.continuous_data.vq.emb_len1 + cfg.model.continuous_data.vq.emb_len2
    
    decoder2_num_layers = cfg.model.seq.decoder2_num_layers

    #0 reserved for tokenizer.pad_token_id, 1 reserved for tokenizer.eos_token_id
    model_cfg = cfg.model.caption.hf_model
    assert model_cfg.name in SUPPORTED_MODELS, "Unsupported model: {}".format(model_cfg.name)

    cfg_kwargs = {"cache_dir": cfg.cache_dir, 
                  "codebook_size": codebook_size, 
                  "code_seq_len": code_seq_len,
                  "decoder2_num_layers": decoder2_num_layers}
    
    tok_kwargs = {"cache_dir": cfg.cache_dir}

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, **tok_kwargs)
    num_added_toks = tokenizer.add_tokens([sep_token], special_tokens=True)

    assert tokenizer.pad_token_id == 0, tokenizer.pad_token_id
    assert tokenizer.eos_token_id == 1, tokenizer.eos_token_id

    if model_cfg.name == 'google/t5-v1_1-large':
        config_class = DoubleDecoderT5Config
        base_model_class = T5ForConditionalGeneration
        model_class = DoubleDecoderT5
    elif model_cfg.name == 'google/pegasus-pubmed':
        config_class = DoubleDecoderPegasusConfig
        base_model_class = PegasusForConditionalGeneration
        model_class = DoubleDecoderPegasus
    elif model_cfg.name == 'google/bigbird-pegasus-large-pubmed':
        config_class = DoubleDecoderBigBirdConfig
        base_model_class = BigBirdPegasusForConditionalGeneration
        model_class = DoubleDecoderBigBird

    hf_config = config_class.from_pretrained(model_cfg.name, **cfg_kwargs)
    model_kwargs = {"cache_dir": cfg.cache_dir, "config": hf_config, "from_tf": False}
    base_model = base_model_class.from_pretrained(model_cfg.name, **model_kwargs)
    base_model.resize_token_embeddings(len(tokenizer))
    assert base_model.config.decoder_start_token_id is not None, "Make sure that `config.decoder_start_token_id` is correctly defined"
    

    model = model_class(base_model, hf_config)
    base_num_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad) / 1e6
    total_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    if cfg.rank in [0,'cpu']: print("Number of trainable parameters | {} | Base: {:.4f} Total: {:.4f}".format(
            model_cfg.name, base_num_params, total_num_params))

    opt = None
    scheduler = None
    opt_mode = int(cfg.model.seq.opt_mode)
    if load_opt:
        if device_id is not 'cpu':
            model.cuda(device_id)

        #1: Text only, (Freeze decoder2)
        #2: Data only, (Freeze pre-trained model)
        params = []
        assert opt_mode in [0,1,2]
        if opt_mode == 0:
            params = list(filter(lambda p: p.requires_grad, model.parameters()))
        else:
            for name, param in model.named_parameters():
                if (opt_mode == 1 and 'model.' in name) or \
                    (opt_mode == 2 and 'model.' not in name):
                    param.requires_grad = True
                    params.append(param)
                else:
                    param.requires_grad = False

            
        lr = cfg.train.optim.learning_rate
        betas = cfg.train.optim.betas
        if cfg.train.optim.type == 'AdamW':
            opt = torch.optim.AdamW(params, lr=lr, betas=betas)
        elif cfg.train.optim.type == 'Adam':
            opt = torch.optim.Adam(params, lr=lr, betas=betas)
        else:
            raise NotImplementedError()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, verbose=False)

    return model, tokenizer, opt, scheduler

class DoubleDecoderT5Config(T5Config):
    def __init__(
        self,
        codebook_size: int = 128,
        code_seq_len: int = 29,
        decoder2_num_layers: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size 
        self.code_seq_len = code_seq_len
        self.decoder2_num_layers = decoder2_num_layers

class DoubleDecoderPegasusConfig(PegasusConfig):
    def __init__(
        self,
        codebook_size: int = 128,
        code_seq_len: int = 29,
        decoder2_num_layers: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size 
        self.code_seq_len = code_seq_len
        self.decoder2_num_layers = decoder2_num_layers

class DoubleDecoderBigBirdConfig(BigBirdPegasusConfig):
    def __init__(
        self,
        codebook_size: int = 128,
        code_seq_len: int = 29,
        decoder2_num_layers: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size 
        self.code_seq_len = code_seq_len
        self.decoder2_num_layers = decoder2_num_layers

class DoubleDecoderBase(object):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"decoder2\.weight", r"data_head\.weight"]
    def __init__(self, **kwargs):
        pass 

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(self, **kwargs):
        return self.model.prepare_inputs_for_generation(**kwargs)
    
    def prepare_decoder_input_ids_from_labels(self, labels):
        return self.model.prepare_decoder_input_ids_from_labels(labels)

    def set_output(self, mode):
        assert mode in ['both', 'text', 'data']
        self.output_mode = mode
    
    def generate_codes(self, encoder_hidden_states, greedy=False):

        bsz = encoder_hidden_states.shape[0]
        input_ids = torch.zeros([bsz, 1], dtype=torch.long, device=self.device)

        for _ in range(self.config.code_seq_len):
            outputs = self.decoder2(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
            )
            code_logits = self.data_head(outputs[0])

            if greedy:
                tokens = code_logits.argmax(-1)
            else:
                tokens = Categorical(logits=code_logits).sample()
                

            new_input_ids = tokens[:,-1:]
            input_ids = torch.cat([input_ids, new_input_ids], dim=-1)
        
        #remove start of sentence token
        code_tokens = input_ids[:,1:]
        return code_tokens
    
    def generate(self, **kwargs):
        if self.output_mode == 'text':
            input_ids = kwargs.pop('input_ids')
            text_tokens = self.model.generate(input_ids, **kwargs)
            return text_tokens
        
        elif self.output_mode == 'data':
            kwargs['output_hidden_states'] = True
            outputs = self.model.get_encoder()(**kwargs)
            code_tokens = self.generate_codes(outputs.last_hidden_state)
            return code_tokens
        else:
            raise ValueError("Only use text or data as output mode")
        
    def set_output(self, mode):
        assert mode in ['both', 'text', 'data']
        self.output_mode = mode

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder2_input_ids=None,
        decoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        return_dict=None,
        **kwargs
    ):  
        # call the parent class's forward method
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # Pass through second decoder
        dec2_out = self.decoder2(
            input_ids=decoder2_input_ids,
            attention_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=outputs.encoder_last_hidden_state,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        code_logits = self.data_head(dec2_out[0])
        
        if self.output_mode == 'text':
            return outputs
        elif self.output_mode == 'data':
            outputs.logits = code_logits
            return outputs
        elif self.output_mode == 'both':
            return (outputs.logits, code_logits,) 
        raise

class DoubleDecoderT5(DoubleDecoderBase, T5PreTrainedModel):
    def __init__(self, model, config):
        super(T5PreTrainedModel, self).__init__(config)
        self.model = model
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.decoder2_num_layers
        data_shared = nn.Embedding(config.codebook_size, config.d_model)
        self.decoder2 = T5Stack(decoder_config, data_shared)
        self.data_head = nn.Linear(config.d_model, config.codebook_size)
        self.output_mode = 'both'

class DoubleDecoderPegasus(DoubleDecoderBase, PegasusPreTrainedModel):
    def __init__(self, model, config):
        super(PegasusPreTrainedModel, self).__init__(config)
        self.model = model
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.decoder2_num_layers
        data_shared = nn.Embedding(config.codebook_size, config.d_model)
        self.decoder2 = PegasusDecoder(decoder_config, data_shared)
        self.data_head = nn.Linear(config.d_model, config.codebook_size)
        self.output_mode = 'both'

class DoubleDecoderBigBird(DoubleDecoderBase, BigBirdPegasusPreTrainedModel):
    def __init__(self, model, config):
        super(BigBirdPegasusPreTrainedModel, self).__init__(config)
        self.model = model
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.decoder2_num_layers
        data_shared = nn.Embedding(config.codebook_size, config.d_model)
        self.decoder2 = BigBirdPegasusDecoder(decoder_config, data_shared) 
        self.data_head = nn.Linear(config.d_model, config.codebook_size)
        self.output_mode = 'both'
