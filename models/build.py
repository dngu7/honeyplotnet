# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------


import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.utils.data import ElasticDistributedSampler


from .conv import Conv1dEncoder, Conv1dDecoder 
from .continuous import ContinuousModel, Discriminator
from .gpt import GPTNoEmbed
from .mh_dropout import MHDropoutNetRandom1D
from models.vq import VectorQuantizer

from .seq_model import init_seq_model
from dataset import init_dataloader

from transformers import (
  AutoConfig, AutoTokenizer, 
  AutoModelForSeq2SeqLM, T5Config,
)

from transformers.models.t5.modeling_t5 import T5Stack

from torch.utils.data import DataLoader

def init_model(cfg, mode, stage, device_id):
  '''Initialize all models according to config file'''
  
  use_distributed = cfg.torch_dist.use
  active_models   = cfg.model.active

  models, toks, opts, schs, colls, loaders = {}, {}, {}, {}, {}, {}
  train_ds, val_ds = {},{}

  if cfg.eval.ksm.active:
    models['ksm'], toks['ksm'] = init_ksm_model(cfg)

  models['continuous'], opts['continuous'] = init_continuous_model(cfg)
  
  if cfg.model.continuous_data.disc.use:
    models['disc'], opts['disc'] = init_disc_model(cfg, device_id)

  if 'seq' in active_models or stage == 'seq':
    models['seq'], toks['seq'], opts['seq'], schs['seq'] = init_seq_model(
      cfg, cfg.device_id, load_opt=stage=='seq')
    
  #Prepare models for distributed training
  if use_distributed:
    train_ds[stage], loaders['val'], colls[stage] = init_dataloader(
        cfg, mode, stage, models, toks, return_dataset=True)
    
    for s in models.keys():
      models[s] = to_distributed(
          cfg=cfg, 
          model=models[s], 
          device_id=device_id, 
          )

      if cfg.rank == 0: print(f"Creating distributed model: {s}")
      
  elif cfg.device_id not in [None, 'cpu']:
    for s in models.keys():
      if models[s] is not None and device_id != 'cpu':
        models[s].to('cuda:{}'.format(device_id))

  return models, toks, opts, schs, loaders

def to_distributed(cfg, model, device_id):
  if model is None: return None
  model.to(f'cuda:{device_id}')
  model = DDP(model, device_ids=[device_id], find_unused_parameters=cfg.debug) 
  return model 

def init_continuous_model(cfg):

  use_fp16      = cfg.fp16.use
  cd_cfg        = cfg.model.continuous_data
  encoder_cfg   = cd_cfg.encoder
  decoder_cfg   = cd_cfg.decoder
  vq_cfg        = cd_cfg.vq
  mhd_cfg       = cd_cfg.mhd
  
  max_blocks   = encoder_cfg.max_blocks

  #### Encoder
  enc_conv_kwargs = encoder_cfg.conv
  enc_conv_kwargs['channels'] = [max_blocks.points] + enc_conv_kwargs['channels'] 
  last_chn_enc = enc_conv_kwargs['channels'][-1]

  enc_conv = Conv1dEncoder(**enc_conv_kwargs)
  enc_proj1 = nn.Conv1d(last_chn_enc, vq_cfg.emb_len1, 1)
  
  enc_tf = None
  if encoder_cfg.transformer.use:
    enc_tf = init_transformer(encoder_cfg.transformer, 
      block_size=max_blocks.points + 1, emb_dim=vq_cfg.emb_dim1, use_pos_embs=False)
  
  #### Decoder
  # y_hat > dec_conv > dec_tf > proj
  dec_inp_channels = int(cfg.model.continuous_data.decoder.chart_type_conditional) + (vq_cfg.emb_len1)
  dec_conv_kwargs = decoder_cfg.conv
  dec_conv_kwargs['channels'] = [dec_inp_channels] + dec_conv_kwargs['channels']
  last_chn_dec = dec_conv_kwargs['channels'][-1]

  dec_conv = Conv1dDecoder(**dec_conv_kwargs)
  dec_proj_col = nn.Conv1d(last_chn_dec, max_blocks.points, 1)
  dec_proj_row = nn.Conv1d(last_chn_dec, max_blocks.series, 1)
  
  dec_tf_name = decoder_cfg.transformer.name
  dec_tf_col = init_transformer(decoder_cfg.transformer, 
    block_size=max_blocks.points + 1, emb_dim=vq_cfg.emb_dim1)
  dec_tf_row = init_transformer(decoder_cfg.transformer, 
    block_size=max_blocks.series + 1, emb_dim=vq_cfg.emb_dim1)

  vq1_kwargs = {
    'n_emb': vq_cfg.n_emb1,
    'emb_dim': vq_cfg.emb_dim1,
    'beta': vq_cfg.beta,
    'tiled': vq_cfg.tiled,
    'ema_update': vq_cfg.ema_update,
    'random_restart': vq_cfg.random_restart
    }

  vq_layer1 = get_vq_layer(vq_cfg.name, **vq1_kwargs)

  ################################
  # 3. MH Dropout Block
  ################################
  enc_proj2 = None
  enc_proj3 = None
  vq_layer2 = None
  mhd_layer = None
  if mhd_cfg.use:
    vq2_kwargs = {
      'n_emb': vq_cfg.n_emb2,
      'emb_dim': vq_cfg.emb_dim2,
      'beta': vq_cfg.beta,
      'tiled': vq_cfg.tiled,
      'ema_update': vq_cfg.ema_update,
      'random_restart': vq_cfg.random_restart
      }

    vq_layer2 = get_vq_layer(vq_cfg.name, **vq2_kwargs)
    enc_proj2 = nn.Conv1d(enc_conv_kwargs['channels'][-1], vq_cfg.emb_len2, 1)
    enc_proj3 = nn.Linear(vq_cfg.emb_dim1, vq_cfg.emb_dim2)

    mhd_inp_dim = vq_cfg.emb_dim2
    
    if mhd_cfg.bottleneck:
      hidden_dim = mhd_cfg.bottleneck_dim
    else:
      hidden_dim = mhd_inp_dim

    mhd_kwargs = {
      'inp_dim': mhd_inp_dim,
      'hidden_dim': hidden_dim,
      'out_dim': vq_cfg.emb_dim1,
      'dist_reduce': mhd_cfg.dist_reduce,
      'loss_reduce': mhd_cfg.dist_reduce,
      'loss_reduce_dims': mhd_cfg.loss_reduce_dims,
      'norm': mhd_cfg.norm,
      'dist_loss': mhd_cfg.dist_loss,
      'gamma': mhd_cfg.gamma,
      'dropout_rate': mhd_cfg.dropout_rate,
      'decoder_cfg': mhd_cfg.decoder,
      'bottleneck': mhd_cfg.bottleneck
    }

    mhd_layer = get_mhd_layer(mhd_cfg.name, **mhd_kwargs)
    

  data_model_kwargs = {
    'enc_conv': enc_conv,
    'enc_proj1': enc_proj1,
    'enc_proj2': enc_proj2,
    'enc_proj3': enc_proj3,
    'enc_tf': enc_tf,
    'dec_conv': dec_conv,
    'dec_tf_col': dec_tf_col,
    'dec_tf_row': dec_tf_row, 
    'dec_tf_name': dec_tf_name,
    'dec_proj_col': dec_proj_col,
    'dec_proj_row': dec_proj_row, 
    'vq_layer1': vq_layer1,
    'vq_layer2': vq_layer2,
    'mhd_layer': mhd_layer,
    'use_mhd': mhd_cfg.use,
    'hypothese_bsz': mhd_cfg.hypothese_bsz,
    'hypothese_count': mhd_cfg.hypothese_count,
    'emb_dim1': vq_cfg.emb_dim1,
    'emb_len1': vq_cfg.emb_len1,
    'emb_len2': vq_cfg.emb_len2,
    'conditional_encoder': cfg.model.continuous_data.encoder.chart_type_conditional,
    'conditional_decoder': cfg.model.continuous_data.decoder.chart_type_conditional,
    'use_pos_embs': cfg.model.continuous_data.use_pos_embs,
    'max_series_blocks': max_blocks.series, 
    'max_cont_blocks': max_blocks.points, 
    'scale_mode': cfg.data.dataset.chart_data.scale_mode,
    'scale_eps': cfg.data.dataset.chart_data.scale_eps,
    'scale_floor': cfg.data.dataset.chart_data.scale_floor,
    'scale_n_head': decoder_cfg.scale.n_head,
    'norm_mode': cfg.data.dataset.chart_data.norm_mode,
    'cont_loss_fn': cfg.model.continuous_data.loss_fn.continuous,
    'scale_loss_fn': cfg.model.continuous_data.loss_fn.scale,
    'fp16': use_fp16, 
    'debug': cfg.debug,
    'device': 'cuda:{}'.format(cfg.device_id),
  }

  model = ContinuousModel(**data_model_kwargs)

  opt = None
  params = list(filter(lambda p: p.requires_grad, model.parameters()))
  lr = cfg.train.optim.learning_rate
  betas = cfg.train.optim.betas
  if cfg.train.optim.type == 'AdamW':
    opt = torch.optim.AdamW(params, lr=lr, betas=betas)
  elif cfg.train.optim.type == 'Adam':
    opt = torch.optim.Adam(params, lr=lr, betas=betas)

  return model, opt


def init_disc_model(cfg, device_id):
  
  use_fp16      = cfg.fp16.use
  cd_cfg        = cfg.model.continuous_data
  max_blocks    = cd_cfg.encoder.max_blocks

  vq_cfg        = cd_cfg.vq
  disc_cfg      = cd_cfg.disc

  #### Encoder
  disc_conv_kwargs = disc_cfg.conv
  disc_conv_kwargs['channels'] = [max_blocks.points] + disc_conv_kwargs['channels'] 

  enc_conv = Conv1dEncoder(**disc_conv_kwargs)
  
  enc_tf = None
  enc_tf = init_transformer(disc_cfg.transformer, 
    block_size=max_blocks.points + 1, emb_dim=vq_cfg.emb_dim1, 
    use_pos_embs=False)

  kwargs = {
    'enc_conv': enc_conv,
    'enc_tf': enc_tf,
    'emb_dim1': vq_cfg.emb_dim1,
    'max_series_blocks': max_blocks.series,
    'max_cont_blocks': max_blocks.points,
    'norm_mode': cfg.data.dataset.chart_data.norm_mode,
    'disc_start': disc_cfg.disc_start,
    'disc_loss': disc_cfg.disc_loss,
    'disc_factor': disc_cfg.disc_factor,
    'disc_weight': disc_cfg.disc_weight,
    'disc_conditional': disc_cfg.disc_conditional,
    'use_pos_embs': disc_cfg.use_pos_embs,
    'device': f'cuda:{device_id}' if device_id != 'cpu' else 'cpu',
    'debug': cfg.debug,
    'fp16': use_fp16
  }

  model = Discriminator(**kwargs)

  opt = None
  params = list(filter(lambda p: p.requires_grad, model.parameters()))
  lr = cfg.train.optim.learning_rate
  betas = cfg.train.optim.betas
  if cfg.train.optim.type == 'AdamW':
    opt = torch.optim.AdamW(params, lr=lr, betas=betas)
  elif cfg.train.optim.type == 'Adam':
    opt = torch.optim.Adam(params, lr=lr, betas=betas)

  return model, opt

def init_chart_text_model(cfg, device_id, sep_token='<SEP>'):
  model_cfg = cfg.model.chart_text.hf_model
  
  hf_config = AutoConfig.from_pretrained(
        model_cfg.name, cache_dir=cfg.cache_dir
        )
  
  tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name,
        use_fast=False,
        cache_dir=cfg.cache_dir,
        model_max_length=model_cfg.model_max_length
        )

  num_added_toks = tokenizer.add_tokens([sep_token], special_tokens=True)
  
  if cfg.rank == 0:
    print("chart_text model config : backbone={} sep_token={} [{}]".format(model_cfg.name,sep_token,num_added_toks))

  model, opt = None, None
  
  model = AutoModelForSeq2SeqLM.from_pretrained(
        model_cfg.name,
        from_tf=False, 
        config=hf_config,
        cache_dir=cfg.cache_dir
        )

  model.resize_token_embeddings(len(tokenizer))
  print("Number of tokens: {}".format(len(tokenizer)))
  #print("Embeddings: {}".format(model.shared.shape))
  
  if model.config.decoder_start_token_id is None:
      raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

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

def init_seq_llm_model(cfg):
  sep_token = cfg.data.dataset.chart_data.sep_token
  model_cfg = cfg.model.seq.hf_model
  hf_config = AutoConfig.from_pretrained(
        model_cfg.name, cache_dir=cfg.cache_dir, 
        )
  tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name,
        use_fast=model_cfg.use_fast, cache_dir=cfg.cache_dir, 
        )
        
  num_added_toks = tokenizer.add_tokens([sep_token], special_tokens=True)

  model = AutoModelForSeq2SeqLM.from_pretrained(
      model_cfg.name,
      from_tf=False, 
      config=hf_config, cache_dir=cfg.cache_dir, 
      )
  model.resize_token_embeddings(len(tokenizer))

  return model, tokenizer

def init_ksm_model(cfg):

  model_cfg = cfg.eval.ksm  

  hf_config = AutoConfig.from_pretrained(
        model_cfg.name, cache_dir=cfg.cache_dir, 
        )
  
  tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name,
        use_fast=model_cfg.use_fast, cache_dir=cfg.cache_dir, 
        )

  model = AutoModelForSeq2SeqLM.from_pretrained(
        model_cfg.name,
        from_tf=False, 
        config=hf_config, cache_dir=cfg.cache_dir, 
        )

  model.resize_token_embeddings(len(tokenizer))
  if cfg.rank == 0:
    print("KSM CFG | backbone={}".format(model_cfg.name))

  return model, tokenizer

def get_vq_layer(name, **kwargs):
  if name == 'vq':
      m = VectorQuantizer
  else:
      raise NotImplementedError()
  return m(**kwargs)

def get_mhd_layer(name, **kwargs):
  if name.lower() == 'mhd1d':
    m = MHDropoutNetRandom1D
  else:
      raise NotImplementedError("Invalid distributed dropout block name: {} ".format(name))
  return m(**kwargs)

def init_transformer(tf_cfg, block_size, emb_dim, use_pos_embs=True):
  if tf_cfg.name == 'gpt':
    m = GPTNoEmbed(
      block_size=block_size,
      n_layer=tf_cfg.n_layer,
      n_head=tf_cfg.n_head,
      n_embd=emb_dim,
      use_pos_embs=use_pos_embs
    )
  elif tf_cfg.name == 't5_decoder':
    decoder_config = T5Config(
        vocab_size=0,
        num_layers=tf_cfg.n_layer,
        num_heads=tf_cfg.n_head,
        d_model=emb_dim,
        d_ff=int(emb_dim*4),
        d_kv=tf_cfg.d_kv,
        relative_attention_num_buckets= int(emb_dim/16) if tf_cfg.num_buckets == 0 else tf_cfg.num_buckets,
        relative_attention_max_distance= int(emb_dim/4) if tf_cfg.max_distance == 0 else tf_cfg.max_distance
      )
    decoder_config.is_decoder = True
    decoder_config.is_encoder_decoder = False
    m = T5Stack(decoder_config)
  else:
    raise NotImplementedError()

  return m

def create_ds_config(cfg):
  micro_bsz = "auto" if cfg.zero_opt.autotuning.enabled else cfg.batch_size
  accum_steps = "auto" if cfg.zero_opt.autotuning.enabled else cfg.train.gradient_accum_steps

  ds_cfg = {
    "tensorboard":{
      "enabled": False,
      "output_path": cfg.tb_dir,
      "job_name": cfg.exp_name,
    },
    "train_micro_batch_size_per_gpu": micro_bsz,
    "gradient_accumulation_steps": accum_steps,
    "gradient_clipping": cfg.train.max_grad_norm,
    "optimizer": {
      "type": cfg.train.optim.type,
      "params": {
        "lr": float(cfg.train.optim.learning_rate),
        "betas": cfg.train.optim.betas,
        "eps": float(cfg.train.optim.eps),
        "weight_decay": float(cfg.train.optim.weight_decay)
      }
    },
    "scheduler": {
      "type": cfg.train.scheduler.type,
      "params": {
          "warmup_type": cfg.train.scheduler.warmup_type,
          "warmup_min_lr": cfg.train.scheduler.warmup_min_lr,
          "warmup_max_lr":  cfg.train.scheduler.warmup_max_lr,
          "warmup_num_steps": cfg.train.scheduler.warmup_num_steps,
      }
    },
    "fp16": {
      "enabled": cfg.fp16.use,
      "loss_scale": cfg.fp16.loss_scale,
      "initial_scale_power": cfg.fp16.initial_scale_power,
      "loss_scale_window": cfg.fp16.loss_scale_window,
      "hysteresis": cfg.fp16.hysteresis,
      "min_loss_scale": cfg.fp16.min_loss_scale,
      "opt_level": cfg.fp16.opt_level
      },
    "zero_optimization": {
      "stage": cfg.zero_opt.stage,
      "allgather_partitions": cfg.zero_opt.allgather_partitions,
      "allgather_bucket_size": cfg.zero_opt.allgather_bucket_size,
      "overlap_comm": cfg.zero_opt.overlap_comm,
      "reduce_scatter": cfg.zero_opt.reduce_scatter,
      "reduce_bucket_size": cfg.zero_opt.reduce_bucket_size,
      "contiguous_gradients" : cfg.zero_opt.contiguous_gradients,
      "grad_hooks" : cfg.zero_opt.grad_hooks,
      "round_robin_gradients" : cfg.zero_opt.round_robin_gradients,
      "offload_param": {"device": cfg.zero_opt.offload_param.device},
      "offload_optimizer": {"device": cfg.zero_opt.offload_optimizer.device}
    },
    "autotuning": {
      "enabled": cfg.zero_opt.autotuning.enabled,
      "arg_mappings": {
        "train_micro_batch_size_per_gpu": cfg.zero_opt.autotuning.enabled,
        "gradient_accumulation_steps ": cfg.train.gradient_accum_steps
      }
      }
  }
  return ds_cfg
