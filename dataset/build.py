# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import os
import copy

from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import DataCollatorForSeq2Seq

from utils import pickle_open

from .seq import PmcSeqDataset
from .caption_train import PmcCaptionDataset
from .generate import PmcGenerateDataset
from .continuous import PmcContinuousDataset
from .chart_text import PmcChartTextDataset
 

def init_dataloader(cfg, mode, stage, models, tokenizers, return_dataset=False):
  
  data_cfg        = cfg.data
  data_path       = cfg.data_path
  batch_size      = cfg.batch_size
  num_workers     = cfg.num_workers
  use_gpu         = cfg.gpu.use
  use_distributed = cfg.torch_dist.use
  use_fp16        = cfg.fp16.use


  #Obtain dataset and model configurations here
  if stage in ['continuous','seq']:
    dataset_cfg = data_cfg.dataset.chart_data
    model_cfg   = cfg.model.continuous_data.text_model

  elif stage  == 'caption':
    dataset_cfg = data_cfg.dataset.caption
    model_cfg   = cfg.model.caption.hf_model

  elif stage in ['chart_text', 'generate']:
    dataset_cfg = data_cfg.dataset.chart_data
    model_cfg   = cfg.model.chart_text_data.hf_model
  
  #Combine into the same
  dataset_cfg = {**dataset_cfg, **model_cfg, 'debug': cfg.debug}
  
  train_dataset, val_dataset = select_dataset(
    mode=mode, stage=stage, 
    root=data_path, 
    dataset_cfg=dataset_cfg, 
    tokenizers=tokenizers, 
    dataset_name=data_cfg.name)

  if stage in ['caption', 'chart_text']:
    module = models[stage].module if hasattr(models[stage], 'module') else models[stage]
    label_pad_token_id = -100 if dataset_cfg.get('ignore_pad_token_for_loss') else tokenizers[stage].pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizers[stage],
        model=module,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if use_fp16 else None,
      )
  else:
    data_collator = train_dataset.collate_fn

  if return_dataset:
    return train_dataset, val_dataset, data_collator

  else:

    train_sampler = ElasticDistributedSampler(train_dataset) if use_distributed else RandomSampler(train_dataset)
    val_sampler = None if use_distributed else SequentialSampler(val_dataset)
  
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_gpu,
        collate_fn=data_collator,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=use_gpu,
        sampler=val_sampler
    )

    return train_loader, val_loader

def select_dataset(mode, stage, root, dataset_cfg, tokenizers, dataset_name='pmc', **kwargs):

  #Setup path to the data directory
  root = os.path.expanduser(root)

  #Remove randomness from validation set
  val_cfg  = copy.deepcopy(dataset_cfg)
  val_cfg['widen_rate'] = 0.0

  data_name  = '{}_{}'.format(dataset_name, 'data')
  train_path = os.path.join(root, "{}_{}.pkl".format(data_name, 'train'))
  val_path   = os.path.join(root, "{}_{}.pkl".format(data_name, 'test'))

  assert os.path.exists(train_path), train_path
  assert os.path.exists(val_path), val_path

  train_data = pickle_open(train_path)
  val_data = pickle_open(val_path)

  if stage == 'continuous':
    train_ds = PmcContinuousDataset(data=train_data, **dataset_cfg)
    val_ds   = PmcContinuousDataset(data=val_data, **val_cfg)
  elif stage == 'seq':
    train_ds = PmcSeqDataset(data=train_data, tokenizer=tokenizers['seq'], **dataset_cfg)
    val_ds   = PmcSeqDataset(data=val_data, tokenizer=tokenizers['seq'], **val_cfg)
  elif stage == 'generate':
    train_ds = PmcGenerateDataset(data=train_data, tokenizer1=tokenizers['caption'], tokenizer2=tokenizers['chart_text'],**dataset_cfg)
    val_ds   = PmcGenerateDataset(data=val_data, tokenizer1=tokenizers['caption'], tokenizer2=tokenizers['chart_text'], **val_cfg)   
  elif stage == 'caption':
    train_ds = PmcCaptionDataset(data=train_data, tokenizer1=tokenizers['caption'], **dataset_cfg)
    val_ds   = PmcCaptionDataset(data=val_data, tokenizer1=tokenizers['caption'], **val_cfg)
  elif stage == 'chart_text':
    train_ds = PmcChartTextDataset(data=train_data, tokenizer1=tokenizers['caption'], tokenizer2=tokenizers['chart_text'],**dataset_cfg)
    val_ds   = PmcChartTextDataset(data=val_data, tokenizer1=tokenizers['caption'], tokenizer2=tokenizers['chart_text'], **val_cfg)
  else:
    raise NotImplementedError()


  return train_ds, val_ds

