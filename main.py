# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import os
import click

from state import State
from dataset import init_dataloader 
from runner import get_runners 
from models import init_model, load_checkpoint 

from utils import (
  load_cfg,
  set_seeds,
  launch_dist_backend,
  start_debug_mode,
  setup_gpu_cfg, 
)

STAGES = ['caption', 'chart_text', 'continuous',  'seq', 'generate']

@click.command()
@click.option('--config_file','-c', default='default.yaml')
@click.option('--mode','-m', default='train')
@click.option('--stage','-s', default='caption')
@click.option('--debug', '-bug', default=0)
@click.option('--work','-w', default='home')
@click.option('--dist', '-di', default=None)
@click.option('--seed', '-se', default=0)
@click.option('--local_rank', '-lr', default=None)
def main(config_file, mode, stage, work, debug, dist, seed, local_rank):

  assert stage in STAGES
  assert mode in ['train', 'eval','save']
  
  if local_rank is not None:
    os.environ["LOCAL_RANK"] = str(local_rank)

  ###########################################
  # Load Configurations 
  cur_dir = os.path.dirname(os.path.realpath(__file__))
  cfg_dir = os.path.join(cur_dir, 'config')

  cfg = load_cfg(config_file, cfg_dir)
  cfg.work_env = work
  cfg.cur_dir = cur_dir
  cfg.cur_stage = stage
  cfg.batch_size = getattr(cfg.batch_size_per_gpu, stage)

  # This allows specification of different work environments
  cfg._exp_dir = getattr(cfg.exp_dir, work)
  cfg.data_path = getattr(cfg.data.path, work)

  ##########################################
  # Check experiment and data directory exist
  if cfg._exp_dir is None:
    cfg._exp_dir = os.path.join(cur_dir, 'exp')
    os.makedirs(cfg._exp_dir, exist_ok=True)
    print(f"Experiment directory not specified in config. Default: {cfg._exp_dir}")
  
  if cfg.data_path is None:
    cfg.data_path = os.path.join(cur_dir, 'data')
    os.makedirs(cfg.data_path, exist_ok=True)
    print(f"Data directory not specified in config.       Default: {cfg.data_path}")

  # Check active model list in config file.
  if stage == 'generate':
    cfg.model.active = ['caption','continuous','seq', 'chart_text']
  elif stage not in cfg.model.active:
    cfg.model.active += [stage]
    
  ##########################################
  # Replace config with command line options (if any)

  if dist is not None:
    cfg.torch_dist.use = True if dist == '1' else False
    if dist == '0':
      cfg.torch_dist.deepspeed = False

  if seed is not None:
    cfg.seed = int(seed)

  set_seeds(cfg.seed)
  if debug or cfg.debug:
    cfg = start_debug_mode(cfg)

  if cfg.gpu.use:
    launch_dist_backend(cfg.torch_dist, debug=cfg.debug, timeout=cfg.timeout)

  if not cfg.train.resume.is_resume:
    cfg.exp_name = '_'.join([config_file.replace('.yaml',''), str(cfg.seed)])
  else:
    cfg.exp_name = cfg.train.resume.exp_name
  
  ###########################################
  # Setup directories
  cfg.tb_dir = os.path.join(cfg._exp_dir, 'tensorboard')
  os.makedirs(cfg.tb_dir, exist_ok=True)

  cfg.save_dir = os.path.join(cfg._exp_dir, cfg.exp_name)
  os.makedirs(cfg.save_dir, exist_ok=True)

  cfg.cache_dir = os.path.join(cfg.data_path, 'cache')
  os.makedirs(cfg.cache_dir, exist_ok=True)
  os.environ['TRANSFORMERS_CACHE'] = cfg.cache_dir

  cfg.ckpt_dir = os.path.join(cfg.save_dir, 'checkpoints')
  os.makedirs(cfg.ckpt_dir, exist_ok=True)
  cfg.ckpt_dirs = {}

  cfg.sample_dir =  os.path.join(cfg.save_dir, 'samples')
  os.makedirs(cfg.sample_dir, exist_ok=True)
  cfg.sample_dirs = {}

  for s in STAGES:
    stage_ckpt_dir = os.path.join(cfg.ckpt_dir, s)
    os.makedirs(stage_ckpt_dir, exist_ok=True)
    cfg.ckpt_dirs[s] = stage_ckpt_dir

    stage_sample_dir = os.path.join(cfg.sample_dir, s)
    os.makedirs(stage_sample_dir, exist_ok=True)
    cfg.sample_dirs[s] = stage_sample_dir


  ###########################################
  cfg = setup_gpu_cfg(cfg)

  if cfg.rank in ['cpu', 0]:
    print("Torch Distributed     : {}".format(cfg.torch_dist.use))
    print("Deepspeed             : {}".format(cfg.torch_dist.deepspeed))    
    print("Stage                 : {}".format(stage))
    print("Mode                  : {}".format(mode))
    print("Experiment Dir        : {}".format(cfg.save_dir))
    print("Dataset               : {}".format(cfg.data.name))
    print("Config                : {}".format(config_file))
    print("--gan                 : {}".format(cfg.model.continuous_data.disc.use))
    print("--mhd                 : {}".format(cfg.model.continuous_data.mhd.use))

  #Initialise model, optimisers and dataset
  models, tokenizers, opts, schs, loaders = init_model(
    cfg, mode, stage, cfg.device_id)

  if loaders.get('train') is not None and loaders.get('val') is not None:
    train_loader = loaders['train']
    val_loader = loaders['val']
  else:
    train_loader, val_loader = init_dataloader(
          cfg, mode, stage, models, tokenizers)

  state = State(models, tokenizers, opts, schs, rank=cfg.rank, use_deepspeed=cfg.torch_dist.deepspeed, mode=mode, stage=stage)
  state = load_checkpoint(
    state, 
    cfg.ckpt_dirs, 
    cfg.device_id, cfg.rank, 
    cfg.torch_dist.use, 
    cfg.torch_dist.deepspeed)

  runner = get_runners(cfg, stage)
  runner.global_step = state.global_step
  runner.metrics = state.metrics

  eval_freq = cfg.train.intervals.eval
  gen_freq  = cfg.train.intervals.gen
  start_epoch = state.epoch + 1

  if mode in ['eval', 'val', 'save'] or stage == 'generate': 
    total_epochs = start_epoch + 1
  elif stage in ['caption', 'continuous' ,'chart_text']:
    total_epochs = int(getattr(cfg.train.epochs,stage))
  elif stage == 'seq':
    if start_epoch < cfg.train.epochs.continuous:
      start_epoch = cfg.train.epochs.continuous
    total_epochs = cfg.train.epochs.seq + cfg.train.epochs.continuous
  else:
    total_epochs = cfg.train.epochs.total
  
  if cfg.rank == 0:
    runner.logger.info(f"=> stage: {stage}, mode: {mode}, start_epoch: {start_epoch}, total_epochs: {total_epochs}, best_score: {state.best_score.get(stage)}")
    runner.logger.info("Active components >>")
    runner.logger.info("model      : {}".format([name for name, m in models.items() if m is not None]))
    runner.logger.info("opt        : {}".format([name for name, m in opts.items() if m is not None]))
    runner.logger.info("tokenizers : {}".format([name for name, m in tokenizers.items() if m is not None]))
    
  for epoch in range(start_epoch, total_epochs):
    is_best = False
    best_score = None
    runner.epoch = epoch
    state.epoch = runner.epoch
    state.global_step = runner.global_step
    state.metrics = runner.metrics

    if cfg.torch_dist.use and not cfg.torch_dist.deepspeed:
      train_loader.batch_sampler.sampler.set_epoch(epoch)

    if stage == 'generate':
      runner.eval(val_loader, models, tokenizers)

    else:
      if mode == 'train':
        runner.train(train_loader, models, tokenizers, opts, schs)
        
      if mode == 'eval' or (epoch % eval_freq == 0 and epoch > cfg.train.epochs.warmup):
        results = runner.eval(
          val_loader, models, tokenizers, metric_key_prefix='eval', epoch=epoch, create_sample=(mode == 'eval'))

        if stage in ['caption', 'chart_text']:
          best_score = sum([results.metrics[m] for m in ["rouge1", "rouge2", "rougeL"]])
          is_best = best_score > state.best_score[stage]
          state.best_score[stage] = max(best_score, state.best_score[stage])
        elif stage in ['continuous','seq']:
          best_score = results['score']
          is_best = best_score < state.best_score[stage]
          state.best_score[stage] = min(best_score, state.best_score[stage])

        if best_score is not None:
          runner.logger.info("Score update | Best Score: {:.4f} Current: {:.4f}  | is_best: {}".format(state.best_score[stage], best_score, is_best))

      state.scaler[stage] = runner.scaler
      state.schs[stage]   = runner.lr_scheduler


    if mode == 'train' and (cfg.rank == 0 or cfg.torch_dist.deepspeed):
      if is_best:
        tag = 'best_{}'.format(round(epoch, -2))
        state.save(ckpt_dirs=cfg.ckpt_dirs, use_deepspeed=cfg.torch_dist.deepspeed, tag=tag, save_latest=False)
      
      tag = 'last_{}'.format(round(epoch, -2))
      state.save(ckpt_dirs=cfg.ckpt_dirs, use_deepspeed=cfg.torch_dist.deepspeed, tag=tag, save_latest=True)
  

if __name__ == "__main__":
  main()
    
  