# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import os
import yaml
import click
import time

from state import State
from dataset import init_dataloader 
from runner import get_runners 
from models import init_model, load_checkpoint 
from fid import init_fid_model

from utils import (
  load_cfg,
  set_seeds,
  launch_dist_backend,
  start_debug_mode,
  setup_gpu_cfg, 
)

STAGES = ['continuous', 'seq']

@click.command()
@click.option('--config_file','-c', default='default.yaml', help='Configuration files in config folder')
@click.option('--mode','-m', default='train', help='Runner mode. Select from ["train","eval"]')
@click.option('--stage','-s', default='continuous', help='Training stages. Select from ["continuous","seq"]')
@click.option('--work','-w', default='home', help='Work environment')
@click.option('--distributed', '-d', default=None, help='Deactivate Pytorch distributed package')
@click.option('--debug', '-bug', default=0, help='Activates debug mode')
@click.option('--local_rank', '-lr', default=None, help='For distributed.')
def main(config_file, mode, stage, work, debug, distributed, local_rank):

  assert stage in STAGES
  assert mode in ['train','eval','generate'], mode
  
  if local_rank is not None:
    os.environ["LOCAL_RANK"] = str(local_rank)

  ###########################################
  # Load Configurations 
  cur_dir = os.path.dirname(os.path.realpath(__file__))
  cfg_dir = os.path.join(cur_dir, 'config')

  cfg = load_cfg(config_file, cfg_dir)
  set_seeds(cfg.seed)
  cfg.work_env = work
  cfg.cur_dir = cur_dir
  cfg.cur_stage = stage
  cfg.batch_size = getattr(cfg.batch_size_per_gpu, stage)

  # This allows specification of different work environments
  cfg._exp_dir = getattr(cfg.exp_dir, work)
  cfg.data_path = getattr(cfg.data.path, work)

  # Automatically remove fid in opt_mode==1
  opt_mode = int(cfg.model.seq.opt_mode)
  if mode == 'train' and stage == 'seq':
    if opt_mode == 1:
      cfg.eval.fid = False
    elif opt_mode == 2:
      cfg.eval.fid = True

  ##########################################
  # Check experiment and data directory exist
  if cfg._exp_dir in [None,'']:
    cfg._exp_dir = os.path.join(cur_dir, 'exp')
    os.makedirs(cfg._exp_dir, exist_ok=True)
    print(f"Experiment directory not specified in config. Default: {cfg._exp_dir}")
  
  if cfg.data_path in [None,'']:
    cfg.data_path = os.path.join(cur_dir, 'data')
    os.makedirs(cfg.data_path, exist_ok=True)
    print(f"Data directory not specified in config.       Default: {cfg.data_path}")
  
  # Check active model list in config file.
  if stage not in cfg.model.active:
    cfg.model.active += [stage]

  ##########################################
  # Replace config with command line options (if any)

  if distributed is not None:
    cfg.torch_dist.use = True if distributed == '1' else False

  if debug or cfg.debug:
    cfg = start_debug_mode(cfg)

  if cfg.use_gpu:
    launch_dist_backend(cfg.torch_dist, debug=cfg.debug, timeout=cfg.timeout)

  if cfg.exp_name is None:
    cfg.exp_name = '_'.join([config_file.replace('.yaml',''), str(cfg.seed)])
  else:
    cfg.exp_name = cfg.exp_name
  
  ###########################################
  # Setup directories
  cfg.save_dir = os.path.join(cfg._exp_dir, cfg.exp_name)
  os.makedirs(cfg.save_dir, exist_ok=True)

  #Save cfg into directory
  cfg_fn = os.path.join(cfg.save_dir, f'config_{int(time.time())}.yaml')
  with open(cfg_fn, 'w') as file:
    yaml.dump(cfg, file)

  cfg.ckpt_dirs = {}
  cfg.sample_dirs = {}

  #Creates new data directories
  for dir_name, cfg_attr, cfg_base in [
      ('fid_stats', 'fid_dir','data_path'),
      ('cache', 'cache_dir','data_path'),
      ('tensorboard','tb_dir', '_exp_dir'), 
      ('checkpoints', 'ckpt_dir','save_dir'),
      ('samples', 'sample_dir','save_dir')
      ]:
    new_path = os.path.join(cfg[cfg_base], dir_name)
    os.makedirs(new_path, exist_ok=True)
    setattr(cfg, cfg_attr, new_path) 
  os.environ['TRANSFORMERS_CACHE'] = cfg.cache_dir

  for s in STAGES:
    stage_ckpt_dir = os.path.join(cfg.ckpt_dir, s)
    os.makedirs(stage_ckpt_dir, exist_ok=True)
    cfg.ckpt_dirs[s] = stage_ckpt_dir

    stage_sample_dir = os.path.join(cfg.sample_dir, s)
    os.makedirs(stage_sample_dir, exist_ok=True)
    cfg.sample_dirs[s] = stage_sample_dir

  gen_sample_dir = os.path.join(cfg.sample_dir, 'generate')
  os.makedirs(gen_sample_dir, exist_ok=True)

  cfg.sample_dirs['generate'] = {}
  cfg.sample_dirs['generate']['base'] = gen_sample_dir
  
  for d in ['json','mpl']:
    new_dir = os.path.join(gen_sample_dir, d)
    os.makedirs(new_dir, exist_ok=True)
    cfg.sample_dirs['generate'][d] = new_dir

  ###########################################
  cfg = setup_gpu_cfg(cfg)

  #Initialise model, optimisers and dataset
  models, tokenizers, opts, schs = init_model(
    cfg, mode, stage, cfg.device_id)

  train_loader, val_loader = init_dataloader(cfg, mode, stage, models, tokenizers)

  state = State(models, tokenizers, opts, schs, rank=cfg.rank, mode=mode, stage=stage)
  
  state = load_checkpoint(
    state, 
    cfg.ckpt_dirs, 
    cfg.device_id, cfg.rank, 
    cfg.torch_dist.use)
  
  #Initialize pre-trained fid model
  fid_stats = None
  if cfg.eval.fid:
    models['fid'], fid_stats = init_fid_model(cfg, load_path=cfg.fid_dir, device_id=cfg.device_id)
  
  runner = get_runners(cfg, stage, mode)
  runner.global_step = state.global_step
  runner.metrics = state.metrics
  runner.fid_stats = fid_stats

  eval_freq = cfg.train.intervals.eval
  start_epoch = state.epoch + 1

  if mode in ['eval', 'generate']: 
    total_epochs = start_epoch + 1
  elif stage in ['continuous']:
    total_epochs = int(getattr(cfg.train.epochs,stage))
  elif stage == 'seq':
    if start_epoch < cfg.train.epochs.continuous:
      start_epoch = cfg.train.epochs.continuous
    total_epochs = cfg.train.epochs.seq + cfg.train.epochs.continuous
  else:
    total_epochs = cfg.train.epochs.total
  
  if mode == 'train' and stage == 'seq':
    if opt_mode == 0:
      pass
    elif opt_mode == 1:
      tasks = ['categorical','series_name','axis','caption']
      train_loader.dataset.set_tasks(tasks)
      val_loader.dataset.set_tasks(tasks)
    elif opt_mode == 2:
      train_loader.dataset.set_tasks('data')
      val_loader.dataset.set_tasks('data')
    else:
      raise

  score_stage = stage + f'_{str(opt_mode)}' if stage == 'seq' else stage
  if score_stage not in state.best_score and stage == 'seq':
    state.best_score[score_stage] = 0.0 if score_stage == 'seq_1' else float('inf')

  if cfg.rank in ['cpu', 0]:
    runner.logger.info("GPU                   : {}".format(cfg.use_gpu))
    runner.logger.info("Torch Distributed     : {}".format(cfg.torch_dist.use))
    runner.logger.info("Stage                 : {}".format(stage))
    runner.logger.info("Mode                  : {}".format(mode))
    runner.logger.info("Experiment Dir        : {}".format(cfg.save_dir))
    runner.logger.info("Dataset               : {}".format(cfg.data.name))
    runner.logger.info("FID Test              : {}".format(cfg.eval.fid))
    runner.logger.info("Config                : {}".format(config_file))
    runner.logger.info("--gan                 : {}".format(cfg.model.continuous_data.disc.use))
    runner.logger.info("--mhd                 : {}".format(cfg.model.continuous_data.mhd.use))
    runner.logger.info(f"start_epoch: {start_epoch}, total_epochs: {total_epochs}, best_score[{score_stage}]: {state.best_score.get(score_stage)}")
    runner.logger.info("Active components >>")
    runner.logger.info("model      : {}".format([name for name, m in models.items() if m is not None]))
    runner.logger.info("opt        : {}".format([name for name, m in opts.items() if m is not None]))
    runner.logger.info("tokenizers : {}".format([name for name, m in tokenizers.items() if m is not None]))

    if hasattr(train_loader.dataset, 'tasks'):
      runner.logger.info("tasks      : {}".format(train_loader.dataset.tasks))

  for epoch in range(start_epoch, total_epochs):
    is_best = False
    best_score = None
    runner.epoch = epoch
    state.epoch = runner.epoch
    state.global_step = runner.global_step
    state.metrics = runner.metrics

    if cfg.torch_dist.use:
      train_loader.batch_sampler.sampler.set_epoch(epoch)

    if mode == 'generate':
      runner.generate(val_loader, models, tokenizers)
    elif mode == 'train':
      runner.train(train_loader, models, tokenizers, opts, schs)
      
    if mode == 'eval' or (epoch % eval_freq == 0 and epoch > cfg.train.epochs.warmup):
      results = runner.eval(
        val_loader, models, tokenizers, 
        metric_key_prefix='eval', epoch=epoch, 
        create_sample=(mode == 'eval'))

      best_score = results['score']
      if stage == 'seq' and opt_mode == 1:
        is_best = best_score > state.best_score[score_stage]
        state.best_score[score_stage] = max(best_score, state.best_score[score_stage])
      else:
        is_best = best_score < state.best_score[score_stage]
        state.best_score[score_stage] = min(best_score, state.best_score[score_stage])
        
      if best_score is not None:
        runner.logger.info("Score update | best: {:.4f} last: {:.4f} [{}]".format(state.best_score[stage], best_score, is_best))

    if mode == 'train' and cfg.rank == 0:
      if is_best:
        tag = 'best_{}'.format(round(epoch, -1))
        state.save(ckpt_dirs=cfg.ckpt_dirs, tag=tag, save_latest=False)
      
      tag = 'last_{}'.format(round(epoch, -1))
      state.save(ckpt_dirs=cfg.ckpt_dirs, tag=tag, save_latest=True)
  

if __name__ == "__main__":
  main()
    
  