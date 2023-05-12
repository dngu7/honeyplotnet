# ---------------------------------------------------------------
# Copyright (c) _____________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import os
import yaml
import click

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

@click.command()
@click.option('--config_file','-c', default='default.yaml', help='Configuration files in config folder')
@click.option('--work','-w', default='home', help='Work environment')
@click.option('--distributed', '-d', default=None, help='Deactivate Pytorch distributed package')
@click.option('--debug', '-bug', default=0, help='Activates debug mode')
@click.option('--local_rank', '-lr', default=None, help='For distributed.')
def main(config_file, work, debug, distributed, local_rank):
  stage = 'seq'
  mode = 'eval'
  
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

  #Create settings for evaluation
  cfg.model.seq.opt_mode = 0    
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
  cfg.model.active  = ['continuous', 'seq']
    
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

  for s in ['continuous','seq']:
    stage_ckpt_dir = os.path.join(cfg.ckpt_dir, s)
    os.makedirs(stage_ckpt_dir, exist_ok=True)
    cfg.ckpt_dirs[s] = stage_ckpt_dir

    stage_sample_dir = os.path.join(cfg.sample_dir, s)
    os.makedirs(stage_sample_dir, exist_ok=True)
    cfg.sample_dirs[s] = stage_sample_dir

  ###########################################
  cfg = setup_gpu_cfg(cfg)
  
  #Initialise model, optimisers and dataset
  models, tokenizers, opts, schs = init_model(
    cfg, mode, stage, cfg.device_id)

  #Initialize pre-trained fid model
  fid_stats = None
  if cfg.eval.fid:
    models['fid'], fid_stats = init_fid_model(cfg, load_path=cfg.fid_dir, device_id=cfg.device_id)

  state = State(models, tokenizers, opts, schs, rank=cfg.rank, mode=mode, stage=stage)

  state = load_checkpoint(
    state, 
    cfg.ckpt_dirs, 
    cfg.device_id, cfg.rank, 
    cfg.torch_dist.use)

  #Initialize pre-trained fid model
  runner = get_runners(cfg, stage, mode)
  runner.global_step = state.global_step
  runner.metrics = state.metrics
  runner.fid_stats = fid_stats

  #Number of data only epochs
  start_epoch = state.epoch + 1
  total_epochs = start_epoch + 1

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
    runner.logger.info(f"start_epoch: {start_epoch}, total_epochs: {total_epochs}")
    runner.logger.info("Active components >>")
    runner.logger.info("model      : {}".format([name for name, m in models.items() if m is not None]))
    runner.logger.info("opt        : {}".format([name for name, m in opts.items() if m is not None]))
    runner.logger.info("tokenizers : {}".format([name for name, m in tokenizers.items() if m is not None]))

  #Loop through each task
  epoch = start_epoch
  runner.epoch = epoch
  state.epoch = runner.epoch
  state.global_step = runner.global_step
  state.metrics = runner.metrics

  for tasks in ['caption', 'categorical','series_name','axis', 'data']: 
    cfg.data.dataset.tasks = [tasks]
    runner.cfg.eval.fid = tasks == 'data'
    
    _, val_loader = init_dataloader(cfg, mode, stage, models, tokenizers, return_dataset=False)
    
    if cfg.rank in ['cpu', 0]:
      runner.logger.info(f"E{epoch} Tasks: {val_loader.dataset.tasks}")
      
    _ = runner.eval(
      val_loader, models, tokenizers, 
      metric_key_prefix='eval', 
      epoch=epoch, 
      step_count=None)


if __name__ == "__main__":
  main()
    
  