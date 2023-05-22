# ---------------------------------------------------------------
# Copyright (c) _____________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import os
import click
import wget

from utils import load_cfg

S3_BUCKET = 'https://decoychart.s3.ap-southeast-2.amazonaws.com/weights/'

@click.command()
@click.option('--config_file','-c', default='mvqgan_t5.yaml', help='Configuration files in config folder')
@click.option('--work','-w', default='home', help='Work environment')
def main(config_file, work):
  print("Downloading pretrained weights for HoneyPlotNet...")
  print("Plot Data Model        : MVQGAN")
  print("Multimodal Transformer : T5")
  print("Configuration File     : {}".format(config_file))
  
  ###########################################
  # Load Configurations 
  cur_dir = os.path.dirname(os.path.realpath(__file__))
  cfg_dir = os.path.join(cur_dir, 'config')

  cfg = load_cfg(config_file, cfg_dir)
  cfg.work_env = work
  cfg.cur_dir = cur_dir

  # This allows specification of different work environments
  cfg._exp_dir = getattr(cfg.exp_dir, work)

  ##########################################
  # Check experiment and data directory exist
  if cfg._exp_dir is None:
    cfg._exp_dir = os.path.join(cur_dir, 'exp')
    os.makedirs(cfg._exp_dir, exist_ok=True)
    print(f"Experiment directory not specified in config. Default: {cfg._exp_dir}")
    
  if cfg.exp_name is None:
    cfg.exp_name = '_'.join([config_file.replace('.yaml',''), str(cfg.seed)])
  
 ###########################################
  # Setup directories
  cfg.save_dir = os.path.join(cfg._exp_dir, cfg.exp_name)
  os.makedirs(cfg.save_dir, exist_ok=True)

  cfg.ckpt_dirs = {}
  #Creates new data directories
  for dir_name, cfg_attr, cfg_base in [
      ('checkpoints', 'ckpt_dir','save_dir'),
      ]:
    new_path = os.path.join(cfg[cfg_base], dir_name)
    os.makedirs(new_path, exist_ok=True)
    setattr(cfg, cfg_attr, new_path) 

  for s in ['continuous','seq']:
    stage_ckpt_dir = os.path.join(cfg.ckpt_dir, s)
    os.makedirs(stage_ckpt_dir, exist_ok=True)
    cfg.ckpt_dirs[s] = stage_ckpt_dir

  #Download snapshots from AWS 
  continuous_snapshot_url = S3_BUCKET + 'mvqgan-t5/continuous/best_100_snapshot.pth'
  seq_snapshot_url = S3_BUCKET + 'mvqgan-t5/seq/best_1040_snapshot.pth' 

  continuous_snapshot_local = os.path.join(cfg.ckpt_dirs['continuous'], 'best_100_snapshot.pth')
  if not os.path.exists(continuous_snapshot_local):
      print(f"Downloading Plot Data Model weights from: {continuous_snapshot_url}")
      print(f"Saving to: {cfg.ckpt_dirs['continuous']}")
      wget.download(continuous_snapshot_url, continuous_snapshot_local)
  
  seq_snapshot_local = os.path.join(cfg.ckpt_dirs['seq'], 'best_1040_snapshot.pth')
  if not os.path.exists(seq_snapshot_local):
      print(f"Downloading Multimodal Transformer weights from: {seq_snapshot_url}")
      print(f"Saving to: {cfg.ckpt_dirs['seq']}")
      wget.download(seq_snapshot_url, seq_snapshot_local)

  print("Download complete.")
  print("To conduct evaluation for each task, please run `python eval.py`")

if __name__ == "__main__":
  main()
    
  