# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

'''

This file demonstrates how to run the FID metrics presented in the paper.

The `init_fid_model` automatically downloads the pre-trained weights from an AWS bucket a
and creates a new model.

The "Example Usage" code block below shows how to calculate FID with your own model. 
The inputs to the FID model need to be in a compatible format.
'''


import click
import os
from utils import load_cfg
from fid import init_fid_model, calculate_frechet_distance

@click.command()
@click.option('--config_file','-c', default='default.yaml')
def main(config_file):

    ###########################################
    # Load Configurations 
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    cfg_dir = os.path.join(cur_dir, 'config')
    cfg = load_cfg(config_file, cfg_dir)

    ###########################################
    # Initialize fid model
    fid_model, fid_stats = init_fid_model(cfg, load_path=cur_dir, device_id=0)

    mu1, sigma1, act1 = fid_stats['test']

    ###########################################
    # Example usage. 
    ###########################################

    # x_hat = chart_model.sample()
    # activations, _, _ = fid_model(x_hat)
    # mu2 = np.mean(acts, axis=0)
    # sigma2 = np.cov(acts, rowvar=False)

    # score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    # print(score)

if __name__ == "__main__":
  main()
    