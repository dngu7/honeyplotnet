# ---------------------------------------------------------------
# Copyright (c) Cybersecurity Cooperative Research Centre 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------


import os
import sys
import torch
import numpy as np
import torch.distributed as dist
import logging
import time
from tensorboardX import SummaryWriter
from datetime import timedelta


def launch_dist_backend(dist_cfg, timeout=1800, debug=False):
  if dist_cfg.use:
    try:
        dist.init_process_group(
            backend=dist_cfg.backend, 
            init_method=dist_cfg.init_method, 
            timeout=timedelta(seconds=timeout)
            )
    except ValueError:
        dist_cfg.use = False
        print("Initialising Pytorch Distributed failed. It might not be available - switching to non-distributed mode.")


def average_gradients(model):
    errors = 0
    size = float(dist.get_world_size())
    for name, param in model.named_parameters():
        # if param.grad is None and param.requires_grad:
        #     print("Unused Gradient >> Name: {}".format(name))
        #     #print(param.grad.data.shape)
        #     raise
        
        if param.grad is None and 'perceptual_loss' not in name:
            print("Unused Gradient >> Name: {} [{}]".format(name, param.requires_grad))
            errors += 1

        if param.requires_grad:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
    
    if errors >= 0:
        print("Error Count: {}".format(errors))

def average_tensor(t):
    size = float(dist.get_world_size())
    dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
    t.data /= size

class Logger(object):
    def __init__(self, rank, save):
        # other libraries may set logging before arriving at this line.
        # by reloading logging, we can get rid of previous configs set by other libraries.
        from importlib import reload
        reload(logging)
        self.rank = rank
        if self.rank == 0:
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(save, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            self.start_time = time.time()

    def info(self, string, *args):
        if self.rank == 0:
            elapsed_time = time.time() - self.start_time
            elapsed_time = time.strftime(
                '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
            if isinstance(string, str):
                string = elapsed_time + string
            else:
                logging.info(elapsed_time)
            logging.info(string, *args)


class Writer(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=save, flush_secs=20)

    def add_scalar(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_figure(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_image(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_histogram(*args, **kwargs)

    def add_histogram_if(self, write, *args, **kwargs):
        if write and False:   # Used for debugging.
            self.add_histogram(*args, **kwargs)

    def close(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.close()



