# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

from .gen import GenRunner
from .text import ChartTextRunner
from .continuous import ContinuousRunner
from .seq import SeqRunner

def get_runners(cfg, stage):
  if stage == 'generate':
    runner = GenRunner
  elif stage == 'continuous':
    runner = ContinuousRunner
  elif stage == 'seq':
    runner = SeqRunner
  else:
    raise NotImplementedError("")
  
  return runner(stage, cfg)



