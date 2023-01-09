# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import torch.nn as nn

class Coder(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)