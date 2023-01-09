# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

from .base import get_text_window
from .data import PmcDataDataset

class PmcGenerateDataset(PmcDataDataset):
  def __init__(self, **kwargs):
    super().__init__( **kwargs)

  def __getitem__(self, index):
      
    while True:
      d = self.data[index % len(self.data)]

      try:
        fig_id = d['fig_id']
        context_start = d['fig_index'][fig_id][0]
      except:
        #print("index: {}     failed to extract fig id: {} from ==> {}".format(index, fig_id, d['fig_index']))
        self.del_list.append(index)
        index += 1
        continue
      
      all_text = d['all_text']
      captions = d['captions']

      all_text = self.preprocess_all_text(all_text)

      _, all_text, context_start = self.get_caption_label(
        captions, all_text, context_start)

      context = get_text_window(
        all_text, context_start, 
        tgt_token=self.tgt_token,
        window_size=self.window_size, 
        widen_rate=self.widen_rate, 
        max_widen_len=self.max_widen_len, 
        min_sent_len=self.min_sent_len)
      
      return context

  def collate_fn(self, list_batch):
    return list_batch