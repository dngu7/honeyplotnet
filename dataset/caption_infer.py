# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------



class PmcCaptionInferenceDataset(object):
  def __init__(self, captions, tokenizer, max_source_len, pad_to_max_len=True):
    super().__init__()

    self.data = captions
    self.tokenizer = tokenizer
    self.max_source_len = max_source_len
    self.padding = "max_length" if pad_to_max_len else False

  def __getitem__(self, index):
      
    caption = self.data[index % len(self.data)]
    
    inputs = self.tokenizer(
      caption, max_length=self.max_source_len, 
      padding=self.padding, truncation=True, return_tensors="pt")

    for k in list(inputs.keys()):
      inputs[k] = inputs[k].squeeze(0)
    
    return inputs
  
  def __len__(self):
      return len(self.data)
    
