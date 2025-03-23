# ---------------------------------------------------------------
# Copyright (c) Cybersecurity Cooperative Research Centre 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

import os
import wget
import torch

S3_BUCKET = 'https://s3.ap-southeast-2.amazonaws.com/decoychart/chartfid/'

def download_pretrained(path=None):
    
    assert os.path.exists(path), path
    
    #Download FID model snapshot and statistics on train and test datasets.
    snapshot_names = ['chartFid_snapshot.pth', 'pmc-train.npz', 'pmc-test.npz']
    for snapshot_name in snapshot_names:
        snapshot_local = os.path.join(path, snapshot_name)
        if not os.path.exists(snapshot_local):
            wget.download(os.path.join(S3_BUCKET, snapshot_name), snapshot_local)

def load_fid_snapshot(model, snapshot_path):
    snapshot = torch.load(snapshot_path)['fid_model']
    model.load_state_dict(snapshot)

    
