# ---------------------------------------------------------------
# Copyright (c) __________________________ 2023.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Forked from:
# https://github.com/pytorch/elastic/blob/master/examples/imagenet/main.py
# ---------------------------------------------------------------

import os
import io
import numpy
import torch
import torch.distributed as dist
import shutil
from contextlib import contextmanager

@contextmanager
def tmp_process_group(backend):
    cpu_pg = dist.new_group(backend=backend)
    try:
        yield cpu_pg
    finally:
        dist.destroy_process_group(cpu_pg)

def load_checkpoint(state, ckpt_dirs, device_id, rank, distributed):
    """
    Loads a local checkpoint (if any). Otherwise, checks to see if any of
    the neighbors have a non-zero state. If so, restore the state
    from the rank that has the most up-to-date checkpoint.

    .. note:: when your job has access to a globally visible persistent storage
              (e.g. nfs mount, S3) you can simply have all workers load
              from the most recent checkpoint from such storage. Since this
              example is expected to run on vanilla hosts (with no shared
              storage) the checkpoints are written to local disk, hence
              we have the extra logic to broadcast the checkpoint from a
              surviving node.
    """

    if any(os.path.exists(c) for c in ckpt_dirs.values()):
        state.load(ckpt_dirs, device_id)
        print("N[{}/{}] Checkpoint loaded.".format(device_id, rank))
    else:
        print("N[{}/{}] Checkpoint does not exist.".format(device_id, rank))

    if not distributed:
        return state

    # logic below is unnecessary when the checkpoint is visible on all nodes!
    # create a temporary cpu pg to broadcast most up-to-date checkpoint
    with tmp_process_group(backend="gloo") as pg:
        rank = dist.get_rank(group=pg)

        # get rank that has the largest state.epoch
        epochs = torch.zeros(dist.get_world_size(), dtype=torch.int32)
        epochs[rank] = state.epoch
        dist.all_reduce(epochs, op=dist.ReduceOp.SUM, group=pg)
        t_max_epoch, t_max_rank = torch.max(epochs, dim=0)
        max_epoch = t_max_epoch.item()
        max_rank = t_max_rank.item()

        # max_epoch == -1 means no one has checkpointed return base state
        # if max_epoch == -1:
        #     print("N[{}/{}] No Ckpt Found".format(device_id, rank))
        #     return state

        # broadcast the state from max_rank (which has the most up-to-date state)
        # pickle the snapshot, convert it into a byte-blob tensor
        # then broadcast it, unpickle it and apply the snapshot
        print("N[{}/{}] Restore Rank: {} , Epoch: {}".format(
            device_id, rank, max_rank, max_epoch))

        with io.BytesIO() as f:
            torch.save(state.capture_snapshot(), f)
            raw_blob = numpy.frombuffer(f.getvalue(), dtype=numpy.uint8).copy()

        blob_len = torch.tensor(len(raw_blob))
        dist.broadcast(blob_len, src=max_rank, group=pg)
        #print("N[{}/{}] Broadcast Size {}".format(device_id, rank, blob_len))

        if rank != max_rank:
            blob = torch.zeros(blob_len.item(), dtype=torch.uint8)
        else:

            blob = torch.as_tensor(raw_blob, dtype=torch.uint8)
            #blob = torch.as_tensor(numpy.array(raw_blob), dtype=torch.uint8)

        dist.broadcast(blob, src=max_rank, group=pg)
        #print("N[{}/{}] Broadcast Complete".format(device_id, rank))

        if rank != max_rank:
            with io.BytesIO(blob.numpy()) as f:
                snapshot = torch.load(f)
            state.apply_snapshot(snapshot, device_id)

        # wait till everyone has loaded the checkpoint
        dist.barrier(group=pg)

    #print("N[{}/{}] Ckpt Restore Complete".format(device_id, rank))
    return state


def save_checkpoint(state, is_best=False, is_latest=False, ckpt_dirs=None, epoch=None, stage=None):
    _save_checkpoint(state, is_best, is_latest, ckpt_dirs, epoch, stage)

def _save_checkpoint(state, is_best=False, is_latest=False, save_dir=None, epoch=None, stage=None):

    if save_dir is None and epoch is None:
        raise ValueError("Need to provide both save dir and epoch")
    elif is_latest:
        filename = os.path.join(save_dir, 'last_snapshot.pth')
    else:
        filename = os.path.join(save_dir, 'snapshot_{:03d}.pth'.format(epoch))

    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    tmp_filename = filename + ".tmp"
    torch.save(state.capture_snapshot(), tmp_filename)
    os.rename(tmp_filename, filename)

    #print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
    if is_best:

        best_name = "model_best.pth.tar" if stage is None else "{}_model_best.pth.tar".format(stage)
        best = os.path.join(save_dir, best_name)
        #print(f"=> best model found at epoch {state.epoch} saving to {best}")
        shutil.copyfile(filename, best)
