"""LIBERO hdf5 dataset: random windows of (frames, actions) for world-model training."""

import glob
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LiberoDataset(Dataset):
    """Returns a window: obs (T, 3, H, W) float in [-1,1], actions (T-1, A)."""

    def __init__(self, data_dir, seq_len=17, camera="agentview_rgb"):
        self.files = sorted(glob.glob(f"{data_dir}/*.hdf5"))
        assert self.files, f"no hdf5 found in {data_dir}"
        self.seq_len = seq_len
        self.camera = camera
        self.episodes = []  # (file_idx, episode_key, length)
        for i, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                for key in f["data"]:
                    self.episodes.append((i, key, f["data"][key].attrs["num_samples"]))

    def __len__(self):
        return len(self.episodes) * 10  # arbitrary epoch size

    def __getitem__(self, idx):
        fi, key, length = self.episodes[idx % len(self.episodes)]
        start = random.randint(0, length - self.seq_len)
        end = start + self.seq_len
        with h5py.File(self.files[fi], "r") as f:
            ep = f["data"][key]
            obs = ep[f"obs/{self.camera}"][start:end]            # (T, H, W, 3) uint8
            act = ep["actions"][start:end - 1].astype(np.float32)  # (T-1, A)
        obs = torch.from_numpy(obs).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        return obs, torch.from_numpy(act)
