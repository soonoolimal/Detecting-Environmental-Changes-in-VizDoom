from typing import List

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from load.hdf5_utils import HDF5


class E2EDataset(Dataset):
    """Dataset for Training Decision Transformer"""
    def __init__(
        self,
        ds: HDF5,
        seq_len: int,  # length of input sequence, i.e., window size T (by episode)
    ):
        self.observations = ds.observations    # (N,C=3,H,W), uint8
        self.actions = ds.actions              # (N,) or (N,ac_dim)
        self.rewards = ds.rewards              # (N,)
        self.returns_to_go = ds.returns_to_go  # (N,)
        self.timesteps = ds.timesteps          # (N,)
        self.labels = ds.labels                # (N,) int64
        
        if seq_len > ds.ep_len:
            raise ValueError(
                f"Expected `seq_len` <= ds.ep_len, got seq_len={seq_len} and ds.ep_len={ds.ep_len}."
            )
        self.seq_len = seq_len
        self.ep_len = ds.ep_len
        self.num_episodes = ds.num_episodes
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: single input sequence of length T
                - observations  (T,C=3,H,W)
                - actions       (T,)
                - rewards       (T,1)
                - returns_to_go (T,1)
                - timesteps     (T,)
                - mask          (T,): all ones, i.e., no padding needed, since all trajectories have same length
                - labels        (T,)
        """
        num_windows = self.ep_len - self.seq_len + 1
        ep_start = (idx // num_windows) * self.ep_len
        wd_start = idx % num_windows
        sl = slice(ep_start + wd_start, ep_start + wd_start + self.seq_len)
        
        return dict(
            observations=torch.from_numpy(self.observations[sl]).float().div(255.0),
            actions=torch.from_numpy(self.actions[sl]).long(),
            rewards=torch.from_numpy(self.rewards[sl]).float().unsqueeze(-1),
            returns_to_go=torch.from_numpy(self.returns_to_go[sl]).float().unsqueeze(-1),
            timesteps=torch.from_numpy(self.timesteps[sl]).long(),
            mask=torch.ones(self.seq_len, dtype=torch.long),
            labels=torch.from_numpy(self.labels[sl]).long(),
        )
    
    def __len__(self):
        return self.num_episodes * (self.ep_len - self.seq_len + 1)


def make_dataloader(
    ds_list: List[HDF5],
    seq_len: int,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
):
    dataset = ConcatDataset([
        E2EDataset(
            ds=ds,
            seq_len=seq_len,
        )
        for ds in ds_list
    ])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
