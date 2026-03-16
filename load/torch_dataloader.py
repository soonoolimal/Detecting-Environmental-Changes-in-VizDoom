from typing import List
import numpy as np

from torch.utils.data import ConcatDataset, DataLoader

from load.h5_dataset import H5Dataset
from load.torch_dataset import DTDataset, TDDataset, AEDataset


def make_dt_dataloader(
    dataset: H5Dataset,
    seq_len: int,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
):
    ds = DTDataset(
        observations=dataset.observations,
        actions=dataset.actions,
        rewards=dataset.rewards,
        returns_to_go=dataset.returns_to_go,
        timesteps=dataset.timesteps,
        ep_len=dataset.ep_len,
        num_episodes=dataset.num_episodes,
        seq_len=seq_len,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def make_td_dataloader(
    datasets: List[H5Dataset],
    seq_len: int,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
):
    ds = ConcatDataset([
        TDDataset(
            observations=h5ds.observations,
            actions=h5ds.actions,
            rewards=h5ds.rewards,
            returns_to_go=h5ds.returns_to_go,
            timesteps=h5ds.timesteps,
            ep_len=h5ds.ep_len,
            num_episodes=h5ds.num_episodes,
            seq_len=seq_len,
            labels=h5ds.label,
        )
        for h5ds in datasets
    ])
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def make_ae_dataloader(
    observations: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
):
    ds = AEDataset(observations)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )