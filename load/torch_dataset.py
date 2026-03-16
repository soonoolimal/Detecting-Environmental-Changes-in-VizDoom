import torch
from torch.utils.data import Dataset


class DTDataset(Dataset):
    """Dataset for Decision Transformer"""
    def __init__(
        self,
        observations,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        ep_len,
        num_episodes,
        seq_len,
    ):
        """
        Args:
            observations:  (N,C=3,H,W)
            actions:       (N,) or (N,ac_dim)
            rewards:       (N,)
            returns_to_go: (N,)
            timesteps:     (N,)
            ep_len:        timeout, fixed
            seq_len:       T
        """
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.returns_to_go = returns_to_go
        self.timesteps = timesteps
        
        total_envsteps = observations.shape[0]
        if total_envsteps != num_episodes * ep_len:
            raise ValueError(
                "Expected fixed segmentation, under the fixed episode length: "
                f"total_envsteps {total_envsteps} = num_episodes {num_episodes} * ep_len {ep_len}"
            )
        if seq_len > ep_len:
            raise ValueError(f"T={seq_len} must be <= ep_len={ep_len}.")
    
        self.num_episodes = num_episodes
        self.ep_len = ep_len
        self.seq_len = seq_len
        
    def __getitem__(self, idx):
        """
        Returns:
            dict: single input sequence of length T (window)
                - observations:  (T,C=3,H,W)
                - actions:       (T,)
                - rewards:       (T,1)
                - returns_to_go: (T,1)
                - timesteps:     (T,)
                - mask:          (T,), all ones (no padding)
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
            mask=torch.ones((self.seq_len,), dtype=torch.long),
        )
    
    def __len__(self):
        return self.num_episodes * (self.ep_len - self.seq_len + 1)


class TDDataset(Dataset):
    """Dataset Class for Task Detector"""
    def __init__(
        self,
        observations,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        ep_len,
        num_episodes,
        seq_len,
        labels,  # (N,) int64, task class label per timestep
    ):
        """
        Args:
            observations:  (N,C=3,H,W)
            actions:       (N,) or (N,ac_dim)
            rewards:       (N,)
            returns_to_go: (N,)
            timesteps:     (N,)
            ep_len:        timeout, fixed
            num_episodes:  number of episodes
            seq_len:       T
            labels:        (N,) int64, from H5Dataset.label (0: vanilla, 1: observation-shifted, 2: reward-shifted)
        """
        total_envsteps = observations.shape[0]
        if total_envsteps != num_episodes * ep_len:
            raise ValueError(
                "Expected fixed segmentation, under the fixed episode length: "
                f"total_envsteps {total_envsteps} = num_episodes {num_episodes} * ep_len {ep_len}"
            )
        if seq_len > ep_len:
            raise ValueError(f"T={seq_len} must be <= ep_len={ep_len}.")
        
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.returns_to_go = returns_to_go
        self.timesteps = timesteps
        self.ep_len = ep_len
        self.num_episodes = num_episodes
        self.seq_len = seq_len
        self.labels = labels
    
    def __len__(self):
        return self.num_episodes * (self.ep_len - self.seq_len + 1)

    def __getitem__(self, idx):
        """
        Returns:
            dict: single input sequence of length T (window)
                - observations:  (T,C=3,H,W)
                - actions:       (T,)
                - rewards:       (T,1)
                - returns_to_go: (T,1)
                - timesteps:     (T,)
                - mask:          (T,), all ones (no padding)
                - labels:        (T,), dataset-level class label broadcast to all timesteps
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


class AEDataset(Dataset):
    """Dataset Class for AutoEncoder"""
    def __init__(self, observations):  # (N,C=3,H,W)
        assert observations.ndim == 4
        self.observations = observations

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        x = self.observations[idx]
        return torch.from_numpy(x).float().div(255.0)  # (C,H,W) in [0,1], float32