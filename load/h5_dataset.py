import os
from dataclasses import dataclass
from typing import List, Tuple
import h5py
import numpy as np

from load import h5_utils as h5u


@dataclass
class H5Dataset:
    # E: num_episodes, N: num_episodes * timeout
    observations: np.ndarray         # (N,C=3,H,W), uint8
    # next_observations: np.ndarray  # (N,C=3,H,W)
    actions: np.ndarray              # (N,) or (N,ac_dim)
    rewards: np.ndarray              # (N,)
    # returns: np.ndarray            # (E+1,)
    returns_to_go: np.ndarray        # (N,)
    # terminals: np.ndarray          # (N,)
    done_idxs: np.ndarray            # (E,)
    timesteps: np.ndarray            # (N,)
    ep_len: int                      # timeout, fixed
    num_episodes: int
    ac_dim: int                      # storage shape per step (1 for Discrete)
    n_actions: int                   # number of actions, for Embedding (== ac_space.n for Discrete)
    label: np.ndarray                # (N,), task class label (0: vanilla, 1: observation, 2: reward)


def load_datasets(
    data_dir: str, env_name: str, exp_name: int, gamma: float = 1.0
) -> Tuple[H5Dataset, H5Dataset, H5Dataset]:
    ds_van = load_dataset(os.path.join(data_dir, f"{env_name}_vanilla_{exp_name}.hdf5"), label=0, gamma=gamma)
    ds_ob = load_dataset(os.path.join(data_dir, f"{env_name}_observation_{exp_name}.hdf5"), label=1, gamma=gamma)
    ds_rew = load_dataset(os.path.join(data_dir, f"{env_name}_reward_{exp_name}.hdf5"), label=2, gamma=gamma)
    
    return ds_van, ds_ob, ds_rew


def load_dataset(data_path: str, label: int, gamma: float = 1.0) -> H5Dataset:
    """
    Assume samed fixed length across all trajectories.
    """
    with h5py.File(data_path, "r") as hf:
        observations = hf["observations"][:]  # (N,H,W,C=3)  # type: ignore
        actions = hf["actions"][:]            # (N,1)        # type: ignore
        rewards = hf["rewards"][:]            # (N,)         # type: ignore
        
        num_episodes = hf.attrs.get("num_episodes").item()
        ep_len = hf.attrs.get("timeout").item()
        ac_dim = hf.attrs.get("ac_dim").item()
        n_actions = hf.attrs.get("n_actions").item()
    
    total_envsteps = rewards.shape[0]
    
    assert num_episodes * ep_len == total_envsteps, (
        f"Inconsistent sizes: num_episodes({num_episodes}) * ep_len({ep_len}) "
        f"!= total_steps({total_envsteps})"
    )
    
    # Termination Indicies: (E,)
    done_idxs = np.arange(ep_len, total_envsteps + 1, ep_len, dtype=np.int64)
    assert ep_len == done_idxs[0]
    
    # Observations: (N,C,H,W)
    if observations.ndim != 4:
        raise ValueError(f"Expected shape of observations (N,H,W,C=3), got {observations.shape}.")
    observations = np.transpose(observations, (0, 3, 1, 2))
    
    # Next Observations: (N,C,H,W), shifted by 1 & self-loop at terminal
    # next_observations = np.empty_like(observations)
    # next_observations[:-1] = observations[1:]
    # next_observations[done_idxs - 1] = observations[done_idxs - 1]
    # next_observations[-1] = observations[-1]
    
    # Discrete Actions: (N,) if (N,1)
    if actions.ndim == 2 and actions.shape[1] == 1:
        actions = np.squeeze(actions, axis=1)
    
    # Rewards: (N,)
    stepwise_returns = rewards.astype(np.float32, copy=False)
    
    # Returns: (E+1,) with trailing 0 slot
    # returns = np.concatenate([
    #     stepwise_returns.reshape(num_episodes, ep_len).sum(axis=1).astype(np.float32, copy=False),
    #     np.array([0.0], dtype=np.float32)
    # ], axis=0)
    
    # RTG: (N,), episode-wise discounted sum of rewards
    returns_to_go = h5u.compute_rtg(stepwise_returns, done_idxs, gamma)

    # Terminals: (N,), 1 at last transition per episode
    # terminals = np.zeros((total_envsteps,), dtype=np.uint8)
    # terminals[done_idxs - 1] = 1
    
    # Timesteps: (N,)
    timesteps = np.tile(np.arange(ep_len, dtype=np.int64), num_episodes)
    
    return H5Dataset(
        observations=observations,  # uint8, (N,C,H,W), float 변환은 Dataset __getitem__에서 수행
        actions=actions.astype(np.int64),
        rewards=stepwise_returns.astype(np.float32),
        returns_to_go=returns_to_go.astype(np.float32),
        done_idxs=done_idxs.astype(np.int64),
        timesteps=timesteps.astype(np.int64),
        ep_len=ep_len,
        num_episodes=len(done_idxs),
        ac_dim=ac_dim,
        n_actions=n_actions,
        label=np.full(total_envsteps, label, dtype=np.int64),
    )



def split_dataset(
    dataset: H5Dataset,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> Tuple[H5Dataset, H5Dataset, H5Dataset]:
    """
    Split H5Dataset into train, valid, and test dataset by episode.
    """
    E = dataset.num_episodes
    n_train = int(E * train_ratio)
    n_valid = int(E * valid_ratio)
    # n_test = E - n_train - n_valid

    if n_train == 0 or n_valid == 0 or (E - n_train - n_valid) == 0:
        raise ValueError(
            f"Not enough episodes to split: E={E}, "
            f"n_train={n_train}, n_valid={n_valid}, n_test={E - n_train - n_valid}. "
            f"Collect more episodes (recommend E >= 10)."
        )
    
    def _slice(ep_start: int, ep_end: int) -> H5Dataset:
        L = dataset.ep_len
        ts_start = ep_start * L
        ts_end = ep_end * L

        # done_idxs: re-index relative to slice start
        raw_done = dataset.done_idxs[ep_start:ep_end]  # absolute indices
        done_idxs = raw_done - ts_start                # relative to slice

        # timesteps: ep-relative, no offset needed
        timesteps = dataset.timesteps[ts_start:ts_end]

        return H5Dataset(
            observations = dataset.observations[ts_start:ts_end],
            actions = dataset.actions[ts_start:ts_end],
            rewards = dataset.rewards[ts_start:ts_end],
            returns_to_go = dataset.returns_to_go[ts_start:ts_end],
            done_idxs = done_idxs,
            timesteps = timesteps,
            ep_len = dataset.ep_len,
            num_episodes = ep_end - ep_start,
            ac_dim = dataset.ac_dim,
            n_actions = dataset.n_actions,
            label = dataset.label[ts_start:ts_end],
        )
    
    train = _slice(0, n_train)
    valid = _slice(n_train, n_train + n_valid)
    test = _slice(n_train + n_valid, E)
    
    return train, valid, test