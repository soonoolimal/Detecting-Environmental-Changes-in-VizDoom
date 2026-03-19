from dataclasses import dataclass
from typing import Tuple, List
import h5py
from pathlib import Path
import numpy as np


@dataclass
class HDF5:
    # E: num_episodes, N = E * timeout
    observations: np.ndarray   # (N,C=3,H,W), uint8, to float in __getitem__()
    actions: np.ndarray        # discrete: (N,) continuous: (N,ac_dim)
    rewards: np.ndarray        # (N,)
    returns_to_go: np.ndarray  # (N,)
    done_idxs: np.ndarray      # (E,)
    timesteps: np.ndarray      # (N,) in [0,ep_len-1]
    ep_len: int                # fixed timeout
    num_episodes: int          # number of episodes
    ac_dim: int                # discrete: 1, continuous: d
    n_actions: int             # number of actions, for embedding
    labels: np.ndarray         # (N,), task class label, 0: vanilla, 1: observation-shifted, 2: reward-shifted


def load_and_split_datasets(
    env_name: str,
    exp_name: str,
    seed: int,
    rew_obj: str,
    gamma: float = 1.0,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.2,
) -> Tuple[List[HDF5], List[HDF5], List[HDF5]]:
    _data_dir = str(Path.cwd() / "data" / "datasets" / env_name / f"{exp_name}_s{seed}")
    
    ds_van = load_dataset(_data_dir + "_vanilla.hdf5", label=0, gamma=gamma)
    ds_ob = load_dataset(_data_dir + "_observation.hdf5", label=1, gamma=gamma)
    ds_rew = load_dataset(_data_dir + f"_reward_{rew_obj}.hdf5", label=2, gamma=gamma)
    
    van_train, van_valid, van_test = split_dataset(ds_van, train_ratio, valid_ratio)
    ob_train, ob_valid, ob_test = split_dataset(ds_ob, train_ratio, valid_ratio)
    rew_train, rew_valid, rew_test = split_dataset(ds_rew, train_ratio, valid_ratio)
    
    train = [van_train, ob_train, rew_train]
    valid = [van_valid, ob_valid, rew_valid]
    test = [van_test, ob_test, rew_test]
    
    return train, valid, test


def load_dataset(data_path: str, label: int, gamma: float = 1.0) -> HDF5:
    with h5py.File(data_path, "r") as hf:
        observations = hf["observations"][:]  # (N,H,W,C=3)          # type: ignore
        actions = hf["actions"][:]            # (N,1) or (N,ac_dim)  # type: ignore
        rewards = hf["rewards"][:]            # (N,)                 # type: ignore
        
        num_episodes = hf.attrs.get("num_episodes").item()
        ep_len = hf.attrs.get("timeout").item()  # assume same fixed length across all trajectories
        ac_dim = hf.attrs.get("ac_dim").item()
        n_actions = hf.attrs.get("n_actions").item()
        num_transitions = hf.attrs.get("num_transitions").item()
        
    total_envsteps = rewards.shape[0]
    if total_envsteps != num_transitions:
        raise ValueError(
            f"Expected total_envsteps = num_transitions, "
            f"got total_envsteps={total_envsteps} and num_episodes={num_episodes}."
        )
    elif total_envsteps != num_episodes * ep_len:
        raise ValueError(
            f"Expected total_envsteps = num_episodes * ep_len, "
            f"got total_envsteps={total_envsteps}, num_episodes={num_episodes}, and ep_len={ep_len}."
        )
    
    # Labels: (N,), all the same per timestep within single hdf5 file
    labels = np.full(total_envsteps, label, dtype=np.int64)
    
    # Termination Indices: (E,)
    done_idxs = np.arange(ep_len, total_envsteps + 1, ep_len, dtype=np.int64)
    if num_episodes != len(done_idxs):
        raise ValueError(
            f"Expected num_episodes = len(done_idxs), "
            f"got num_episodes={num_episodes} and len(done_idxs)={len(done_idxs)}."
        )
    elif ep_len != done_idxs[0]:
        raise ValueError(
            f"Expected ep_len = done_idxs[0], "
            f"got ep_len={ep_len} and done_idxs[0]={done_idxs[0]}."
        )
    
    # Observations: (N,C=3,H,W)
    if observations.ndim != 4:
        raise ValueError(
            f"Expected observations to have shape (N,H,W,C=3), got {observations.shape}."
        )
    observations = np.transpose(observations, (0, 3, 1, 2))
    
    # Actions: (N,) if (N,1)
    if actions.ndim == 2 and actions.shape[1] == 1:
        actions = np.squeeze(actions, axis=1)
    
    # Rewards: (N,)
    stepwise_returns = rewards.astype(np.float32, copy=False)
    
    # Returns-To-Go: (N,), episode-wise discounted sum of rewards
    returns_to_go = compute_rtg(stepwise_returns, done_idxs, gamma)
    
    # Timesteps: (N,)
    timesteps = np.tile(np.arange(ep_len, dtype=np.int64), num_episodes)
    
    return HDF5(
        observations=observations,
        actions=actions.astype(np.int64),
        rewards=stepwise_returns.astype(np.float32),
        returns_to_go=returns_to_go.astype(np.float32),
        done_idxs=done_idxs.astype(np.int64),
        timesteps=timesteps.astype(np.int64),
        ep_len=ep_len,
        num_episodes=num_episodes,
        ac_dim=ac_dim,
        n_actions=n_actions,
        labels=labels,
    )


def split_dataset(dataset: HDF5, train_ratio: float = 0.7, valid_ratio: float = 0.2) -> Tuple[HDF5, HDF5, HDF5]:
    """
    Split HDF5 into train, valid and test data by episode,
    despite of fixed episode length doesn't require padding.
    """
    n = dataset.num_episodes
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = (n - n_train - n_valid)
    
    if n_train * n_valid * n_test == 0:
        raise ValueError(
            f"Not enough episodes to split: num_episodes={n}, "
            f"n_train={n_train}, n_valid={n_valid}, n_test={n_test}."
            f"Collect more episodes (recommend at least 10)."
        )
    
    def _split(ep_start: int, ep_end: int) -> HDF5:
        ts_start = ep_start * dataset.ep_len
        ts_end = ep_end * dataset.ep_len
        
        # done_idxs: re-index relative to split start
        done_idxs = dataset.done_idxs[ep_start:ep_end] - ts_start
        
        # timesteps: ep-relative, no offset needed
        timesteps = dataset.timesteps[ts_start:ts_end]
        
        return HDF5(
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
            labels = dataset.labels[ts_start:ts_end],
        )

    train = _split(0, n_train)
    valid = _split(n_train, n_train + n_valid)
    test = _split(n_train + n_valid, n)
    
    return train, valid, test


def compute_rtg(stepwise_returns: np.ndarray, done_idxs: np.ndarray, gamma: float) -> np.ndarray:
    rtg = np.zeros_like(stepwise_returns, dtype=np.float32)
    start = 0
    for end in done_idxs:
        r = stepwise_returns[start:end]
        if np.isclose(gamma, 1.0, rtol=1e-09, atol=1e-09):
            rtg[start:end] = np.cumsum(r[::-1], axis=0)[::-1]
        else:
            out = np.empty_like(r, dtype=np.float32)
            running = 0.0
            for i in range(len(r) - 1, -1, -1):
                running = r[i] + gamma * running
                out[i] = running
            rtg[start:end] = out
        start = end

    return rtg
