import random
import warnings
from typing import Optional
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import vizdoom as vzd
from vizdoom import gymnasium_wrapper

from data import env_wrappers as wrappers
from data import data_utils as dtu


class RandomAgent:
    def __init__(
        self,
        env_name: str,
        exp_name: str,
        shift_type: str,
        rew_obj: Optional[str],
        seed: int,
        num_episodes: int,
        level: int = 3,
        timeout: int = None,
        resize: int = 84,
        skipframe: int = 3,
        save_ob_as_uint8: bool = True,
        video_save_freq: int = 20,
    ):
        # ID
        if env_name not in dtu.SCN_TO_CFG.keys():
            choices = ", ".join(f"'{k}'" for k in dtu.SCN_TO_CFG)
            raise ValueError(
                f"`env_name` must be one of: {choices} (got '{env_name}')"
            )
        if shift_type in ["vanilla", "observation"]:
            if rew_obj is not None:
                warnings.warn(
                    f"`rew_obj` is ignored when `shift_type` is '{shift_type}' (got '{rew_obj}')", UserWarning)
                rew_obj = None
            self.id = f"{exp_name}_{shift_type}_s{seed}"
        elif shift_type == "reward":
            if rew_obj not in ["survive", "attack", "move"]:
                raise ValueError(
                    f"`rew_obj` must be one of: 'survive', 'attack', 'move' when `shift_type` is 'reward' "
                    f"(got '{rew_obj}')"
                )
            self.id = f"{exp_name}_reward_{rew_obj}_s{seed}"
        else:
            raise ValueError(
                f"`shift_type` must be one of: 'vanilla', 'observation', 'reward' (got '{shift_type}')"
            )
        self.env_name = env_name
        self.exp_name = exp_name
        self.shift_type = shift_type
        self.rew_obj = rew_obj
        self.seed = seed

        # Game settings
        self.doom_skill = level
        self.timeout = timeout
        self.resize = resize
        self.skipframe = skipframe

        # Save options
        self.num_episodes = num_episodes
        self.save_ob_as_uint8 = save_ob_as_uint8
        self.video_save_freq = video_save_freq

        # Directories
        _data_dir = Path.cwd() / "data"
        self.data_save_dir = _data_dir / "datasets" / env_name
        self.video_save_dir = _data_dir / "videos" / env_name
        self.custom_scn_cfg_path = _data_dir / "custom" / dtu.SCN_TO_CFG[env_name]

    def make_env(self) -> gym.Env:
        env_id = "Vizdoom" + self.env_name + "-v1"
        env_kwargs = dict(render_mode="rgb_array")

        if self.shift_type == "observation":
            env_kwargs["scenario_config_file"] = str(self.custom_scn_cfg_path) + "_ob_shift.cfg"
        else:
            env_kwargs["scenario_config_file"] = str(self.custom_scn_cfg_path) + ".cfg"

        env = gym.make(env_id, **env_kwargs)

        # Wrapper 1: skip frame
        if self.skipframe > 1:
            env = wrappers.SkipFrame(env, skip=self.skipframe)

        # Wrapper 2: change reward function
        if self.shift_type == "reward":
            env = wrappers.ShiftReward(env, self.rew_obj)

        # Wrapper 3: annotate video (original resolution)
        # Wrapper 4: resize observation
        # Wrapper 5: record video
        if self.video_save_freq == -1:
            env = wrappers.ResizeObservation(env, self.resize)
            return env
        else:
            env = wrappers.AnnotateVideo(env)
            env = wrappers.ResizeObservation(env, self.resize)

            video_folder = self.video_save_dir / self.id
            video_folder.mkdir(parents=True, exist_ok=False)
            if self.video_save_freq == 0:
                episode_trigger = lambda ep: ep == 0
            elif self.video_save_freq == 1:
                episode_trigger = lambda ep: True
            else:
                episode_trigger = lambda ep: ep % self.video_save_freq == 0
            fps = 35 // self.skipframe

            env = RecordVideo(
                env,
                video_folder=str(video_folder),
                name_prefix=self.id,
                episode_trigger=episode_trigger,
                fps=fps,
            )

            return env

    def run(self, env: gym.Env):
        game: vzd.DoomGame = env.unwrapped.game

        # Game setting: difficulty
        game.set_doom_skill(self.doom_skill)

        # Game setting: timeout
        # hardcode WAD to set HP and AMMO to very large values so that the player is effectively invincible,
        # while still preserving trackability of HP and AMMO (unlike using cheat commands)
        # therefore reaching the timeout becomes the only termination condition
        # i.e., all episodes have the same fixed length determined by the timeout and skipframe
        if self.timeout is None:
            if game.get_episode_timeout() == 0:
                raise ValueError(
                    f"Scenario {self.env_name} has no internal `episode_timeout`. Pass `timeout` manually"
                )
            else:
                raw_timeout = game.get_episode_timeout()
        else:
            raw_timeout = self.timeout
        game.set_episode_timeout(raw_timeout)

        max_envsteps = raw_timeout // self.skipframe
        total_envsteps = int(self.num_episodes * max_envsteps)

        # Get environment information
        ob_shape, _, ac_dim, ac_dtype, _, _, ac_aux_dim, ac_aux_dtype, ac_store_dim = dtu.get_env_info(env)
        ob_store_dtype = np.dtype("uint8") if self.save_ob_as_uint8 else np.dtype("float32")

        # Chunk HDF5
        self.data_save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.data_save_dir / (self.id + ".hdf5")
        if save_path.exists():
            raise FileExistsError(f"Dataset file already exists: {save_path}")

        with h5py.File(save_path, "w") as hf:
            def create_ds(name, tail_shape, dtype):
                shape = (total_envsteps, *tail_shape)
                h5_chunksteps = int(min(max(256, max_envsteps), 8192))
                return hf.create_dataset(
                    name,
                    shape=shape,
                    dtype=dtype,
                    chunks=(h5_chunksteps, *tail_shape),
                    compression="gzip",
                    shuffle=True,
                )

            ds_ob = create_ds("observations", ob_shape, ob_store_dtype)
            ds_ac = create_ds("actions", (ac_store_dim,), ac_dtype)
            ds_rew = create_ds("rewards", (), np.float32)
            ds_tout = create_ds("timeouts", (), np.float32)
            if ac_aux_dim is not None:
                ds_ac_aux = create_ds("actions_auxiliary", (ac_aux_dim,), ac_aux_dtype)
            else:
                ds_ac_aux = None

            # attributes
            hf.attrs["timeout"] = max_envsteps
            hf.attrs["num_episodes"] = self.num_episodes
            hf.attrs["ac_dim"] = ac_dim

            # per-episode RAM buffers
            buf_ob = np.empty((max_envsteps, *ob_shape), dtype=ob_store_dtype)
            buf_ac = np.empty((max_envsteps, ac_store_dim), dtype=ac_dtype)
            buf_rew = np.empty((max_envsteps,), dtype=np.float32)
            buf_tout = np.empty((max_envsteps,), dtype=np.float32)
            if ds_ac_aux is not None:
                buf_ac_aux = np.empty((max_envsteps, ac_aux_dim), dtype=ac_aux_dtype)
            else:
                buf_ac_aux = None

            # run
            rng = np.random.default_rng(self.seed)
            random.seed(self.seed)

            from_idx = 0
            for ep in range(self.num_episodes):
                ob_st, _ = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
                ob = ob_st["screen"]
                total_rew = 0.0

                pbar = tqdm(
                    total=max_envsteps,
                    desc=f"[{self.env_name}/{self.id}] Ep {ep}",
                    dynamic_ncols=True,
                    leave=True,
                )
                for t in range(max_envsteps):
                    ac = env.action_space.sample()
                    next_ob_st, rew, done, trunc, _ = env.step(ac)
                    next_ob = next_ob_st["screen"]
                    total_rew += float(rew)

                    postfix = dtu.annotate_progress_bar(game, t, rew, total_rew)
                    pbar.set_postfix(postfix)
                    pbar.update(1)

                    if (done or trunc) and (t != max_envsteps - 1):
                        raise RuntimeError(
                            f"Episode terminated before timeout={max_envsteps}. Check that cheat mode is activated"
                        )

                    # write into buffers
                    if self.save_ob_as_uint8:
                        buf_ob[t] = ob.astype(np.uint8, copy=False)
                    else:
                        buf_ob[t] = ob.astype(np.float32, copy=False)

                    if buf_ac_aux is not None:
                        buf_ac[t] = np.atleast_1d(ac["binary"])
                        buf_ac_aux[t] = ac["continuous"]
                    else:
                        buf_ac[t] = np.atleast_1d(ac)

                    buf_rew[t] = float(rew)
                    buf_tout[t] = 1.0 if bool(trunc) else 0.0

                    ob = next_ob

                # flush episode
                to_idx = from_idx + max_envsteps

                if to_idx > total_envsteps:
                    raise RuntimeError(
                        f"Required {to_idx} steps, but only {total_envsteps} were preallocated"
                    )

                ds_ob[from_idx:to_idx] = buf_ob
                ds_ac[from_idx:to_idx] = buf_ac
                ds_rew[from_idx:to_idx] = buf_rew
                ds_tout[from_idx:to_idx] = buf_tout
                if buf_ac_aux is not None:
                    ds_ac_aux[from_idx:to_idx] = buf_ac_aux

                pbar.close()

                from_idx = to_idx

            if from_idx != total_envsteps:
                raise RuntimeError

            hf.attrs["num_transitions"] = int(from_idx)

            env.close()

        print(f"[SAVED] {save_path}")
