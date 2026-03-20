import random
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
        seed: int,
        shift_type: str,
        rew_obj: str = None,
        level: int = 3,
        num_episodes: int = 200,
        timeout: int = None,
        resize: int = 84,
        frameskip: int = 3,
        save_obs_as_uint8: bool = True,
        video_save_freq: int = 20,
    ):
        self.env_name = env_name
        self.exp_name = exp_name
        self.seed = seed
        if shift_type in ["vanilla", "observation"]:
            self.id = f"{exp_name}_s{seed}_{shift_type}"
        elif shift_type == "reward":
            if rew_obj is None:
                raise ValueError(
                    f"Expected `rew_obj` to be one of ['longlive', 'hunter', 'dodger'], got '{rew_obj}'."
                )
            self.id = f"{exp_name}_s{seed}_{shift_type}_{rew_obj}"
        else:
            raise ValueError(
                f"Expected `shift_type` to be one of ['vanilla', 'observation', 'reward'], got '{shift_type}'."
            )
        self.shift_type = shift_type
        self.rew_obj = rew_obj
        
        self.doom_skill = level
        self.num_episodes = num_episodes
        self.timeout = timeout
        self.resize = resize
        self.frameskip = frameskip
        
        self.save_obs_as_uint8 = save_obs_as_uint8
        self.video_save_freq = video_save_freq
        if video_save_freq == -1:   # disable video saving
            self.episode_trigger = lambda ep: False
        elif video_save_freq == 0:  # save only very first episode as video
            self.episode_trigger = lambda ep: ep == 0
        elif video_save_freq == 1:  # save all episodes as videos
            self.episode_trigger = lambda ep: True
        else:
            self.episode_trigger = lambda ep: ep % video_save_freq == 0
        
        self.scn_to_cfg = dtu.SCN_TO_CFG
        
        _data_dir = Path.cwd() / "data"
        self.custom_scn_cfg_path = _data_dir / "custom" / self.scn_to_cfg[env_name]
        self.data_save_dir = _data_dir / "datasets" / env_name
        self.video_save_dir = self.data_save_dir / "videos"
        
    def make_env(self) -> gym.Env:
        env_id = "Vizdoom" + self.env_name + "-v1"
        env_kwargs = dict(render_mode="rgb_array")
        
        # Observation-Shifted task
        if self.shift_type == "observation":
            env_kwargs["scenario_config_file"] = str(self.custom_scn_cfg_path) + "_ob_shift.cfg"
        # Vanila task
        else:
            env_kwargs["scenario_config_file"] = str(self.custom_scn_cfg_path) + ".cfg"
        
        env = gym.make(env_id, **env_kwargs)

        # Skip frame
        if self.frameskip > 1:
            env = wrappers.FrameSkip(env, skip=self.frameskip)

        # Reward-Shifted task
        if self.shift_type == "reward":
            env = wrappers.ShiftReward(env, self.rew_obj)
        
        # Set doom skill
        env.unwrapped.game.set_doom_skill(self.doom_skill)
        
        # Resize observation
        env = wrappers.ResizeObservation(env, self.resize)
        
        # Record (optional)
        if self.video_save_freq != -1:
            video_folder = self.video_save_dir / self.id
            if video_folder.exists():
                raise FileExistsError(
                    f"Video folder already exists: {video_folder}"
                )
            video_folder.mkdir(parents=True)
            env = RecordVideo(
                wrappers.AnnotateVideo(env),
                video_folder=str(video_folder),
                name_prefix=self.id,
                episode_trigger=self.episode_trigger,
                fps=35 / self.frameskip,
            )
        
        return env
    
    def run(self, env: gym.Env):
        spec = env.unwrapped.spec
        if spec is None or not spec.name.endswith(self.env_name):
            raise ValueError(
                f"Expected `env_name` to be one of {list(self.scn_to_cfg.keys())}, got {self.env_name}."
            )
        
        # Set game
        game: vzd.DoomGame = env.unwrapped.game
        
        if self.timeout is None:
            if game.get_episode_timeout() == 0:
                raise ValueError(
                    f"Scenario {self.env_name} has no internal episode_timeout. Pass 'timeout' manually."
                )
            else:
                raw_timeout = game.get_episode_timeout()
        else:
            raw_timeout = self.timeout
        game.set_episode_timeout(raw_timeout)
        
        max_envsteps = raw_timeout // self.frameskip
            
        # Get environment information
        ob_shape, _, ac_dim, ac_dtype, _, _, ac_aux_dim, ac_aux_dtype, n_actions = dtu.get_env_info(env)
        ob_store_dtype = np.dtype("uint8") if self.save_obs_as_uint8 else np.dtype("float32")
        
        # Chunk HDF5
        self.data_save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.data_save_dir / (self.id + ".hdf5")
        if save_path.exists():
            raise FileExistsError(
                f"Dataset file already exists: {save_path}"
            )
        
        total_envsteps = int(self.num_episodes * max_envsteps)
        h5_chunksteps = int(min(max(256, max_envsteps), 8192))
        h5_kwargs = dict(
            compression="gzip",
            shuffle=True,
        )

        with h5py.File(save_path, "w") as hf:
            # datasets
            ds_ob = hf.create_dataset(
                "observations",
                shape=(total_envsteps, *ob_shape),
                dtype=ob_store_dtype,
                chunks=(h5_chunksteps, *ob_shape),
                **h5_kwargs,
            )
            ds_ac = hf.create_dataset(
                "actions",
                shape=(total_envsteps, ac_dim),
                dtype=ac_dtype,
                chunks=(h5_chunksteps, ac_dim),
                **h5_kwargs,
            )
            ds_rew = hf.create_dataset(
                "rewards",
                shape=(total_envsteps,),
                dtype=np.float32,
                chunks=(h5_chunksteps,),
                **h5_kwargs,
            )
            ds_tout = hf.create_dataset(
                "timeouts",
                shape=(total_envsteps,),
                dtype=np.float32,
                chunks=(h5_chunksteps,),
                **h5_kwargs,
            )
            if ac_aux_dim is not None:
                ds_ac_aux = hf.create_dataset(
                    "actions_auxiliary",
                    shape=(total_envsteps, ac_aux_dim),
                    dtype=ac_aux_dtype,
                    chunks=(h5_chunksteps, ac_aux_dim),
                    **h5_kwargs,
                )
            else:
                ds_ac_aux = None
            
            # attributes
            hf.attrs["env_name"] = self.env_name
            hf.attrs["shift_type"] = self.shift_type
            hf.attrs["exp_name"] = self.exp_name
            
            hf.attrs["timeout"] = max_envsteps
            hf.attrs["num_episodes"] = self.num_episodes
            
            hf.attrs["ac_dim"] = ac_dim
            hf.attrs["n_actions"] = n_actions
            
            # per-episode RAM buffers
            buf_ob = np.empty((max_envsteps, *ob_shape), dtype=ob_store_dtype)
            buf_ac = np.empty((max_envsteps, ac_dim), dtype=ac_dtype)
            buf_rew = np.empty((max_envsteps,), dtype=np.float32)
            buf_tout = np.empty((max_envsteps,), dtype=np.float32)
            if ac_aux_dim is not None:
                buf_ac_aux = np.empty((max_envsteps, ac_aux_dim), dtype=ac_aux_dtype)
            else:
                buf_ac_aux = None
            
            # run
            rng = np.random.default_rng(self.seed)
            random.seed(self.seed)
            
            from_idx = 0
            for ep in range(self.num_episodes):
                ob_st, _ = env.reset(seed=int(rng.integers(0, 2**31-1)))
                ob = ob_st["screen"]
                total_rew = 0.0

                pbar = tqdm(
                    total=max_envsteps,
                    desc=f"({self.env_name}) [{self.id}] Ep {ep}",
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
                    
                    if done and (t != max_envsteps - 1):
                        raise RuntimeError(
                            "Episode terminated early. Check cheat mode is activated."
                        )
                    elif trunc and (t != max_envsteps - 1):
                        raise RuntimeError(
                            "Episode reached timeout early. Check cheat mode is activated."
                        )
                    
                    # write into buffers
                    if self.save_obs_as_uint8:
                        buf_ob[t] = ob.astype(np.uint8, copy=False)
                    else:
                        buf_ob[t] = ob.astype(np.float32, copy=False)
                        
                    if buf_ac_aux is not None:
                        buf_ac[t] = ac["binary"].item()
                        buf_ac_aux[t] = ac["continuous"]
                    else:
                        buf_ac[t] = ac
                    
                    buf_rew[t] = float(rew)
                    buf_tout[t] = 1.0 if bool(trunc) else 0.0
                    
                    ob = next_ob
                
                # flush episode
                to_idx = from_idx + max_envsteps
                
                if to_idx > total_envsteps:
                    raise RuntimeError(
                        f"Exceeded preallocated 'total_envsteps' {total_envsteps}, need {to_idx}."
                    )
                
                ds_ob[from_idx:to_idx] = buf_ob
                ds_ac[from_idx:to_idx] = buf_ac
                ds_rew[from_idx:to_idx] = buf_rew
                ds_tout[from_idx:to_idx] = buf_tout
                if buf_ac_aux is not None:
                    ds_ac_aux[from_idx:to_idx] = buf_ac_aux

                pbar.close()
                
                from_idx = to_idx
            
            assert from_idx == total_envsteps
            hf.attrs["num_transitions"] = int(from_idx)
            
            env.close()
        
        print(f"[SAVED] {save_path}")
