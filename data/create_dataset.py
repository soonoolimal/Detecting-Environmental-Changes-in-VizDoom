import os
import argparse
from typing import Optional

import gymnasium as gym

from data.random_agent import RandomAgent
from data.data_utils import SCN_TO_CFG


def add_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--scn_to_cfg", type=str, default="default")
    p.add_argument("-env", "--env_name", type=str, default="DefendLine")
    p.add_argument("-shift", "--shift_type", type=str, choices=["vanilla", "observation", "reward", "all"], default="all")
    p.add_argument("-exp", "--exp_name", type=str, default="default")
    
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-lv", "--level", type=int, default=3)
    p.add_argument("-n", "--num_episodes", type=int, default=3)
    p.add_argument("-tout", "--timeout", type=int, default=None)
    p.add_argument("--resize", type=int, default=84)
    
    p.add_argument("--save_obs_as_uint8", dest="save_obs_as_uint8", action="store_true", default=True)
    p.add_argument("--no_save_obs_as_uint8", dest="save_obs_as_uint8", action="store_false")
    
    p.add_argument("--video_save_freq", type=int, default=10)
    
    return p


def resolve_scn_cfg_map(key: str):
    if key == "default":
        return SCN_TO_CFG
    raise ValueError(f"Unknown --scn_cfg key: {key}")


def resolve_shift_types(shift_type: str):
    return ["vanilla", "observation", "reward"] if shift_type == "all" else [shift_type]


def resolve_timeout(env_name: str, timeout: Optional[int]):
    if timeout is not None:
        return timeout
    return 2100 if env_name == "DefendLine" else None


def run_random_agent(config: dict):
    config = dict(config)
    config["timeout"] = resolve_timeout(config["env_name"], config.get("timeout"))
    
    random_agent = RandomAgent(**config)
    env = random_agent.make_env()
    random_agent.run(env)


if __name__ == "__main__":
    gym.logger.min_level = gym.logger.ERROR
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    args = add_args().parse_args()
    config = vars(args)
    
    config["scn_to_cfg"] = resolve_scn_cfg_map(config.pop("scn_to_cfg"))
    
    print(f"[{config['env_name']}] Experiment: {config['exp_name']}")
    for shift in resolve_shift_types(config["shift_type"]):
        run_config = dict(config)
        run_config["shift_type"] = shift
        run_random_agent(run_config)
