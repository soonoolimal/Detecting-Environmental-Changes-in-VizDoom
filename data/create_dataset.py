import os
import argparse

import gymnasium as gym

from data.random_agent import RandomAgent


def add_args():
    p = argparse.ArgumentParser()
    
    # ID
    p.add_argument("--env", "--env_name", dest="env_name", type=str, default="DefendLine")
    p.add_argument("--exp", "--exp_name", dest="exp_name", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--shift", "--shift_type", dest="shift_type", type=str, choices=["vanilla", "observation", "reward", "all"], default="all")
    p.add_argument("--robj", "--rew_obj", dest="rew_obj", type=str, choices=["tanker", "hunter", "dodger", "all"], default=None)
    
    # Game settings
    p.add_argument("--lv", "--level", dest="level", type=int, default=3)
    p.add_argument("--n", "--num_episodes", dest="num_episodes", type=int, default=300)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--resize", type=int, default=84)
    p.add_argument("--frameskip", type=int, default=3)
    
    # Save options
    p.add_argument("--save_obs_as_uint8", dest="save_obs_as_uint8", action="store_true", default=True)
    p.add_argument("--no_save_obs_as_uint8", dest="save_obs_as_uint8", action="store_false")
    p.add_argument("--vsf", "--video_save_freq", dest="video_save_freq", type=int, default=30)

    return p


def resolve_shift_types(shift_type: str):
    return ["vanilla", "observation", "reward"] if shift_type == "all" else [shift_type]


def resolve_tasks(shift_types: list, rew_obj: str):
    # TODO: dodger
    # rew_objs = ["tanker", "hunter", "dodger"] if rew_obj == "all" else [rew_obj]
    rew_objs = ["tanker", "hunter"] if rew_obj == "all" else [rew_obj]
    tasks = []
    for shift in shift_types:
        if shift == "reward":
            if rew_obj is None:
                raise ValueError(
                    "shift_type 'reward' requires --robj to be specified."
                )
            for obj in rew_objs:
                tasks.append((shift, obj))
        else:
            tasks.append((shift, None))
    return tasks


if __name__ == "__main__":
    gym.logger.min_level = gym.logger.ERROR
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    args = add_args().parse_args()
    config = vars(args)

    shift_types = resolve_shift_types(config.pop("shift_type"))
    rew_obj = config.pop("rew_obj")
    tasks = resolve_tasks(shift_types, rew_obj)

    print("Sampling interactions...")
    
    for shift, obj in tasks:
        agent = RandomAgent(**config, shift_type=shift, rew_obj=obj)
        env = agent.make_env()
        agent.run(env)
    
    print("Done")
