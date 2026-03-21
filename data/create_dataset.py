import os
import argparse

import gymnasium as gym

from data.random_agent import RandomAgent


def add_args():
    p = argparse.ArgumentParser()

    # ID
    p.add_argument("--env", "--env_name", dest="env_name", type=str, default="DefendLine")
    p.add_argument("--exp", "--exp_name", dest="exp_name", type=str, required=True)

    p.add_argument("--van", "--vanilla", dest="shift_type", action="append_const", const="vanilla")
    p.add_argument("--ob", "--observation", dest="shift_type", action="append_const", const="observation")
    p.add_argument("--rew", "--reward", dest="shift_type", action="append_const", const="reward")

    p.add_argument("--survive", dest="rew_obj", action="append_const", const="survive")
    p.add_argument("--attack", dest="rew_obj", action="append_const", const="attack")
    p.add_argument("--move", dest="rew_obj", action="append_const", const="move")
    p.add_argument("--rew_all", dest="rew_obj_all", action="store_true", default=False)

    p.add_argument("--seed", type=int, required=True)

    # Game settings
    p.add_argument("--lv", "--level", dest="level", type=int, default=3)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--resize", type=int, default=84)
    p.add_argument("--skip", "--skipframe", dest="skipframe", type=int, default=5)

    # Save options
    p.add_argument("--n", "--num_episodes", dest="num_episodes", type=int, default=500)
    p.add_argument("--no_save_ob_as_uint8", dest="save_ob_as_uint8", action="store_false", default=True)
    p.add_argument("--vsf", "--video_save_freq", dest="video_save_freq", type=int, default=50)

    return p.set_defaults(
        shift_type=["vanilla", "observation", "reward"],
        rew_obj=None,
    )


if __name__ == "__main__":
    gym.logger.min_level = gym.logger.ERROR
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    args = add_args().parse_args()
    config = vars(args)

    shift_types = config.pop("shift_type") or []
    rew_objs = config.pop("rew_obj")
    rew_obj_all = config.pop("rew_obj_all")

    if rew_obj_all:
        rew_objs = ["survive", "attack", "move"]

    if "reward" in shift_types and not rew_objs:
        raise ValueError(
            "At least one of --survive, --attack, --move, or --rew_all must be specified with --rew."
        )

    instances = []
    for shift_type in shift_types:
        if shift_type in ["vanilla", "observation"]:
            instances.append((shift_type, None))
        elif shift_type == "reward":
            for rew_obj in rew_objs:
                instances.append(("reward", rew_obj))

    print("Create datasets...")

    for shift_type, rew_obj in instances:
        agent = RandomAgent(shift_type=shift_type, rew_obj=rew_obj, **config)
        env = agent.make_env()
        agent.run(env)

    print("Done")
