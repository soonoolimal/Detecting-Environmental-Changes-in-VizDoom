import numpy as np

import gymnasium as gym
import vizdoom as vzd


SCN_TO_CFG = dict(
    Basic="basic",
    BasicNotifications="basic_notifications",
    DeadlyCorridor="deadly_corridor",
    Deathmatch="deathmatch",
    DefendCenter="defend_the_center",
    DefendLine="defend_the_line",
    HealthGathering="health_gathering",
    HealthGatheringSupreme="health_gathering_supreme",
    MyWayHome="my_way_home",
    PredictPosition="predict_position",
    TakeCover="take_cover",
)


def get_env_info(env: gym.Env):
    ob_space = env.observation_space
    ac_space = env.action_space
    
    assert isinstance(ob_space, gym.spaces.Dict) and (
        ("screen" in ob_space.spaces) and ("gamevariables" in ob_space.spaces)
    )
    ob = ob_space.spaces["screen"].sample()
    st = ob_space.spaces["gamevariables"].sample()
    ob_shape, ob_dtype = ob.shape, ob.dtype
    st_shape, st_dtype = st.shape, st.dtype
    
    if isinstance(ac_space, gym.spaces.Dict) and (
        ("binary" in ac_space.spaces) and ("continuous" in ac_space.spaces)
    ):
        ac = ac_space.spaces["binary"].sample()
        ac_aux = ac_space.spaces["continuous"].sample()
        ac_dim, ac_dtype = np.atleast_1d(ac).shape[0], ac.dtype
        ac_aux_dim, ac_aux_dtype = ac_aux.shape[0], ac_aux.dtype
        n_actions = ac_dim  # continuous-binary: n_actions == ac_dim
    elif isinstance(ac_space, gym.spaces.Discrete):
        ac_dim, ac_dtype = 1, np.dtype("int64")  # storage shape: scalar per step
        ac_aux_dim, ac_aux_dtype = None, None
        n_actions = int(ac_space.n)              # number of discrete actions (for Embedding)
    else:
        ac = ac_space.sample()
        ac_dim, ac_dtype = np.atleast_1d(ac).shape[0], ac.dtype
        ac_aux_dim, ac_aux_dtype = None, None
        n_actions = ac_dim

    return ob_shape, ob_dtype, ac_dim, ac_dtype, st_shape, st_dtype, ac_aux_dim, ac_aux_dtype, n_actions


def annotate_progress_bar(game: vzd.DoomGame, t, rew, total_rew):
    def get_var(var, cast=float):
        try:
            return cast(game.get_game_variable(var))
        except Exception:
            return None

    hp = get_var(vzd.GameVariable.HEALTH)
    ammo = get_var(vzd.GameVariable.AMMO2, int)
    kill = get_var(vzd.GameVariable.KILLCOUNT, int)

    postfix = {
        "t": t,
        "r": f"{float(rew):.1f}",
        "R": f"{float(total_rew):.1f}",
        **({"HP": f"{hp:.1f}"} if hp is not None else {}),
        **({"AMMO": f"{ammo}"} if ammo is not None else {}),
        **({"KILL": f"{kill}"} if kill is not None else {}),
    }
    
    return postfix