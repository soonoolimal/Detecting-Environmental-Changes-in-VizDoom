import warnings
import cv2
import numpy as np

import gymnasium as gym
import vizdoom as vzd


class SkipFrame(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int):
        super().__init__(env)
        
        if skip < 1:
            raise ValueError(f"`skip` must be >= 1 (got {skip})")
        self.skip = skip

    def step(self, action):
        for _ in range(self.skip):
            next_ob_st, rew, done, trunc, info = self.env.step(action)
        
        return next_ob_st, rew, done, trunc, info  # type: ignore


class ShiftReward(gym.Wrapper):
    def __init__(self, env: gym.Env, rew_obj: str):
        super().__init__(env)
        
        self.game: vzd.DoomGame = env.unwrapped.game
                
        if rew_obj not in ["survive", "attack", "move"]:
            raise ValueError(f"`rew_obj` must be one of: 'survive', 'attack', 'move' (got '{rew_obj}')")
        self.rew_obj = rew_obj

        self.hp_log = []
        self.ammo_log = []
        self.kill_log = []

    def reset(self, **kwargs):
        ob_st, info = self.env.reset(**kwargs)

        self.hp_log.clear()
        self.ammo_log.clear()
        self.kill_log.clear()
        
        init_hp = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        init_ammo = self.game.get_game_variable(vzd.GameVariable.AMMO2)
        init_kill = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        
        self.hp_log.append(init_hp)
        self.ammo_log.append(init_ammo)
        self.kill_log.append(init_kill)

        return ob_st, info

    def step(self, action):
        next_ob_st, rew, done, trunc, info = self.env.step(action)

        now_hp = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        now_ammo = self.game.get_game_variable(vzd.GameVariable.AMMO2)
        now_kill = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        
        bonus = False
        if self.rew_obj == "survive":
            bonus = (now_hp > 99900)
        elif self.rew_obj == "attack":
            bonus = (now_kill > self.kill_log[-1])
        elif self.rew_obj == "move":
            bonus = (now_ammo == self.ammo_log[-1]) and (now_hp == self.hp_log[-1])
        rew = 1.0 if bonus else 0.0
        
        self.hp_log.append(now_hp)
        self.ammo_log.append(now_ammo)
        self.kill_log.append(now_kill)

        return next_ob_st, rew, done, trunc, info


class AnnotateVideo(gym.Wrapper):
    def __init__(self, env: gym.Env, font_scale: float = 0.3, thickness: int = 1, line_height: int = 22):
        super().__init__(env)
        
        self.game: vzd.DoomGame = env.unwrapped.game

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = float(font_scale)
        self.thickness = int(thickness)
        self.line_height = int(line_height)
        
        self.t = 0
        self.status = None
        self.rew = 0.0
        self.total_rew = 0.0
        self.cached_lines = []

        self.status_list = [v.name for v in self.game.get_available_game_variables()]
        self.button_list = [b.name for b in self.game.get_available_buttons()]

    def reset(self, **kwargs):
        ob_st, info = self.env.reset(**kwargs)
        
        self.t = 0
        self.rew = 0.0
        self.total_rew = 0.0
        self.cached_lines = []
        self.status = self.get_status(ob_st)
        
        return ob_st, info

    def step(self, action):
        ac_button = self.get_button(action)
        next_ob_st, rew, done, trunc, info = self.env.step(action)
        cur_status = self.get_status(next_ob_st)

        self.t += 1
        self.status.update(cur_status)
        self.rew = float(rew)
        self.total_rew += float(rew)
        self.cached_lines = [
            f"t={self.t}",
            f"action:{ac_button}",
            *[f"\t{k}: {v:.0f}" for k, v in self.status.items()],
            f"r={self.rew:.0f}",
            f"R={self.total_rew:.0f}",
        ]

        return next_ob_st, rew, done, trunc, info

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None

        rgb = np.asarray(frame)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        x, y = 10, 30
        for line in self.cached_lines:
            cv2.putText(
                bgr,
                line,
                (x, y),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.thickness,
                cv2.LINE_AA,
            )
            y += self.line_height

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_status(self, ob_st) -> dict:
        status = dict(zip(self.status_list, ob_st["gamevariables"].tolist()))
        if "KILLCOUNT" not in status.keys():
            status["KILLCOUNT"] = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        return status

    def get_button(self, action):
        button_idx = action["binary"].item() if isinstance(action, dict) else action.item()
        if button_idx == 0:
            return "NOOP"
        return self.button_list[button_idx - 1]


class ResizeObservation(gym.Wrapper):
    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)

        self.size = int(size)

        screen_space = env.observation_space.spaces["screen"]
        orig_shape = screen_space.shape
        orig_dtype = screen_space.dtype

        H, W, C = orig_shape
        if C != 3:
            raise ValueError(f"Expected 3 RGB channels, got {C} channel(s)")
        if (H, W) != (240, 320):
            warnings.warn(
                f"Expected default resolution (240, 320), got ({H}, {W})", UserWarning
            )

        new_space = dict(env.observation_space.spaces)
        new_space["screen"] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.size, self.size, C),
            dtype=orig_dtype,
        )
        self.observation_space = gym.spaces.Dict(new_space)

    def reset(self, **kwargs):
        ob_st, info = self.env.reset(**kwargs)
        return self.resize_ob(ob_st), info

    def step(self, action):
        next_ob_st, rew, done, trunc, info = self.env.step(action)
        return self.resize_ob(next_ob_st), rew, done, trunc, info
    
    def resize_ob(self, ob_st: dict) -> dict:
        screen = ob_st["screen"]
        resized = cv2.resize(screen, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return {**ob_st, "screen": resized}