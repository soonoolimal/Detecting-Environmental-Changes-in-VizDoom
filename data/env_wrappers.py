import cv2
import numpy as np

import gymnasium as gym
import vizdoom as vzd


class FrameSkip(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int):
        super().__init__(env)
        if skip < 1:
            raise ValueError(f"Expected `skip` >= 1, got {skip}.")
        self._skip = skip

    def step(self, action):
        next_ob_st, rew, done, trunc, info = self.env.step(action)
        total_rew = float(rew)
        
        for _ in range(self._skip - 1):
            next_ob_st, rew, done, trunc, info = self.env.step(action)
            total_rew += float(rew)  # TODO: sum of rewards during raw tics(skip), or only last? (ShiftReward: latter)
        
        return next_ob_st, total_rew, done, trunc, info


class ShiftReward(gym.Wrapper):
    def __init__(self, env: gym.Env, rew_obj: str):
        super().__init__(env)
                
        if rew_obj not in ["tanker", "hunter", "dodger"]:
            raise ValueError(
                f"Expected `rew_obj` to be one of ['longlive', 'hunter', 'dodger'], got '{rew_obj}'."
            )
        self.rew_obj = rew_obj

        self.hp_log = []
        self.ammo_log = []
        self.kill_log = []

        self.game: vzd.DoomGame = env.unwrapped.game
        
        self._button_list = [b.name for b in self.game.get_available_buttons()]

    def reset(self, **kwargs):
        ob_st, info = self.env.reset(**kwargs)

        self.hp_log.clear()
        self.ammo_log.clear()
        self.kill_log.clear()

        _init_hp = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        _init_ammo = self.game.get_game_variable(vzd.GameVariable.AMMO2)
        _init_kill = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)

        self.hp_log.append(_init_hp)
        self.ammo_log.append(_init_ammo)
        self.kill_log.append(_init_kill)

        return ob_st, info

    def step(self, action):
        # ac_button = self._get_button(action)
        next_ob_st, rew, done, trunc, info = self.env.step(action)

        now_hp = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        now_ammo = self.game.get_game_variable(vzd.GameVariable.AMMO2)
        now_kill = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        
        # Change Entire Objective
        penalty = False
        # objective 1: long live the agent
        if self.rew_obj == "tanker":
            penalty = now_hp < 99900
        # objective 2: kill monsters
        elif self.rew_obj == "hunter":
            penalty = now_kill == self.kill_log[-1]
        # TODO: dodger
        # objective 3: evade attacks
        # elif self.rew_obj == "dodger":
        #     penalty = now_hp < self.hp_log[-1]
        rew = -1.0 if penalty else 0.0
        
        self.hp_log.append(now_hp)
        self.ammo_log.append(now_ammo)
        self.kill_log.append(now_kill)

        return next_ob_st, rew, done, trunc, info

    def _get_button(self, action):
        button_idx = action["binary"].item() if isinstance(action, dict) else action.item()
        if button_idx == 0:
            return "NOOP"
        return self._button_list[button_idx - 1]


class ResizeObservation(gym.Wrapper):
    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)

        self._size = int(size)

        ob_space = env.observation_space
        screen_space = ob_space["screen"]  # type: ignore
        _, _, C = screen_space.shape       # (H=240,W=320,C=3)

        new_space = dict(ob_space.spaces)
        new_space["screen"] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._size, self._size, C),
            dtype=screen_space.dtype,
        )
        self.observation_space = gym.spaces.Dict(new_space)

    def _resize_ob(self, ob_st: dict) -> dict:
        screen = ob_st["screen"]  # (H,W,C)
        resized = cv2.resize(screen, (self._size, self._size), interpolation=cv2.INTER_AREA)
        return {**ob_st, "screen": resized}

    def reset(self, **kwargs):
        ob_st, info = self.env.reset(**kwargs)
        return self._resize_ob(ob_st), info

    def step(self, action):
        next_ob_st, rew, done, trunc, info = self.env.step(action)
        return self._resize_ob(next_ob_st), rew, done, trunc, info


class AnnotateVideo(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        font_scale: float = 0.3,
        thickness: int = 1,
        line_height: int = 22,
    ):
        super().__init__(env)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = float(font_scale)
        self.thickness = int(thickness)
        self.line_height = int(line_height)
        
        self._t = 0
        self._status = None
        self._rew = 0.0
        self._total_rew = 0.0
        self._cached_lines = []

        self.game: vzd.DoomGame = env.unwrapped.game
        
        self._status_list = [v.name for v in self.game.get_available_game_variables()]
        self._button_list = [b.name for b in self.game.get_available_buttons()]

    def reset(self, **kwargs):
        ob_st, info = self.env.reset(**kwargs)
        
        self._t = 0
        self._rew = 0.0
        self._total_rew = 0.0
        self._cached_lines = []
        self._status = self._get_status(ob_st)
        
        return ob_st, info

    def step(self, action):
        _ac_button = self._get_button(action)
        next_ob_st, rew, done, trunc, info = self.env.step(action)
        _cur_status = self._get_status(next_ob_st)

        self._t += 1
        self._status.update(_cur_status)
        self._rew = float(rew)
        self._total_rew += float(rew)
        self._cached_lines = [
            f"t={self._t}",
            *[f"  {k}: {v:.0f}" for k, v in self._status.items()],
            f"  KILL: {self.game.get_game_variable(vzd.GameVariable.KILLCOUNT):.0f}",
            f"action:{_ac_button}",
            f"r={self._rew:.0f}",
            f"R={self._total_rew:.0f}",
        ]

        return next_ob_st, rew, done, trunc, info

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None

        rgb = np.asarray(frame)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        x, y = 10, 30
        for line in self._cached_lines:
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

    def _get_status(self, ob_st) -> dict:
        return dict(zip(self._status_list, ob_st["gamevariables"].tolist()))

    def _get_button(self, action):
        button_idx = action["binary"].item() if isinstance(action, dict) else action.item()
        if button_idx == 0:
            return "NOOP"
        return self._button_list[button_idx - 1]
