"""Mode-driven tag game on an open grid (ECE 595 final report, Sec. IV).

World     size x size open grid (default 10 x 10 = 100 cells), cell = y * size + x.
State     s_t = (x_t, y_t, m_t): robot cell, opponent cell, opponent mode
          m in {up, down, left, right}.  100 * 100 * 4 = 40,000 states.
Actions   {up, down, left, right}. Moving into the boundary = stay.
Opponent  ignores the robot. With prob p_o it moves in the direction of its mode m_t;
          otherwise it stays or moves in one of the other three directions, each with
          prob (1 - p_o) / 4. Moving into the boundary = stay.
Mode      m_{t+1} = m_t with prob p_m, else one of the other three modes uniformly.
          (Or a deterministic schedule m(t) via ``mode_schedule``.)
Capture   checked after both agents move: if x_{t+1} = y_{t+1} the robot gets
          reward_capture (+10) and the episode terminates.
Reward    otherwise "sparse": reward_step (-1); "dense": -d(x_{t+1}, y_{t+1}).
Observe   o_t = (x_t, y^o_t), y^o_t = y_t if the opponent is visible, else hidden (-1,
          encoded as index ``n_cells``). Visibility:
            "row_col" (experiment A): same row or same column;
            "radius"  (experiment B): d(x_t, y_t) <= eps(t). eps(t) is never observed.
Start     robot and opponent uniform and independent, re-drawn until they differ;
          mode uniform.
Horizon   max_steps = 100, then truncated.

p_o, p_m and eps accept a number, a Schedule (tag_env.schedules) or a callable
fn(t, rng); they are sampled once per reset() from the env RNG (seed => whole episode).

Model components (the only definitions; step() and the exact model use them):
  reward(state, action, next_state)      r_t
  transition(state, action, t, rng)      sample s_{t+1} ~ P_t(. | s_t, a_t)
  observation_probs(state, action, t)    {opp_obs: prob}
  observe(state, action, t, rng)         sample from observation_probs
Exact model (opponent ignores the robot and the robot is observed, so beliefs are over
the 400 hidden values h = (opponent cell, mode), index h = opp * 4 + mode):
  opponent_kernel(t)   K_t[h, h'] for the step t -> t+1
  initial_belief(robot, opp_obs) / belief_update(b, robot_next, opp_obs, t)
"""
from __future__ import annotations

from typing import NamedTuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .grid import ACTION_NAMES, TagGrid, open_layout
from .schedules import as_schedule

MODE_NAMES = ACTION_NAMES[:4]          # modes use the action indices: 0 up, 1 down, 2 left, 3 right
MODE_ARROWS = "^v<>"
N_MODES = 4


def simple_eps(t, rng=None):
    """Placeholder time-varying radius for experiment B: 3 + 2 sin(2 pi t / 50), i.e.
    between 1 and 5 (Manhattan), two cycles per 100-step episode. Replace once the eps(t)
    design is decided (any Schedule or fn(t, rng) works)."""
    return 3.0 + 2.0 * np.sin(2 * np.pi * np.asarray(t) / 50.0)


class ModeTagState(NamedTuple):
    robot: int
    opponent: int
    mode: int


class ModeTagEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        size: int = 10,
        p_o=0.8,
        p_m=0.9,
        mode_schedule=None,
        observation: str = "row_col",
        eps=simple_eps,
        distance_metric: str = "manhattan",
        reward: str = "sparse",
        reward_capture: float = 10.0,
        reward_step: float = -1.0,
        max_steps: int = 100,
        gamma: float = 0.95,
        obs_mode: str = "discrete",
        obs_include_time: bool = False,
        render_mode: str | None = None,
    ):
        self.grid = TagGrid(open_layout(size))
        n = self.grid.n
        self.size, self.n_cells, self.n_modes = size, n, N_MODES
        self.UNSEEN = n
        self.n_hidden = n * N_MODES
        self.n_obs = n * (n + 1)

        if observation not in ("row_col", "radius"):
            raise ValueError("observation must be 'row_col' or 'radius'")
        if reward not in ("sparse", "dense"):
            raise ValueError("reward must be 'sparse' or 'dense'")
        self.observation, self.reward_type = observation, reward
        self.schedules = {"p_o": as_schedule(p_o), "p_m": as_schedule(p_m), "eps": as_schedule(eps)}
        self.mode_schedule = None if mode_schedule is None else as_schedule(mode_schedule)
        self.distance_metric = distance_metric
        self._dist = self.grid.distance(distance_metric)
        self.reward_capture, self.reward_step = reward_capture, reward_step
        self.max_steps, self.gamma = int(max_steps), gamma
        self.obs_mode, self.obs_include_time = obs_mode, obs_include_time
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        if obs_mode == "discrete":
            if obs_include_time:
                raise ValueError("obs_include_time needs obs_mode='vector'")
            self.observation_space = spaces.Discrete(self.n_obs)
        elif obs_mode == "vector":
            dim = n + (n + 1) + int(obs_include_time)
            self.observation_space = spaces.Box(0.0, 1.0, shape=(dim,), dtype=np.float32)
        else:
            raise ValueError("obs_mode must be 'discrete' or 'vector'")

        xy = self.grid.coords
        self._same_line = (xy[:, None, 0] == xy[None, :, 0]) | (xy[:, None, 1] == xy[None, :, 1])
        self._kernel_cache: dict = {}
        self.realization: dict[str, np.ndarray] = {}
        self.robot = self.opponent = self.mode = 0
        self.t = 0
        self._last = None

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng, H = self.np_random, self.max_steps
        r = {k: s.sample(rng, H) for k, s in self.schedules.items()}
        r["p_o"] = np.clip(r["p_o"], 0.0, 1.0)
        r["p_m"] = np.clip(r["p_m"], 0.0, 1.0)
        r["eps"] = np.maximum(r["eps"], 0.0)
        if self.mode_schedule is not None:
            r["mode"] = np.rint(self.mode_schedule.sample(rng, H)).astype(int) % N_MODES
        self.realization = r

        options = options or {}
        n = self.n_cells
        self.robot = int(options.get("robot", rng.integers(n)))
        if "opponent" in options:
            self.opponent = int(options["opponent"])
        else:
            self.opponent = int(rng.integers(n - 1))
            self.opponent += self.opponent >= self.robot     # uniform over cells != robot
        if "mode" in options:
            self.mode = int(options["mode"])
        elif self.mode_schedule is not None:
            self.mode = int(r["mode"][0])
        else:
            self.mode = int(rng.integers(N_MODES))
        self.t, self._last = 0, None
        opp_obs = self.observe(self.state, None, 0)
        return self._encode(opp_obs), self._info(opp_obs)

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        if self._last is not None and self._last[2]:
            raise RuntimeError("episode is over; call reset()")
        s = self.state
        s_next = self.transition(s, action, self.t)
        reward = self.reward(s, action, s_next)
        self.robot, self.opponent, self.mode = s_next
        self.t += 1
        terminated = s_next.robot == s_next.opponent
        truncated = (not terminated) and self.t >= self.max_steps
        opp_obs = self.observe(s_next, action, self.t)
        self._last = (action, reward, terminated)
        if self.render_mode == "human":
            print(self.render())
        return self._encode(opp_obs), float(reward), terminated, truncated, self._info(opp_obs)

    def render(self):
        return self._render_rgb() if self.render_mode == "rgb_array" else self._render_ansi()

    # ------------------------------------------------------------ model components
    def reward(self, state, action: int, next_state) -> float:
        """r_t from the post-move state: capture bonus, else -1 (sparse) or -d (dense)."""
        x, y = next_state[0], next_state[1]
        if x == y:
            return self.reward_capture
        return self.reward_step if self.reward_type == "sparse" else -float(self._dist[x, y])

    def transition(self, state, action: int, t: int, rng=None) -> ModeTagState:
        """Sample s_{t+1}: robot moves (deterministic), opponent moves by mode with p_o(t),
        then the mode switches with prob 1 - p_m(t) (or follows the mode schedule)."""
        rng = self.np_random if rng is None else rng
        robot, opp, mode = state
        robot_next = int(self.grid.move[robot, action])
        p_o, p_m = self._dynamics_params(t)
        dirs, probs = self._opponent_moves(mode, p_o)
        d = dirs[rng.choice(5, p=probs)]
        opp_next = opp if d == 4 else int(self.grid.move[opp, d])
        if self.mode_schedule is not None:
            mode_next = int(self.realization["mode"][min(t + 1, self.max_steps)])
        elif rng.random() < p_m:
            mode_next = mode
        else:
            mode_next = int((mode + 1 + rng.integers(3)) % N_MODES)
        return ModeTagState(robot_next, opp_next, mode_next)

    def observation_probs(self, state, action, t: int) -> dict:
        """Z_t: {opp_obs: prob}. Deterministic given the state and eps(t)."""
        robot, opp = state[0], state[1]
        return {opp if self._visible(robot, opp, t) else self.UNSEEN: 1.0}

    def observe(self, state, action, t: int, rng=None) -> int:
        probs = self.observation_probs(state, action, t)
        if len(probs) == 1:
            return next(iter(probs))
        rng = self.np_random if rng is None else rng
        keys = list(probs)
        return keys[rng.choice(len(keys), p=list(probs.values()))]

    def _opponent_moves(self, mode: int, p_o: float):
        """(directions, probs) for one opponent step. Direction 4 = stay."""
        dirs = (mode, *[d for d in range(4) if d != mode], 4)
        probs = np.array([p_o] + [(1.0 - p_o) / 4.0] * 4)
        return dirs, probs

    def _visible(self, robot: int, opp, t: int):
        """Visibility of opponent cell(s) ``opp`` from ``robot`` at time t."""
        if self.observation == "row_col":
            return self._same_line[robot, opp]
        return self._dist[robot, opp] <= self.realization["eps"][min(int(t), self.max_steps)]

    def _dynamics_params(self, t: int):
        r, t = self.realization, min(int(t), self.max_steps)
        return float(r["p_o"][t]), float(r["p_m"][t])

    # ------------------------------------------------------------------ exact model
    def opponent_kernel(self, t: int) -> np.ndarray:
        """K_t[h, h'] = P(opp', mode' | opp, mode) for the step t -> t+1 (h = opp*4 + mode).
        Built from the same pieces as transition()."""
        p_o, p_m = self._dynamics_params(t)
        mode_next = None
        if self.mode_schedule is not None:
            mode_next = int(self.realization["mode"][min(int(t) + 1, self.max_steps)])
        key = (p_o, p_m, mode_next)
        if key in self._kernel_cache:
            return self._kernel_cache[key]
        n = self.n_cells
        M = np.zeros((N_MODES, N_MODES))
        if mode_next is None:
            M[:] = (1.0 - p_m) / 3.0
            np.fill_diagonal(M, p_m)
        else:
            M[:, mode_next] = 1.0
        K = np.zeros((n, N_MODES, n, N_MODES))
        move_stay = np.concatenate([self.grid.move, np.arange(n)[:, None]], axis=1)  # [n, 5]
        for m in range(N_MODES):
            dirs, probs = self._opponent_moves(m, p_o)
            for d, p in zip(dirs, probs):
                np.add.at(K, (np.arange(n), m, move_stay[:, d]), p * M[m])
        K = K.reshape(self.n_hidden, self.n_hidden)
        K.setflags(write=False)
        if len(self._kernel_cache) > 16:
            self._kernel_cache.clear()
        self._kernel_cache[key] = K
        return K

    def obs_likelihood(self, robot: int, opp_obs: int, t: int) -> np.ndarray:
        """P(opp_obs | robot, h) at time t, as a vector over h."""
        n = self.n_cells
        vis = np.asarray(self._visible(robot, np.arange(n), t), dtype=bool)
        if opp_obs == self.UNSEEN:
            like = (~vis).astype(float)
        else:
            like = np.zeros(n)
            like[opp_obs] = float(vis[opp_obs])
        return np.repeat(like, N_MODES)

    def initial_belief(self, robot: int, opp_obs: int) -> np.ndarray:
        """b_0(h) given o_0: opponent uniform over cells != robot, mode uniform (or m(0))."""
        b = np.full((self.n_cells, N_MODES), 1.0)
        b[robot] = 0.0
        if self.mode_schedule is not None:
            b[:, [m for m in range(N_MODES) if m != self.realization["mode"][0]]] = 0.0
        b = b.ravel() * self.obs_likelihood(robot, opp_obs, 0)
        return b / b.sum()

    def belief_update(self, b: np.ndarray, robot_next: int, opp_obs: int, t: int) -> np.ndarray:
        """b_{t+1}(h') ∝ P(o_{t+1} | x_{t+1}, h') sum_h b_t(h) K_t[h, h'], for a
        non-terminal step (so the opponent is not on the robot's cell)."""
        post = (b @ self.opponent_kernel(t)) * self.obs_likelihood(robot_next, opp_obs, t + 1)
        post.reshape(self.n_cells, N_MODES)[robot_next] = 0.0     # not captured
        z = post.sum()
        if z <= 0:
            raise ValueError("observation has zero probability under the model")
        return post / z

    def reward_table(self) -> np.ndarray:
        """R[x', h'] for the post-move state (robot x', hidden h')."""
        n = self.n_cells
        R = np.empty((n, n))
        for x in range(n):
            for y in range(n):
                R[x, y] = self.reward(None, None, (x, y))
        return np.repeat(R, N_MODES, axis=1)

    # ------------------------------------------------------------ encoding / info
    @property
    def state(self) -> ModeTagState:
        return ModeTagState(self.robot, self.opponent, self.mode)

    def _encode(self, opp_obs: int):
        if self.obs_mode == "discrete":
            return self.robot * (self.n_cells + 1) + opp_obs
        n = self.n_cells
        v = np.zeros(self.observation_space.shape, dtype=np.float32)
        v[self.robot] = 1.0
        v[n + opp_obs] = 1.0
        if self.obs_include_time:
            v[2 * n + 1] = self.t / self.max_steps
        return v

    def _info(self, opp_obs: int) -> dict:
        """Everything except obs_index / t is privileged: never feed it to the policy."""
        t = min(self.t, self.max_steps)
        r = self.realization
        info = {
            "t": self.t,
            "obs_index": self.robot * (self.n_cells + 1) + opp_obs,
            "robot": self.robot,
            "opponent": self.opponent,
            "mode": self.mode,
            "hidden_index": self.opponent * N_MODES + self.mode,
            "opponent_visible": opp_obs != self.UNSEEN,
            "p_o": float(r["p_o"][t]),
            "p_m": float(r["p_m"][t]),
        }
        if self.observation == "radius":
            info["eps"] = float(r["eps"][t])
        return info

    # ------------------------------------------------------------------ rendering
    def _visible_cells(self) -> np.ndarray:
        return np.asarray(self._visible(self.robot, np.arange(self.n_cells), self.t), dtype=bool)

    def _render_ansi(self) -> str:
        g, s = self.grid, self.size
        canvas = [["."] * s for _ in range(s)]
        vis = self._visible_cells()
        for i, (x, y) in enumerate(g.coords):
            if vis[i]:
                canvas[s - 1 - y][x] = "*"
        ox, oy = g.coords[self.opponent]
        canvas[s - 1 - oy][ox] = "O" if vis[self.opponent] else "o"
        x, y = g.coords[self.robot]
        canvas[s - 1 - y][x] = "X" if self.robot == self.opponent else "R"
        head = f"t={self.t} mode={MODE_ARROWS[self.mode]}"
        if self.observation == "radius":
            head += f" eps={self.realization['eps'][min(self.t, self.max_steps)]:.2f}"
        if self._last:
            head += f" | last: {ACTION_NAMES[self._last[0]]} r={self._last[1]:+g}"
        return head + "\n" + "\n".join(" ".join(row) for row in canvas) + "\n"

    def _render_rgb(self, cell: int = 32) -> np.ndarray:
        g, s = self.grid, self.size
        img = np.full((s * cell, s * cell, 3), 40, dtype=np.uint8)
        vis = self._visible_cells()

        def paint(i, color, pad=1):
            x, y = g.coords[i]
            py = (s - 1 - y) * cell
            img[py + pad:py + cell - pad, x * cell + pad:(x + 1) * cell - pad] = color

        for i in range(g.n):
            paint(i, (250, 235, 160) if vis[i] else (225, 225, 225))
        paint(self.opponent, (215, 50, 50) if vis[self.opponent] else (240, 160, 160), pad=6)
        paint(self.robot, (40, 90, 210), pad=10)
        return img
