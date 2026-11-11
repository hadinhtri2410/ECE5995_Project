"""Time-varying tag game POMDP (ECE 595 project).

Base game = Pineau, Gordon & Thrun (2003) "Tag":
  * 29 grid cells; state = (robot cell, opponent cell or TAGGED) -> 29*30 = 870 states.
  * Actions {Up, Down, Left, Right, Tag}.
  * Reward -1 per move, +10 for Tag when co-located (episode ends), -10 for a failed Tag.
  * The robot position is always observed.
  * The opponent tries to move away from the robot with prob. p_move, else stays
    (see TagGrid._away_table for the "axis" (default) and "greedy" rules). By default it
    reacts to the robot's position *before* the robot's move (T(s'|s,a) depends on s only,
    as in the paper); opponent_reacts_to="post_move" makes it react to the new position.

Time-varying extensions (proposal Sec. II), each given as a Schedule:
  * visibility_radius d(t): opponent observed iff dist(robot, opponent) < d(t)
    (d = 0.5 reproduces Pineau's "only when in the same cell").
  * detection_prob(t): probability that a visible opponent is actually reported.
  * opponent_speed(t): number of opponent sub-moves per step (0 = not moving).
  * opponent_move_prob(t): p_move used by each sub-move.

Time indexing: the observation o_t is generated from s_t with Z_t (radius d(t),
detection_prob(t)); the step s_t -> s_{t+1} uses opponent_speed(t), opponent_move_prob(t).

Model components (design doc Sec. 3). ``step()`` and the exact matrices in ``model(t)`` are
both built from these, so the simulator and the planner model cannot drift apart:
  * reward(state, action)                    R(s, a)
  * transition(state, action, t, rng)        sample s_{t+1} ~ P_t(. | s_t, a_t)
  * observation_probs(state, action, t)      Z_t(. | s_t, a_{t-1}) as {opp_obs: prob}
  * observe(state, action, t, rng)           sample o_t from observation_probs
where ``state`` is a TagState(robot, opponent).
To change Z or R, edit/override observation_probs or reward. ``transition`` is a sampler, so
the matrix T is built from its two building blocks instead: _robot_step (robot move / tag)
and the opponent move table grid.away + _dynamics_params(t). Change P there.

Observation encodings (``obs_mode``):
  * "discrete": int  robot * 30 + opp_obs, opp_obs in 0..28 or 29 (= unseen). Discrete(870).
  * "vector":   one-hot(robot, 29) ++ one-hot(opp_obs, 30) [++ t/max_steps] [++ d(t)].
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import NamedTuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .grid import ACTION_NAMES, PINEAU_LAYOUT, TAG, TagGrid
from .schedules import as_schedule


class TagState(NamedTuple):
    """True state s = (robot cell, opponent cell or TAGGED). Hidden from the policy."""

    robot: int
    opponent: int


@dataclass
class POMDPModel:
    """Exact model at one time step. Indices: s = robot*(n+1) + opp, o = robot*(n+1) + opp_obs.

    T[a, s, s']  transition for the step taken at time t
    Z[s, o]      observation probabilities of state s at time t (independent of the action)
    R[s, a]      reward
    """

    T: np.ndarray
    Z: np.ndarray
    R: np.ndarray
    gamma: float


class TimeVaryingTagEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        visibility_radius=0.5,
        detection_prob=1.0,
        opponent_speed=1,
        opponent_move_prob=0.8,
        opponent_rule: str = "axis",
        opponent_reacts_to: str = "pre_move",
        distance_metric: str = "manhattan",
        max_steps: int = 100,
        obs_mode: str = "discrete",
        obs_include_time: bool = False,
        obs_include_radius: bool = False,
        reward_move: float = -1.0,
        reward_tag_success: float = 10.0,
        reward_tag_fail: float = -10.0,
        gamma: float = 0.95,
        layout=PINEAU_LAYOUT,
        render_mode: str | None = None,
    ):
        self.grid = TagGrid(layout)
        n = self.grid.n
        self.n_cells = n
        self.TAGGED = n  # opponent index meaning "already tagged"
        self.UNSEEN = n  # opp_obs index meaning "not observed"
        self.n_states = n * (n + 1)
        self.n_obs = n * (n + 1)
        self.n_actions = 5

        self.schedules = {
            "visibility_radius": as_schedule(visibility_radius),
            "detection_prob": as_schedule(detection_prob),
            "opponent_speed": as_schedule(opponent_speed),
            "opponent_move_prob": as_schedule(opponent_move_prob),
        }
        if opponent_rule not in self.grid.away:
            raise ValueError(f"opponent_rule must be one of {list(self.grid.away)}")
        self.opponent_rule = opponent_rule
        self._away = self.grid.away[opponent_rule]
        if opponent_reacts_to not in ("pre_move", "post_move"):
            raise ValueError("opponent_reacts_to must be 'pre_move' or 'post_move'")
        self.opponent_reacts_to = opponent_reacts_to
        self.distance_metric = distance_metric
        self._dist = self.grid.distance(distance_metric)
        self.max_steps = int(max_steps)
        self.obs_mode = obs_mode
        self.obs_include_time = obs_include_time
        self.obs_include_radius = obs_include_radius
        self.reward_move = reward_move
        self.reward_tag_success = reward_tag_success
        self.reward_tag_fail = reward_tag_fail
        self.gamma = gamma
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(5)
        if obs_mode == "discrete":
            if obs_include_time or obs_include_radius:
                raise ValueError("obs_include_* need obs_mode='vector'")
            self.observation_space = spaces.Discrete(self.n_obs)
        elif obs_mode == "vector":
            dim = n + (n + 1) + int(obs_include_time) + int(obs_include_radius)
            high = np.ones(dim, dtype=np.float32)
            if obs_include_radius:
                high[-1] = np.float32(self._dist.max() + 1.0)  # radii beyond this see everything
            self.observation_space = spaces.Box(0.0, high, shape=(dim,), dtype=np.float32)
        else:
            raise ValueError("obs_mode must be 'discrete' or 'vector'")

        # Small LRU: continuous schedules would otherwise create one ~30 MB T per value.
        self._model_cache: OrderedDict = OrderedDict()
        self._model_cache_size = 8
        self.realization: dict[str, np.ndarray] = {}
        self.robot = self.opponent = 0
        self.t = 0
        self._last = None

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        H = self.max_steps
        self.realization = {k: s.sample(rng, H) for k, s in self.schedules.items()}
        self.realization["opponent_speed"] = np.maximum(np.rint(self.realization["opponent_speed"]), 0).astype(int)
        self.realization["detection_prob"] = np.clip(self.realization["detection_prob"], 0.0, 1.0)
        self.realization["opponent_move_prob"] = np.clip(self.realization["opponent_move_prob"], 0.0, 1.0)

        options = options or {}
        self.robot = int(options.get("robot", rng.integers(self.n_cells)))
        self.opponent = int(options.get("opponent", rng.integers(self.n_cells)))
        self.t = 0
        self._last = None
        opp_obs = self.observe(self.state, None, 0)
        return self._encode(opp_obs), self._info(opp_obs)

    def step(self, action):
        """(s_t, a_t, t) -> s_{t+1} -> o_{t+1} -> r_t, done   (design doc Sec. 3)."""
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        if self.opponent == self.TAGGED:
            raise RuntimeError("episode is over; call reset()")
        s = self.state
        reward = self.reward(s, action)
        s_next = self.transition(s, action, self.t)
        self.robot, self.opponent = s_next
        self.t += 1
        terminated = s_next.opponent == self.TAGGED
        truncated = (not terminated) and self.t >= self.max_steps
        opp_obs = self.observe(s_next, action, self.t)
        self._last = (action, reward)
        if self.render_mode == "human":
            print(self.render())
        return self._encode(opp_obs), float(reward), terminated, truncated, self._info(opp_obs)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        return self._render_ansi()

    # ------------------------------------------- model components (design doc Sec. 3)
    def reward(self, state, action: int) -> float:
        """R(s, a). The absorbing TAGGED state gives 0."""
        robot, opp = state
        if opp == self.TAGGED:
            return 0.0
        if action == TAG:
            return self.reward_tag_success if robot == opp else self.reward_tag_fail
        return self.reward_move

    def transition(self, state, action: int, t: int, rng=None) -> TagState:
        """Sample s_{t+1} ~ P_t(. | s_t, a_t). Uses opponent_speed(t), opponent_move_prob(t)."""
        rng = self.np_random if rng is None else rng
        robot, opp = state
        if opp == self.TAGGED:
            return TagState(robot, opp)
        robot_next, tagged, ref = self._robot_step(robot, opp, action)
        if tagged:
            return TagState(robot_next, self.TAGGED)
        speed, p = self._dynamics_params(t)
        for _ in range(speed):
            if rng.random() < p:
                cells, w = zip(*self._away[ref][opp])
                opp = int(cells[rng.choice(len(cells), p=w)])
        return TagState(robot_next, opp)

    def observation_probs(self, state, action, t: int) -> dict:
        """Z_t(. | s_t, a_{t-1}) as {opp_obs: prob}, opp_obs = opponent cell or UNSEEN.

        This is the observation model; change it here to change Z everywhere. ``action`` is
        the action that led to ``state`` (None at reset); the default model ignores it. If
        you make Z action-dependent, POMDPModel.Z will also need an action axis.
        """
        radius, detect = self._observation_params(t)
        return self._obs_dist(state, radius, detect)

    def observe(self, state, action, t: int, rng=None) -> int:
        """Sample the opponent part of o_t from observation_probs (robot cell is always known)."""
        rng = self.np_random if rng is None else rng
        probs = self.observation_probs(state, action, t)
        if len(probs) == 1:
            return next(iter(probs))
        u, acc = rng.random(), 0.0
        for o, p in probs.items():
            acc += p
            if u < acc:
                return o
        return o  # numerical slack

    def _robot_step(self, robot: int, opp: int, action: int):
        """Deterministic part of P: (robot_next, tagged, cell the opponent reacts to)."""
        if action == TAG:
            return robot, robot == opp, robot
        robot_next = int(self.grid.move[robot, action])
        ref = robot if self.opponent_reacts_to == "pre_move" else robot_next
        return robot_next, False, ref

    def _obs_dist(self, state, radius: float, detect: float) -> dict:
        robot, opp = state
        if opp == self.TAGGED or not self._dist[robot, opp] < radius:
            return {self.UNSEEN: 1.0}
        return {opp: detect, self.UNSEEN: 1.0 - detect}

    def _dynamics_params(self, t: int):
        r, t = self.realization, min(int(t), self.max_steps)
        return int(r["opponent_speed"][t]), float(r["opponent_move_prob"][t])

    def _observation_params(self, t: int):
        r, t = self.realization, min(int(t), self.max_steps)
        return float(r["visibility_radius"][t]), float(r["detection_prob"][t])

    # ------------------------------------------------------------ observation
    def _encode(self, opp_obs: int):
        if self.obs_mode == "discrete":
            return self.robot * (self.n_cells + 1) + opp_obs
        n = self.n_cells
        v = np.zeros(self.observation_space.shape, dtype=np.float32)
        v[self.robot] = 1.0
        v[n + opp_obs] = 1.0
        i = 2 * n + 1
        if self.obs_include_time:
            v[i] = self.t / self.max_steps
            i += 1
        if self.obs_include_radius:
            v[i] = min(self.realization["visibility_radius"][self.t], self.observation_space.high[i])
        return v

    def _info(self, opp_obs: int) -> dict:
        """Everything here except obs_index/t is privileged (hidden from the agent)."""
        t = self.t
        return {
            "t": t,
            "obs_index": self.robot * (self.n_cells + 1) + opp_obs,
            "state": self.state_index,
            "robot": self.robot,
            "opponent": self.opponent,
            "opponent_visible": opp_obs != self.UNSEEN,
            "visibility_radius": float(self.realization["visibility_radius"][t]),
            "detection_prob": float(self.realization["detection_prob"][t]),
            "opponent_speed": int(self.realization["opponent_speed"][t]),
            "opponent_move_prob": float(self.realization["opponent_move_prob"][t]),
        }

    @property
    def state(self) -> TagState:
        return TagState(self.robot, self.opponent)

    @property
    def state_index(self) -> int:
        return self.robot * (self.n_cells + 1) + self.opponent

    # ------------------------------------------------------------ exact model
    def initial_belief(self) -> np.ndarray:
        """mu_0: robot and opponent independent and uniform over the cells."""
        n = self.n_cells
        b = np.zeros((n, n + 1))
        b[:, :n] = 1.0 / n**2
        return b.ravel()

    def model(self, t: int | None = None) -> POMDPModel:
        """Exact POMDP model at time t (default: current time) for the current episode's
        realisation of the schedules. Call after reset()."""
        if not self.realization:
            raise RuntimeError("call reset() first")
        t = self.t if t is None else int(t)
        T = self._transition(*self._dynamics_params(t))
        Z = self._observation(t)
        return POMDPModel(T=T, Z=Z, R=self._reward(), gamma=self.gamma)

    def belief_update(self, b: np.ndarray, action: int, obs_index: int, t: int) -> np.ndarray:
        """Bayes filter: b_{t+1}(s') ∝ Z_{t+1}(o|s') * sum_s T_t(s'|s,a) b_t(s)."""
        pred = b @ self._transition(*self._dynamics_params(t))[action]
        post = pred * self._observation(t + 1)[:, obs_index]
        z = post.sum()
        if z <= 0:
            raise ValueError("observation has zero probability under the model")
        return post / z

    def _opponent_kernel(self, speed: int, p: float) -> np.ndarray:
        """K[r, o, o'] for `speed` sub-moves with the robot fixed at r (TAGGED absorbing)."""
        n = self.n_cells
        K = np.zeros((n, n + 1, n + 1))
        K[:, n, n] = 1.0
        for r in range(n):
            for o in range(n):
                K[r, o, o] = 1.0 - p
                for c, w in self._away[r][o]:
                    K[r, o, c] += p * w
        return np.linalg.matrix_power(K, speed)

    def _transition(self, speed: int, p: float) -> np.ndarray:
        """T[a, s, s'] from _robot_step + the opponent kernel (same rules as transition())."""
        key = ("T", speed, p)
        if (hit := self._cache_get(key)) is not None:
            return hit
        n, no = self.n_cells, self.n_cells + 1
        K = self._opponent_kernel(speed, p)
        T = np.zeros((5, self.n_states, self.n_states))
        # Robot part from _robot_step (python loop, cheap); opponent rows written vectorised.
        r, o = np.divmod(np.arange(n * n), n)          # all live (robot, opponent) pairs
        rows = r * no + o
        cols = np.arange(no)
        for a in range(5):
            step = [self._robot_step(int(ri), int(oi), a) for ri, oi in zip(r, o)]
            r2, tagged, ref = (np.array(x) for x in zip(*step))
            live = ~tagged.astype(bool)
            T[a, rows[live, None], r2[live, None] * no + cols] = K[ref[live], o[live]]
            T[a, rows[~live], r2[~live] * no + self.TAGGED] = 1.0
            dead = np.arange(n) * no + self.TAGGED      # absorbing TAGGED states
            T[a, dead, dead] = 1.0
        return self._cache_put(key, T)

    def _observation(self, t: int) -> np.ndarray:
        """Z[s, o] at time t, built by calling observation_probs() on every state.

        Cached by _observation_params(t): if you override observation_probs so that it
        depends on t through anything else, override _observation_params to return it too.
        """
        key = ("Z", *self._observation_params(t))
        if (hit := self._cache_get(key)) is not None:
            return hit
        no = self.n_cells + 1
        Z = np.zeros((self.n_states, self.n_obs))
        for r in range(self.n_cells):
            for o in range(no):
                for opp_obs, p in self.observation_probs(TagState(r, o), None, t).items():
                    Z[r * no + o, r * no + opp_obs] += p
        return self._cache_put(key, Z)

    def _reward(self) -> np.ndarray:
        """R[s, a] from reward()."""
        if (hit := self._cache_get("R")) is not None:
            return hit
        no = self.n_cells + 1
        R = np.zeros((self.n_states, 5))
        for r in range(self.n_cells):
            for o in range(no):
                for a in range(5):
                    R[r * no + o, a] = self.reward((r, o), a)
        return self._cache_put("R", R)

    def _cache_get(self, key):
        if key in self._model_cache:
            self._model_cache.move_to_end(key)
            return self._model_cache[key]
        return None

    def _cache_put(self, key, value):
        value.setflags(write=False)
        self._model_cache[key] = value
        while len(self._model_cache) > self._model_cache_size:
            self._model_cache.popitem(last=False)
        return value

    # ------------------------------------------------------------ rendering
    def _visible_cells(self) -> np.ndarray:
        return self._dist[self.robot] < self.realization["visibility_radius"][min(self.t, self.max_steps)]

    def _render_ansi(self) -> str:
        g = self.grid
        canvas = [[" "] * g.width for _ in range(g.height)]
        vis = self._visible_cells()
        for i, (x, y) in enumerate(g.coords):
            canvas[g.height - 1 - y][x] = "*" if vis[i] else "."
        if self.opponent != self.TAGGED:
            x, y = g.coords[self.opponent]
            canvas[g.height - 1 - y][x] = "O" if vis[self.opponent] else "o"
        x, y = g.coords[self.robot]
        canvas[g.height - 1 - y][x] = "X" if self.opponent in (self.robot, self.TAGGED) else "R"
        r = self.realization
        t = min(self.t, self.max_steps)
        head = (f"t={self.t} d={r['visibility_radius'][t]:.2f} detect={r['detection_prob'][t]:.2f} "
                f"speed={r['opponent_speed'][t]} p_move={r['opponent_move_prob'][t]:.2f}")
        if self._last:
            head += f" | last: {ACTION_NAMES[self._last[0]]} r={self._last[1]:+g}"
        return head + "\n" + "\n".join("".join(row) for row in canvas) + "\n"

    def _render_rgb(self, cell: int = 32) -> np.ndarray:
        g = self.grid
        img = np.full((g.height * cell, g.width * cell, 3), 40, dtype=np.uint8)
        vis = self._visible_cells()

        def paint(i, color, pad=1):
            x, y = g.coords[i]
            py = (g.height - 1 - y) * cell
            img[py + pad:py + cell - pad, x * cell + pad:(x + 1) * cell - pad] = color

        for i in range(g.n):
            paint(i, (250, 235, 160) if vis[i] else (225, 225, 225))
        if self.opponent != self.TAGGED:
            paint(self.opponent, (215, 50, 50) if vis[self.opponent] else (240, 160, 160), pad=6)
        paint(self.robot, (40, 90, 210), pad=10)
        return img
