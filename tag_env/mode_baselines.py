"""Reference policies for ModeTagEnv (final-report game)."""
from __future__ import annotations

import numpy as np

from .baselines import RandomPolicy  # noqa: F401  (re-export: works for both games)
from .evaluation import Policy, obs_to_index
from .grid import DOWN, LEFT, RIGHT, UP


def _split(env, obs):
    i = obs_to_index(env, obs)
    return divmod(i, env.n_cells + 1)          # (robot cell, opp_obs or UNSEEN)


def _step_towards(env, robot: int, target: int, rng) -> int:
    """Move that reduces the larger coordinate gap to target (random tie-break)."""
    (rx, ry), (tx, ty) = env.grid.coords[robot], env.grid.coords[target]
    options = []
    if tx != rx:
        options.append((abs(tx - rx), RIGHT if tx > rx else LEFT))
    if ty != ry:
        options.append((abs(ty - ry), UP if ty > ry else DOWN))
    if not options:
        return int(rng.integers(4))
    best = max(g for g, _ in options)
    return int(rng.choice([a for g, a in options if g == best]))


class ChaseLastSeenPolicy(Policy):
    """Heuristic with one cell of memory: head for where the opponent was last seen; once
    there (or if never seen) move randomly. No model knowledge."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env, obs) -> None:
        self.env = env
        self.target = None
        self._see(obs)

    def _see(self, obs):
        robot, opp = _split(self.env, obs)
        self.robot = robot
        if opp != self.env.UNSEEN:
            self.target = opp
        elif self.target == robot:
            self.target = None

    def act(self, obs) -> int:
        if self.target is None:
            return int(self.rng.integers(4))
        return _step_towards(self.env, self.robot, self.target, self.rng)

    def update(self, action: int, obs) -> None:
        self._see(obs)


def qmdp_values(env, t: int = 0, iters: int = 2000, tol: float = 1e-6) -> np.ndarray:
    """Q[x, a, h] of the fully observable MDP with the model at time t (robot cell x,
    hidden h = (opponent, mode)). Capture is terminal; reward is on the post-move state."""
    K = env.opponent_kernel(t)                       # [H, H]
    R = env.reward_table()                           # [n, H], indexed by post-move robot
    n, H = R.shape
    cap = np.zeros((n, env.n_cells), dtype=bool)
    cap[np.arange(n), np.arange(n)] = True
    cap = np.repeat(cap, env.n_modes, axis=1)        # [n, H]
    move = env.grid.move                              # [n, 4]
    V = np.zeros((n, H))
    Q = np.zeros((n, 4, H))
    for _ in range(iters):
        W = R + env.gamma * np.where(cap, 0.0, V)    # value of landing in (x', h')
        for a in range(4):
            Q[:, a, :] = W[move[:, a]] @ K.T         # sum_h' K[h, h'] W[x'(x, a), h']
        V_new = Q.max(1)
        if np.abs(V_new - V).max() < tol:
            V = V_new
            break
        V = V_new
    return Q


class QMDPPolicy(Policy):
    """QMDP with the exact factored belief over (opponent, mode). Knows p_o, p_m and eps(t)
    (model-based oracle reference, not a learner).

    replan=False: Q from the model at t = 0 for the whole episode.
    replan=True:  Q recomputed whenever the dynamics at t differ from the last solve.
    """

    def __init__(self, replan: bool = False):
        self.replan = replan
        self._cache: dict = {}

    def _Q(self, t):
        key = (id(self.env), self.env._dynamics_params(t), self.env.reward_type)
        if key not in self._cache:
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = qmdp_values(self.env, t)
        return self._cache[key]

    def reset(self, env, obs) -> None:
        self.env, self.t = env, 0
        robot, opp = _split(env, obs)
        self.robot = robot
        self.b = env.initial_belief(robot, opp)
        self.Q = self._Q(0)

    def act(self, obs) -> int:
        if self.replan:
            self.Q = self._Q(self.t)
        return int(np.argmax(self.Q[self.robot] @ self.b))

    def update(self, action: int, obs) -> None:
        robot, opp = _split(self.env, obs)
        self.b = self.env.belief_update(self.b, robot, opp, self.t)
        self.robot = robot
        self.t += 1

    def opponent_estimate(self):
        return self.b
