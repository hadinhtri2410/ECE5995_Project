"""Reference policies for the legacy Pineau Tag env (RandomPolicy works for both games;
ModeTag baselines are in mode_baselines.py)."""
from __future__ import annotations

import numpy as np

from .evaluation import Policy, obs_to_index


class RandomPolicy(Policy):
    """Uniform over the env's actions (works for both games)."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.n_actions = 5

    def reset(self, env, obs) -> None:
        self.n_actions = env.action_space.n

    def act(self, obs) -> int:
        return int(self.rng.integers(self.n_actions))


def qmdp_values(model, iters: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """Q[s, a] of the fully observable MDP (Littman et al. 1995)."""
    T, R, g = model.T, model.R, model.gamma
    V = np.zeros(T.shape[1])
    for _ in range(iters):
        Q = R + g * np.einsum("asn,n->sa", T, V)
        V_new = Q.max(1)
        if np.abs(V_new - V).max() < tol:
            break
        V = V_new
    return Q


class QMDPPolicy(Policy):
    """QMDP with an exact Bayes filter. Uses env.model(t), i.e. the *realised* schedule of
    this episode, so on time-varying presets it is an oracle-model baseline.

    replan=False: Q from model(0) for the whole episode (frozen in time).
    replan=True:  Q re-solved from model(t) at every step.
    """

    def __init__(self, replan: bool = False):
        self.replan = replan

    def reset(self, env, obs) -> None:
        self.env = env
        self.t = 0
        self.Q = qmdp_values(env.model(0))
        b = env.initial_belief() * env.model(0).Z[:, obs_to_index(env, obs)]
        self.b = b / b.sum()

    def act(self, obs) -> int:
        if self.replan and self.t > 0:
            self.Q = qmdp_values(self.env.model(self.t))
        return int(np.argmax(self.b @ self.Q))

    def update(self, action: int, obs) -> None:
        self.b = self.env.belief_update(self.b, action, obs_to_index(self.env, obs), self.t)
        self.t += 1

    def opponent_estimate(self):
        return self.b
