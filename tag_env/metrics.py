"""Per-episode metrics (design doc Sec. 7).

``TagMetrics`` wraps the env and records, for every finished episode:

    seed                  seed passed to reset() (None if not given)
    return                undiscounted sum of rewards
    discounted_return     sum_t gamma^t r_t
    length                number of steps taken
    tagged                1 if the episode ended with a successful tag, else 0
    truncated             1 if it hit max_steps
    visible_fraction      fraction of decision steps (o_0 .. o_{T-1}) at which the
                          opponent was in the agent's observation
    failed_tags           number of Tag actions that missed (NaN for ModeTag: no Tag action)
    estimation_error      mean ||x_hat_opp - x_opp|| (Euclidean, grid units) over the steps
                          where the method logged an estimate via ``log_estimate``; NaN if none
    mode_accuracy         ModeTag only: share of logged estimates whose most likely mode is
                          the true mode (needs a belief over (opponent, mode)); NaN otherwise

``summarize`` also reports ``capture_time``: mean length of the episodes that ended in a
capture/tag.

The wrapper reads the privileged ``info`` for bookkeeping only; it never changes the
observation the policy receives.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from .grid import TAG

FIELDS = ("return", "discounted_return", "length", "tagged", "truncated",
          "visible_fraction", "failed_tags", "estimation_error", "mode_accuracy")


class TagMetrics(gym.Wrapper):
    def __init__(self, env: gym.Env, gamma: float | None = None):
        super().__init__(env)
        base = env.unwrapped
        self.gamma = base.gamma if gamma is None else gamma
        self._coords = base.grid.coords.astype(float)
        self._n = base.n_cells
        self._has_tag = base.action_space.n == 5          # Pineau Tag has a Tag action
        self._n_modes = getattr(base, "n_modes", None)    # ModeTag: hidden opponent mode
        self.episodes: list[dict] = []
        self._ep = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._ep = dict(seed=seed, ret=0.0, dret=0.0, length=0, failed_tags=0,
                        visible=[bool(info["opponent_visible"])], errors=[], modes=[])
        self._info = info
        return obs, info

    def step(self, action):
        if self._ep is None:
            raise RuntimeError("call reset() first")
        obs, reward, terminated, truncated, info = self.env.step(action)
        ep = self._ep
        ep["dret"] += self.gamma ** ep["length"] * reward
        ep["ret"] += reward
        ep["length"] += 1
        if self._has_tag and int(action) == TAG and not terminated:  # Tag that missed
            ep["failed_tags"] += 1
        self._info = info
        if terminated or truncated:
            record = self._finish(terminated, truncated)
            info = dict(info, episode_metrics=record)
        else:
            ep["visible"].append(bool(info["opponent_visible"]))
        return obs, reward, terminated, truncated, info

    def log_estimate(self, estimate) -> float:
        """Log the method's estimate of the opponent position for the *current* observation
        (call it after reset/step and before the next step). ``estimate`` may be

          * an int cell index,
          * an (x, y) pair of grid coordinates,
          * a probability vector over opponent cells (length n_cells or n_cells + 1), or
          * a full belief over states (length n_states); its opponent marginal is used,
          * ModeTag: a belief over (opponent, mode), length n_cells * 4 (index opp*4+mode);
            this also scores the mode estimate.

        Distributions are reduced to their mean position. Returns the error."""
        true = int(self._info["opponent"])
        if true >= self._n:  # already tagged
            return math.nan
        if self._n_modes and np.size(estimate) == self._n * self._n_modes:
            b = np.asarray(estimate, dtype=float).reshape(self._n, self._n_modes)
            self._ep["modes"].append(float(np.argmax(b.sum(0)) == self._info["mode"]))
            estimate = b.sum(1)
        x_hat = self._estimate_xy(estimate)
        err = float(np.linalg.norm(x_hat - self._coords[true]))
        self._ep["errors"].append(err)
        return err

    def _estimate_xy(self, estimate) -> np.ndarray:
        if np.isscalar(estimate):
            return self._coords[int(estimate)]
        e = np.asarray(estimate, dtype=float).ravel()
        if e.size == 2:
            return e
        n, no = self._n, self._n + 1
        if e.size == n * no:
            e = e.reshape(n, no).sum(0)
        if e.size == no:
            e = e[:n]
        if e.size != n:
            raise ValueError(f"cannot interpret estimate of size {e.size}")
        total = e.sum()
        if total <= 0:
            raise ValueError("estimate puts no mass on opponent cells")
        return (e / total) @ self._coords

    def _finish(self, terminated, truncated) -> dict:
        ep = self._ep
        record = {
            "seed": ep["seed"],
            "return": ep["ret"],
            "discounted_return": ep["dret"],
            "length": ep["length"],
            "tagged": int(terminated),
            "truncated": int(truncated),
            "visible_fraction": float(np.mean(ep["visible"])),
            "failed_tags": ep["failed_tags"] if self._has_tag else math.nan,
            "estimation_error": float(np.mean(ep["errors"])) if ep["errors"] else math.nan,
            "mode_accuracy": float(np.mean(ep["modes"])) if ep["modes"] else math.nan,
        }
        self.episodes.append(record)
        self._ep = None
        return record


def summarize(records: list[dict]) -> dict:
    """Mean and standard error of every metric over episodes (NaNs ignored).
    The mean of ``tagged`` is the successful tag rate."""
    out = {"episodes": len(records)}
    caught = np.array([r["length"] for r in records if r["tagged"]], dtype=float)
    out["capture_time"] = float(caught.mean()) if caught.size else math.nan
    out["capture_time_se"] = (float(caught.std(ddof=1) / np.sqrt(caught.size))
                              if caught.size > 1 else math.nan)
    for k in FIELDS:
        x = np.array([r[k] for r in records], dtype=float)
        x = x[~np.isnan(x)]
        out[k] = float(x.mean()) if x.size else math.nan
        out[k + "_se"] = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else math.nan
    return out
