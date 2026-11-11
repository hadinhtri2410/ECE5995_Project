"""Time-varying parameter processes.

A schedule turns an episode into a realised sequence ``x_0, x_1, ..., x_H``.
The env samples every schedule once in ``reset()`` (with the env's RNG), so the
realisation is reproducible from the seed and the exact model at time ``t`` is
available to model-based planners.

Any env parameter accepts a number (-> Constant), a Schedule, or a callable
``fn(t: np.ndarray, rng) -> np.ndarray``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Schedule(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator, horizon: int) -> np.ndarray:
        """Return an array of length ``horizon + 1`` (values for t = 0..horizon)."""

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class Constant(Schedule):
    def __init__(self, value: float):
        self.value = value

    def sample(self, rng, horizon):
        return np.full(horizon + 1, self.value, dtype=float)


class PerEpisode(Schedule):
    """Constant within an episode, re-drawn each episode (domain randomisation).

    Give either ``values`` (discrete choice, optional probs ``p``) or ``low``/``high``.
    """

    def __init__(self, values=None, p=None, low=None, high=None):
        self.values, self.p, self.low, self.high = values, p, low, high

    def sample(self, rng, horizon):
        v = _draw(rng, self.values, self.p, self.low, self.high, size=None)
        return np.full(horizon + 1, v, dtype=float)


class IID(Schedule):
    """A fresh independent draw at every time step."""

    def __init__(self, values=None, p=None, low=None, high=None):
        self.values, self.p, self.low, self.high = values, p, low, high

    def sample(self, rng, horizon):
        return np.asarray(_draw(rng, self.values, self.p, self.low, self.high, size=horizon + 1), dtype=float)


class RegimeSwitching(Schedule):
    """Markov regime switching: each step, with probability ``hazard``, jump to a
    different value chosen uniformly from ``values``. Initial regime ~ ``initial_p``."""

    def __init__(self, values, hazard: float = 0.05, initial_p=None):
        self.values, self.hazard, self.initial_p = list(values), hazard, initial_p

    def sample(self, rng, horizon):
        k = len(self.values)
        idx = np.empty(horizon + 1, dtype=int)
        idx[0] = rng.choice(k, p=self.initial_p)
        for t in range(1, horizon + 1):
            if k > 1 and rng.random() < self.hazard:
                j = rng.integers(k - 1)
                idx[t] = j if j < idx[t - 1] else j + 1
            else:
                idx[t] = idx[t - 1]
        return np.asarray(self.values, dtype=float)[idx]


class Sinusoid(Schedule):
    """mean + amplitude * sin(2*pi*t/period + phase) + N(0, noise_std), clipped.
    ``phase=None`` draws a random phase each episode."""

    def __init__(self, mean, amplitude, period, phase=None, noise_std=0.0, low=None, high=None):
        self.mean, self.amplitude, self.period = mean, amplitude, period
        self.phase, self.noise_std, self.low, self.high = phase, noise_std, low, high

    def sample(self, rng, horizon):
        t = np.arange(horizon + 1)
        phase = rng.uniform(0, 2 * np.pi) if self.phase is None else self.phase
        x = self.mean + self.amplitude * np.sin(2 * np.pi * t / self.period + phase)
        if self.noise_std:
            x = x + rng.normal(0, self.noise_std, size=x.shape)
        return _clip(x, self.low, self.high)


class RandomWalk(Schedule):
    """Gaussian random walk clipped to [low, high]. ``start=None`` draws uniformly."""

    def __init__(self, low, high, step_std, start=None):
        self.low, self.high, self.step_std, self.start = low, high, step_std, start

    def sample(self, rng, horizon):
        x = np.empty(horizon + 1)
        x[0] = rng.uniform(self.low, self.high) if self.start is None else self.start
        steps = rng.normal(0, self.step_std, size=horizon)
        for t in range(1, horizon + 1):
            x[t] = np.clip(x[t - 1] + steps[t - 1], self.low, self.high)
        return x


class Piecewise(Schedule):
    """Deterministic piecewise-constant: ``values[i]`` holds from ``breakpoints[i-1]``
    (or 0) up to ``breakpoints[i]``. ``len(values) == len(breakpoints) + 1``."""

    def __init__(self, breakpoints, values):
        if len(values) != len(breakpoints) + 1:
            raise ValueError("need len(values) == len(breakpoints) + 1")
        self.breakpoints, self.values = list(breakpoints), list(values)

    def sample(self, rng, horizon):
        t = np.arange(horizon + 1)
        return np.asarray(self.values, dtype=float)[np.searchsorted(self.breakpoints, t, side="right")]


class FunctionSchedule(Schedule):
    def __init__(self, fn):
        self.fn = fn

    def sample(self, rng, horizon):
        out = np.broadcast_to(np.asarray(self.fn(np.arange(horizon + 1), rng), dtype=float), (horizon + 1,))
        return out.copy()


def as_schedule(x) -> Schedule:
    if isinstance(x, Schedule):
        return x
    if callable(x):
        return FunctionSchedule(x)
    return Constant(float(x))


def _draw(rng, values, p, low, high, size):
    if values is not None:
        return rng.choice(np.asarray(values, dtype=float), size=size, p=p)
    if low is None or high is None:
        raise ValueError("give either `values` or both `low` and `high`")
    return rng.uniform(low, high, size=size)


def _clip(x, low, high):
    if low is not None or high is not None:
        x = np.clip(x, -np.inf if low is None else low, np.inf if high is None else high)
    return x
