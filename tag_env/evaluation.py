"""Fair evaluation harness (design doc Sec. 8).

Every method is run on the same list of seeds. A seed fixes the start positions and the
whole schedule realisation (they are drawn in reset() before any action), so all methods
face identical initial states and identical d(t) / opponent-speed sequences. Reward,
termination and max_steps come from the env config, which is passed unchanged.

Policy interface (subclass ``Policy``):

    reset(env, obs)        new episode; ``env`` is the unwrapped env, ``obs`` is o_0
    act(obs) -> action     choose a_t from o_t
    update(action, obs)    called after a non-terminal step with (a_t, o_{t+1})
    opponent_estimate()    optional; return an estimate of the opponent position for the
                           latest observation (see TagMetrics.log_estimate), or None

Policies get ``env`` so model-based methods can call env.model(t) / belief_update. Learned
policies should ignore it: only ``obs`` is legitimate input.
"""
from __future__ import annotations

import csv
import json

import gymnasium as gym
import numpy as np

from .metrics import FIELDS, TagMetrics, summarize

# Shared evaluation seeds. Use a prefix (EVAL_SEEDS[:n]) for quick runs; never reuse these
# for training.
EVAL_SEEDS = tuple(range(100_000, 101_000))


class Policy:
    def reset(self, env, obs) -> None:
        pass

    def act(self, obs) -> int:
        raise NotImplementedError

    def update(self, action: int, obs) -> None:
        pass

    def opponent_estimate(self):
        return None


def make_env(env_id: str, **env_kwargs) -> TagMetrics:
    return TagMetrics(gym.make(env_id, **env_kwargs))


def evaluate(policy: Policy, env_id: str, seeds=EVAL_SEEDS, env_kwargs: dict | None = None,
             verbose: bool = False):
    """Run ``policy`` once per seed. Returns (per-episode records, summary)."""
    env = make_env(env_id, **(env_kwargs or {}))
    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        policy.reset(env.unwrapped, obs)
        while True:
            est = policy.opponent_estimate()
            if est is not None:
                env.log_estimate(est)
            action = int(policy.act(obs))
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            policy.update(action, obs)
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(seeds)} episodes")
    env.close()
    return env.episodes, summarize(env.episodes)


def obs_to_index(env, obs) -> int:
    """Discrete observation index robot * (n+1) + opp_obs, for either obs_mode."""
    if np.isscalar(obs) or np.ndim(obs) == 0:
        return int(obs)
    n = env.n_cells
    v = np.asarray(obs)
    return int(np.argmax(v[:n])) * (n + 1) + int(np.argmax(v[n:2 * n + 1]))


def format_summary(name: str, s: dict) -> str:
    def f(k, p=2):
        return f"{s[k]:.{p}f} ± {s[k + '_se']:.{p}f}"
    out = (f"{name}: n={s['episodes']}  disc. return {f('discounted_return')}  "
           f"return {f('return')}  capture rate {s['tagged']:.1%}  length {f('length', 1)}  "
           f"visible {s['visible_fraction']:.1%}  est. error {f('estimation_error')}")
    if not np.isnan(s["failed_tags"]):
        out += f"  failed tags {f('failed_tags')}"
    if not np.isnan(s["mode_accuracy"]):
        out += f"  mode acc. {s['mode_accuracy']:.1%}"
    return out


def save_records(records: list[dict], path: str, meta: dict | None = None) -> None:
    """Write per-episode records to .csv or .json (json also stores ``meta``)."""
    if path.endswith(".json"):
        with open(path, "w") as fh:
            json.dump({"meta": meta or {}, "summary": summarize(records), "episodes": records},
                      fh, indent=1)
    else:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["seed", *FIELDS])
            w.writeheader()
            w.writerows(records)
