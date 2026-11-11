"""Watch a random policy play one episode and print the schedule realisation.

    python examples/random_rollout.py --env ModeTag-B-v0 --seed 0
"""
import argparse

import gymnasium as gym
import numpy as np

import tag_env  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument("--env", default="ModeTag-A-v0")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=15)
args = p.parse_args()

env = gym.make(args.env, render_mode="ansi")
obs, info = env.reset(seed=args.seed)
env.action_space.seed(args.seed)
print("legend: R robot, O opponent (visible), o opponent (hidden), * visible cells, X same cell\n")
print(env.render())
rng = np.random.default_rng(args.seed)
G = 0.0
for _ in range(args.steps):
    obs, r, term, trunc, info = env.step(int(env.action_space.sample()))
    G += r
    print(env.render())
    if term or trunc:
        break
print(f"return over {info['t']} steps: {G:+g}")
real = env.unwrapped.realization
for k, v in real.items():
    print(f"{k:>20}: {np.array2string(v[:30], precision=2, max_line_width=200)} ...")
