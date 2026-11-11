"""Experiment A: sweep p_o x p_m on ModeTag-A-v0 for each baseline; one row per cell.

    python examples/sweep_mode_tag.py --episodes 300 --out results/sweep_A.csv

p_o = probability the opponent follows its mode (higher = more predictable).
p_m = probability the mode stays the same each step (higher = switches less often).
"""
import argparse
import csv
import itertools

from tag_env import EVAL_SEEDS, evaluate
from tag_env.mode_baselines import ChaseLastSeenPolicy, QMDPPolicy, RandomPolicy

p = argparse.ArgumentParser()
p.add_argument("--env", default="ModeTag-A-v0")
p.add_argument("--p-o", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0])
p.add_argument("--p-m", type=float, nargs="+", default=[0.5, 0.8, 0.9, 0.95, 1.0])
p.add_argument("--policies", nargs="+", default=["random", "chase", "qmdp"])
p.add_argument("--reward", default="sparse", choices=["sparse", "dense"])
p.add_argument("--episodes", type=int, default=300)
p.add_argument("--out", default="results/sweep_A.csv")
a = p.parse_args()

make = {"random": lambda: RandomPolicy(0), "chase": lambda: ChaseLastSeenPolicy(0),
        "qmdp": lambda: QMDPPolicy()}
keys = ["discounted_return", "return", "tagged", "length", "capture_time",
        "visible_fraction", "estimation_error", "mode_accuracy"]
rows = []
for p_o, p_m, pol in itertools.product(a.p_o, a.p_m, a.policies):
    _, s = evaluate(make[pol](), a.env, seeds=EVAL_SEEDS[:a.episodes],
                    env_kwargs=dict(p_o=p_o, p_m=p_m, reward=a.reward))
    row = {"policy": pol, "p_o": p_o, "p_m": p_m, "episodes": s["episodes"]}
    for k in keys:
        row[k], row[k + "_se"] = s[k], s[k + "_se"]
    rows.append(row)
    print(f"p_o={p_o:.2f} p_m={p_m:.2f} {pol:6s} disc. return {s['discounted_return']:7.2f} "
          f"capture {s['tagged']:6.1%} length {s['length']:5.1f}")
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", a.out)
