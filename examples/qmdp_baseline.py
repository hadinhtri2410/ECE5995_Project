"""QMDP baseline (Littman et al. 1995) on any preset, using the env's exact model.

Sanity check against Pineau et al. (2003), Table 1: QMDP on Tag gets -16.769 discounted
reward (gamma = 0.95) and a 17% tag rate.

    python examples/qmdp_baseline.py --env Tag-Pineau-v0 --episodes 1000

Shortcut for: python examples/evaluate.py --policy qmdp[-replan] ...
"""
import argparse

from tag_env import EVAL_SEEDS, evaluate
from tag_env.baselines import QMDPPolicy
from tag_env.evaluation import format_summary

p = argparse.ArgumentParser()
p.add_argument("--env", default="Tag-Pineau-v0")
p.add_argument("--episodes", type=int, default=500)
p.add_argument("--replan", action="store_true", help="re-solve QMDP every step with model(t)")
args = p.parse_args()

_, summary = evaluate(QMDPPolicy(replan=args.replan), args.env, seeds=EVAL_SEEDS[:args.episodes])
print(format_summary(f"QMDP on {args.env}", summary))
