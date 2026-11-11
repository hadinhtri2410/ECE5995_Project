"""Evaluate a baseline on the shared evaluation seeds and log the metrics.

    python examples/evaluate.py --env ModeTag-A-v0 --policy qmdp --episodes 500
    python examples/evaluate.py --env ModeTag-B-v0 --policy chase --out results/chase_B.csv
    python examples/evaluate.py --env ModeTag-A-v0 --policy qmdp --set p_o=0.6 --set reward=dense
    python examples/evaluate.py --env Tag-Pineau-v0 --policy qmdp      # legacy game

Every method evaluated with the same --env / --set / --episodes sees exactly the same start
states and schedule realisations (tag_env.EVAL_SEEDS).
"""
import argparse
import ast

import gymnasium as gym

import tag_env
from tag_env import EVAL_SEEDS, ModeTagEnv, evaluate
from tag_env import baselines as pineau
from tag_env import mode_baselines as mode
from tag_env.evaluation import format_summary, save_records

POLICIES = {
    "mode": {
        "random": lambda a: mode.RandomPolicy(seed=a.policy_seed),
        "chase": lambda a: mode.ChaseLastSeenPolicy(seed=a.policy_seed),
        "qmdp": lambda a: mode.QMDPPolicy(replan=False),
        "qmdp-replan": lambda a: mode.QMDPPolicy(replan=True),
    },
    "pineau": {
        "random": lambda a: pineau.RandomPolicy(seed=a.policy_seed),
        "qmdp": lambda a: pineau.QMDPPolicy(replan=False),
        "qmdp-replan": lambda a: pineau.QMDPPolicy(replan=True),
    },
}


def parse_set(items):
    out = {}
    for item in items or []:
        k, v = item.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


p = argparse.ArgumentParser()
p.add_argument("--env", default="ModeTag-A-v0")
p.add_argument("--policy", default="qmdp")
p.add_argument("--episodes", type=int, default=500, help=f"first N of EVAL_SEEDS (max {len(EVAL_SEEDS)})")
p.add_argument("--policy-seed", type=int, default=0)
p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override an env argument")
p.add_argument("--out", help="write per-episode records to .csv or .json")
a = p.parse_args()

game = "mode" if isinstance(gym.make(a.env).unwrapped, ModeTagEnv) else "pineau"
if a.policy not in POLICIES[game]:
    p.error(f"policy for {a.env} must be one of {list(POLICIES[game])}")
overrides = parse_set(a.set)
records, summary = evaluate(POLICIES[game][a.policy](a), a.env, seeds=EVAL_SEEDS[:a.episodes],
                            env_kwargs=overrides)
print(format_summary(f"{a.policy} on {a.env} {overrides or ''}", summary))
if a.out:
    save_records(records, a.out, meta={**vars(a), "overrides": overrides})
    print("wrote", a.out)
