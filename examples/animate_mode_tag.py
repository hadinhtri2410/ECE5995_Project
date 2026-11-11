"""Animate one ModeTag episode as a GIF: the true game (left) and the policy's belief about
the opponent's position (right, QMDP only).

    python examples/animate_mode_tag.py --env ModeTag-A-v0 --policy qmdp --out results/anim_A.gif
    python examples/animate_mode_tag.py --env ModeTag-B-v0 --policy chase --seed 100004

Left panel:  blue = robot, red = opponent (solid when visible to the robot, faded when
             hidden), white arrow = opponent's hidden mode, yellow = cells the robot can
             currently see. Right panel: belief P(opponent in cell), darker = more likely.
Without --seed, picks the first evaluation seed whose episode lasts at least --min-length steps.
"""
import argparse

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import tag_env  # noqa: F401
from tag_env import EVAL_SEEDS
from tag_env.mode_baselines import ChaseLastSeenPolicy, QMDPPolicy, RandomPolicy
from tag_env.mode_tag import MODE_NAMES

CELL, PAD, HEAD = 36, 16, 70
POLICIES = {"qmdp": QMDPPolicy, "chase": ChaseLastSeenPolicy, "random": RandomPolicy}
ARROW = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}   # up/down/left/right in image coords


def run_episode(env_id, policy_name, seed):
    env = gym.make(env_id).unwrapped
    pol = POLICIES[policy_name]()
    obs, info = env.reset(seed=seed)
    pol.reset(env, obs)
    frames = [snapshot(env, info, pol, None, 0.0)]
    G = 0.0
    while True:
        a = pol.act(obs)
        obs, r, term, trunc, info = env.step(a)
        G += r
        if not (term or trunc):
            pol.update(a, obs)
        frames.append(snapshot(env, info, pol, (a, r), G, term, trunc))
        if term or trunc:
            return env, frames


def snapshot(env, info, pol, last, G, term=False, trunc=False):
    belief = pol.opponent_estimate() if not (term or trunc) else None
    if belief is not None:
        belief = np.asarray(belief).reshape(env.n_cells, env.n_modes).sum(1)
    return dict(robot=info["robot"], opp=info["opponent"], mode=info["mode"],
                visible=info["opponent_visible"], vis_cells=env._visible_cells().copy(),
                eps=info.get("eps"), t=info["t"], last=last, G=G, term=term, trunc=trunc,
                belief=belief)


def draw(env, f, title, font, small):
    n = env.size
    grid_px = n * CELL
    has_belief = f["belief"] is not None or title.startswith("qmdp")
    W = PAD * 3 + grid_px * (2 if has_belief else 1) if has_belief else PAD * 2 + grid_px
    img = Image.new("RGB", (W, HEAD + grid_px + PAD + 18), (250, 250, 247))
    d = ImageDraw.Draw(img)

    head = f"{title}   t = {f['t']}"
    if f["eps"] is not None:
        head += f"   eps(t) = {f['eps']:.2f}"
    d.text((PAD, 8), head, fill=(30, 30, 30), font=font)
    line2 = f"hidden mode: {MODE_NAMES[f['mode']]}   return: {f['G']:+g}"
    if f["last"]:
        line2 += f"   last: {tag_env.ACTION_NAMES[f['last'][0]]} ({f['last'][1]:+g})"
    d.text((PAD, 30), line2, fill=(70, 70, 70), font=small)
    status = "CAPTURED!" if f["term"] else "time limit" if f["trunc"] else \
        ("opponent visible" if f["visible"] else "opponent hidden")
    d.text((PAD, 48), status, font=small,
           fill=(20, 140, 60) if f["term"] else (190, 40, 40) if f["visible"] else (120, 120, 120))

    def cell_box(ox, i, pad=1):
        x, y = env.grid.coords[i]
        px, py = ox + x * CELL, HEAD + (n - 1 - y) * CELL
        return [px + pad, py + pad, px + CELL - pad, py + CELL - pad]

    # left: true game
    ox = PAD
    for i in range(env.n_cells):
        d.rectangle(cell_box(ox, i), fill=(250, 232, 140) if f["vis_cells"][i] else (226, 226, 222))
    if not f["term"]:
        d.rectangle(cell_box(ox, f["opp"], 5), fill=(210, 45, 45) if f["visible"] else (240, 170, 170))
        cx, cy = [(a + b) / 2 for a, b in zip(cell_box(ox, f["opp"])[:2], cell_box(ox, f["opp"])[2:])]
        dx, dy = ARROW[f["mode"]]
        tip = (cx + dx * 11, cy + dy * 11)
        d.line([(cx - dx * 9, cy - dy * 9), tip], fill=(255, 255, 255), width=3)
        d.polygon([tip, (tip[0] - dx * 6 - dy * 5, tip[1] - dy * 6 - dx * 5),
                   (tip[0] - dx * 6 + dy * 5, tip[1] - dy * 6 + dx * 5)], fill=(255, 255, 255))
    d.ellipse(cell_box(ox, f["robot"], 8), fill=(40, 90, 210) if not f["term"] else (20, 140, 60))
    d.text((ox, HEAD + grid_px + 4), "true state", fill=(90, 90, 90), font=small)

    # right: belief
    if has_belief:
        ox = PAD * 2 + grid_px
        b = f["belief"]
        bmax = b.max() if b is not None and b.max() > 0 else 1.0
        for i in range(env.n_cells):
            if b is None:
                col = (226, 226, 222)
            else:
                v = (b[i] / bmax) ** 0.5
                col = tuple(int(255 - v * (255 - c)) for c in (88, 40, 150))
            d.rectangle(cell_box(ox, i), fill=col)
        if not f["term"]:
            d.rectangle(cell_box(ox, f["opp"], 2), outline=(210, 45, 45), width=2)
        d.ellipse(cell_box(ox, f["robot"], 8), fill=(40, 90, 210))
        d.text((ox, HEAD + grid_px + 4), "belief P(opponent cell); red box = truth",
               fill=(90, 90, 90), font=small)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="ModeTag-A-v0")
    p.add_argument("--policy", choices=list(POLICIES), default="qmdp")
    p.add_argument("--seed", type=int)
    p.add_argument("--min-length", type=int, default=20)
    p.add_argument("--ms", type=int, default=450, help="milliseconds per frame")
    p.add_argument("--out", default="results/animation.gif")
    a = p.parse_args()

    seeds = [a.seed] if a.seed is not None else EVAL_SEEDS
    for seed in seeds:
        env, frames = run_episode(a.env, a.policy, seed)
        if a.seed is not None or len(frames) - 1 >= a.min_length:
            break
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    title = f"{a.policy} on {a.env} (seed {seed})"
    imgs = [draw(env, f, title, font, small) for f in frames]
    durations = [a.ms] * (len(imgs) - 1) + [2500]
    imgs[0].save(a.out, save_all=True, append_images=imgs[1:], duration=durations, loop=0)
    print(f"seed {seed}: {len(frames) - 1} steps, return {frames[-1]['G']:+g}, "
          f"{'captured' if frames[-1]['term'] else 'not captured'} -> {a.out}")


if __name__ == "__main__":
    main()
