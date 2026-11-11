import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import tag_env
from tag_env import ModeTagEnv
from tag_env.grid import DOWN, LEFT, RIGHT, UP
from tag_env.mode_baselines import ChaseLastSeenPolicy, QMDPPolicy, RandomPolicy

MODE_IDS = list(tag_env.MODE_PRESETS)


def cell(env, x, y):
    return env.grid.index(x, y)


@pytest.mark.parametrize("env_id", MODE_IDS)
@pytest.mark.parametrize("obs_mode", ["discrete", "vector"])
def test_gym_api(env_id, obs_mode):
    kw = {"obs_mode": obs_mode}
    if obs_mode == "vector":
        kw["obs_include_time"] = True
    check_env(gym.make(env_id, **kw).unwrapped, skip_render_check=True)


def test_grid_and_spaces():
    env = ModeTagEnv()
    assert env.n_cells == 100 and env.n_hidden == 400 and env.action_space.n == 4
    assert env.max_steps == 100
    assert tuple(env.grid.coords[cell(env, 3, 7)]) == (3, 7)
    corner = cell(env, 0, 0)
    assert env.grid.move[corner, LEFT] == corner and env.grid.move[corner, DOWN] == corner
    assert env.grid.move[corner, UP] == cell(env, 0, 1)


def test_start_positions_differ_and_mode_uniform():
    env = ModeTagEnv()
    modes = []
    for s in range(400):
        _, info = env.reset(seed=s)
        assert info["robot"] != info["opponent"]
        modes.append(info["mode"])
    assert set(modes) == {0, 1, 2, 3}


@pytest.mark.parametrize("p_o,p_m", [(0.8, 0.9), (0.4, 0.6)])
def test_opponent_and_mode_dynamics_match_kernel(p_o, p_m):
    """Monte-Carlo (opponent', mode') frequencies vs. the exact kernel, incl. a wall."""
    env = ModeTagEnv(p_o=p_o, p_m=p_m)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for opp, mode in [(cell(env, 5, 5), RIGHT), (cell(env, 9, 4), RIGHT), (cell(env, 0, 0), DOWN)]:
        state = tag_env.ModeTagState(cell(env, 0, 9), opp, mode)
        N = 40000
        counts = np.zeros(env.n_hidden)
        for _ in range(N):
            s2 = env.transition(state, UP, 0, rng)
            counts[s2.opponent * 4 + s2.mode] += 1
        K = env.opponent_kernel(0)
        np.testing.assert_allclose(K.sum(1), 1.0)
        assert np.abs(counts / N - K[opp * 4 + mode]).max() < 0.01


def test_kernel_values_from_spec():
    env = ModeTagEnv(p_o=0.6, p_m=0.7)
    env.reset(seed=0)
    K = env.opponent_kernel(0).reshape(100, 4, 100, 4)
    c = cell(env, 5, 5)
    move_prob = K[c, RIGHT].sum(1)                     # opponent marginal
    assert move_prob[cell(env, 6, 5)] == pytest.approx(0.6)
    for x, y in [(5, 5), (4, 5), (5, 6), (5, 4)]:      # stay + the three other directions
        assert move_prob[cell(env, x, y)] == pytest.approx(0.1)
    mode_prob = K[c, RIGHT].sum(0)                     # mode marginal
    assert mode_prob[RIGHT] == pytest.approx(0.7)
    assert mode_prob[UP] == pytest.approx(0.1)
    edge = cell(env, 9, 5)                             # pushing into the wall = stay
    assert K[edge, RIGHT].sum(1)[edge] == pytest.approx(0.6 + 0.1)


def test_mode_schedule_is_deterministic():
    env = ModeTagEnv(mode_schedule=lambda t, rng: (t // 10) % 4)
    _, info = env.reset(seed=0)
    assert info["mode"] == 0
    for t in range(25):
        *_, term, trunc, info = env.step(UP)
        if term:
            break
        assert info["mode"] == ((t + 1) // 10) % 4


def test_row_col_observation():
    env = ModeTagEnv(observation="row_col")
    r = cell(env, 2, 3)
    for opp, seen in [(cell(env, 8, 3), True), (cell(env, 2, 9), True), (cell(env, 3, 4), False)]:
        _, info = env.reset(seed=0, options={"robot": r, "opponent": opp})
        assert info["opponent_visible"] == seen


def test_radius_observation_is_inclusive_and_time_varying():
    env = ModeTagEnv(observation="radius", eps=lambda t, rng: np.where(t < 1, 2.0, 0.5))
    r, opp = cell(env, 0, 0), cell(env, 1, 1)          # Manhattan distance 2
    _, info = env.reset(seed=0, options={"robot": r, "opponent": opp, "mode": LEFT})
    assert info["opponent_visible"]                    # d = 2 <= eps(0) = 2
    env2 = ModeTagEnv(observation="radius", eps=1.0, p_o=1.0, p_m=1.0)
    _, info = env2.reset(seed=0, options={"robot": r, "opponent": cell(env, 5, 5), "mode": UP})
    assert not info["opponent_visible"]


def test_rewards_capture_and_truncation():
    env = ModeTagEnv(p_o=1.0, p_m=1.0)                 # opponent moves deterministically
    a, b = cell(env, 4, 4), cell(env, 6, 4)
    env.reset(seed=0, options={"robot": a, "opponent": b, "mode": LEFT})
    _, r, term, _, info = env.step(RIGHT)              # robot -> (5,4), opponent -> (5,4)
    assert term and r == 10 and info["robot"] == info["opponent"]
    with pytest.raises(RuntimeError):
        env.step(UP)

    dense = ModeTagEnv(p_o=1.0, p_m=1.0, reward="dense")
    dense.reset(seed=0, options={"robot": cell(env, 0, 0), "opponent": cell(env, 5, 5), "mode": UP})
    _, r, *_ = dense.step(RIGHT)                       # robot (1,0), opponent (5,6): d = 10
    assert r == -10

    sparse = ModeTagEnv(p_o=1.0, p_m=1.0)
    sparse.reset(seed=0, options={"robot": cell(env, 0, 0), "opponent": cell(env, 9, 9), "mode": UP})
    for i in range(100):
        _, r, term, trunc, _ = sparse.step(LEFT)
        assert r == -1 and not term
    assert trunc


def test_observation_never_leaks_hidden_info():
    """With the opponent hidden, the observation is the same whatever the mode / eps."""
    obs = []
    for mode, eps in [(UP, 1.0), (RIGHT, 1.5)]:
        env = ModeTagEnv(observation="radius", eps=eps, obs_mode="vector", obs_include_time=True)
        o, _ = env.reset(seed=0, options={"robot": 0, "opponent": 99, "mode": mode})
        obs.append(o)
    np.testing.assert_array_equal(obs[0], obs[1])
    assert obs[0].shape == (100 + 101 + 1,)


def test_seed_determinism():
    def rollout(seed):
        env = gym.make("ModeTag-B-v0").unwrapped
        o, _ = env.reset(seed=seed)
        out = [o]
        rng = np.random.default_rng(1)
        for _ in range(100):
            o, r, term, trunc, info = env.step(int(rng.integers(4)))
            out.append((o, r, info["hidden_index"]))
            if term or trunc:
                break
        return out
    assert rollout(3) == rollout(3)


@pytest.mark.parametrize("env_id", MODE_IDS)
def test_belief_tracks_truth(env_id):
    env = gym.make(env_id).unwrapped
    rng = np.random.default_rng(0)
    for seed in range(5):
        obs, info = env.reset(seed=seed)
        b = env.initial_belief(*divmod(obs, 101))
        assert b[info["hidden_index"]] > 0
        for t in range(100):
            obs, _, term, trunc, info = env.step(int(rng.integers(4)))
            if term or trunc:
                break
            b = env.belief_update(b, info["robot"], obs % 101, t)
            assert b[info["hidden_index"]] > 0 and b.sum() == pytest.approx(1.0)


def test_baselines_and_metrics():
    seeds = tag_env.EVAL_SEEDS[:30]
    res = {}
    for name, pol in [("random", RandomPolicy(0)), ("chase", ChaseLastSeenPolicy(0)),
                      ("qmdp", QMDPPolicy())]:
        recs, summ = tag_env.evaluate(pol, "ModeTag-A-v0", seeds=seeds)
        assert [r["seed"] for r in recs] == list(seeds)
        assert all(np.isnan(r["failed_tags"]) for r in recs)
        res[name] = summ
    assert res["qmdp"]["discounted_return"] > res["random"]["discounted_return"]
    assert 0.0 <= res["qmdp"]["mode_accuracy"] <= 1.0
    assert not np.isnan(res["qmdp"]["capture_time"])
