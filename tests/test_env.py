import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import tag_env
from tag_env import TAG, UP, TimeVaryingTagEnv

PRESET_IDS = list(tag_env.PRESETS)


@pytest.mark.parametrize("env_id", PRESET_IDS)
@pytest.mark.parametrize("obs_mode", ["discrete", "vector"])
def test_gym_api(env_id, obs_mode):
    kw = {"obs_mode": obs_mode}
    if obs_mode == "vector":
        kw.update(obs_include_time=True, obs_include_radius=True)
    env = gym.make(env_id, **kw)
    check_env(env.unwrapped, skip_render_check=True)


def test_layout_matches_paper():
    env = TimeVaryingTagEnv()
    g = env.grid
    assert g.n == 29 and env.n_states == 870
    assert tuple(g.coords[0]) == (0, 0) and tuple(g.coords[10]) == (0, 1)
    assert tuple(g.coords[20]) == (5, 2) and tuple(g.coords[28]) == (7, 4)
    # Pr(Robot=s10 | Robot=s0, North) = 1
    assert g.move[0, UP] == 10


@pytest.mark.parametrize("rule", ["axis", "greedy"])
def test_opponent_example_from_paper(rule):
    env = TimeVaryingTagEnv(opponent_rule=rule)
    K = env._opponent_kernel(1, 0.8)
    row = K[0, 15]  # robot at s0, opponent at s15
    assert row[16] == pytest.approx(0.4)
    assert row[20] == pytest.approx(0.4)
    assert row[15] == pytest.approx(0.2)


@pytest.mark.parametrize("env_id", PRESET_IDS)
def test_model_is_stochastic(env_id):
    env = gym.make(env_id).unwrapped
    env.reset(seed=0)
    for t in (0, 17, env.max_steps):
        m = env.model(t)
        np.testing.assert_allclose(m.T.sum(-1), 1.0)
        np.testing.assert_allclose(m.Z.sum(-1), 1.0)
    np.testing.assert_allclose(env.initial_belief().sum(), 1.0)


def test_pineau_only_same_cell_visible():
    env = TimeVaryingTagEnv(visibility_radius=0.5)
    _, info = env.reset(seed=0, options={"robot": 0, "opponent": 1})
    assert not info["opponent_visible"]
    _, info = env.reset(seed=0, options={"robot": 3, "opponent": 3})
    assert info["opponent_visible"]


def test_tag_rewards_and_termination():
    env = TimeVaryingTagEnv(opponent_speed=0)
    env.reset(seed=0, options={"robot": 0, "opponent": 5})
    _, r, term, trunc, _ = env.step(TAG)
    assert r == -10 and not term
    _, r, term, _, _ = env.step(UP)
    assert r == -1 and not term
    env.reset(seed=0, options={"robot": 7, "opponent": 7})
    _, r, term, _, info = env.step(TAG)
    assert r == 10 and term and info["opponent"] == env.TAGGED


def test_static_opponent_never_moves():
    env = gym.make("Tag-Static-v0").unwrapped
    _, info = env.reset(seed=3)
    start = info["opponent"]
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.integers(4)
        *_, info = env.step(a)
        assert info["opponent"] == start


def test_truncation():
    env = TimeVaryingTagEnv(max_steps=5, opponent_speed=0)
    env.reset(seed=0, options={"robot": 0, "opponent": 28})
    for i in range(5):
        _, _, term, trunc, _ = env.step(UP if i % 2 else 1)
    assert trunc and not term


def test_seed_determinism():
    def rollout(seed):
        env = gym.make("Tag-TV-v0").unwrapped
        obs, _ = env.reset(seed=seed)
        out = [obs]
        rng = np.random.default_rng(1)
        for _ in range(40):
            obs, r, term, trunc, _ = env.step(rng.integers(5))
            out.append((obs, r))
            if term or trunc:
                break
        return out, {k: v.copy() for k, v in env.realization.items()}

    a, ra = rollout(11)
    b, rb = rollout(11)
    assert a == b
    for k in ra:
        np.testing.assert_array_equal(ra[k], rb[k])


@pytest.mark.parametrize("reacts", ["pre_move", "post_move"])
@pytest.mark.parametrize("rule", ["axis", "greedy"])
def test_simulator_matches_model(rule, reacts):
    """Monte-Carlo one-step (s', o') frequencies vs. T_t and Z_{t+1}."""
    env = TimeVaryingTagEnv(opponent_rule=rule, opponent_reacts_to=reacts,
        visibility_radius=tag_env.Piecewise([1], [0.5, 2.5]),
        detection_prob=0.7, opponent_speed=2, opponent_move_prob=0.6,
    )
    robot, opp, action = 12, 14, 3  # Right
    N = 20000
    s_counts = np.zeros(env.n_states)
    o_counts = np.zeros(env.n_obs)
    env.reset(seed=0)
    for i in range(N):
        env.reset(options={"robot": robot, "opponent": opp})
        obs, *_ , info = env.step(action)
        s_counts[info["state"]] += 1
        o_counts[obs] += 1
    s0 = robot * 30 + opp
    p_s = env.model(0).T[action, s0]
    p_o = p_s @ env.model(1).Z
    assert np.abs(s_counts / N - p_s).max() < 0.015
    assert np.abs(o_counts / N - p_o).max() < 0.015


def test_belief_update_tracks_truth():
    env = gym.make("Tag-TV-v0").unwrapped
    obs, info = env.reset(seed=5)
    b = env.initial_belief() * env.model(0).Z[:, obs]
    b /= b.sum()
    rng = np.random.default_rng(0)
    for t in range(30):
        a = int(rng.integers(4))
        obs, _, term, trunc, info = env.step(a)
        b = env.belief_update(b, a, obs, t)
        assert b[info["state"]] > 0  # true state always in the support
        if term or trunc:
            break


# --------------------------------------------------------------- modular components
def test_model_matrices_built_from_components():
    env = gym.make("Tag-TV-v0").unwrapped
    env.reset(seed=2)
    t = 7
    m = env.model(t)
    no = env.n_cells + 1
    for s in range(env.n_states):
        state = tag_env.TagState(s // no, s % no)
        for a in range(5):
            assert m.R[s, a] == env.reward(state, a)
        row = np.zeros(env.n_obs)
        for o, p in env.observation_probs(state, None, t).items():
            row[state.robot * no + o] += p
        np.testing.assert_allclose(m.Z[s], row)


def test_step_uses_components():
    """Overriding observation_probs changes what step() emits (single source of truth)."""
    class Blind(TimeVaryingTagEnv):
        def observation_probs(self, state, action, t):
            return {self.UNSEEN: 1.0}
    env = Blind(visibility_radius=100)
    _, info = env.reset(seed=0, options={"robot": 3, "opponent": 3})
    assert not info["opponent_visible"]
    assert env.model(0).Z[3 * 30 + 3, 3 * 30 + 29] == 1.0


# --------------------------------------------------------------------------- metrics
def test_metrics_wrapper_counts():
    env = tag_env.TagMetrics(TimeVaryingTagEnv(opponent_speed=0, visibility_radius=0.5))
    env.reset(seed=0, options={"robot": 0, "opponent": 1})
    assert env.log_estimate(1) == 0.0
    assert env.log_estimate(np.array([2.0, 0.0])) == pytest.approx(1.0)
    env.step(TAG)                    # miss: -10, opponent not visible (o_1)
    env.step(3)                      # Right -> robot at s1, opponent visible (o_2)
    *_, term, trunc, info = env.step(TAG)
    rec = info["episode_metrics"]
    assert term and rec == env.episodes[-1]
    assert rec["return"] == -1 and rec["length"] == 3 and rec["tagged"] == 1
    assert rec["discounted_return"] == pytest.approx(-10 - 0.95 + 10 * 0.95**2)
    assert rec["failed_tags"] == 1
    assert rec["visible_fraction"] == pytest.approx(1 / 3)
    assert rec["estimation_error"] == pytest.approx(0.5)


def test_metrics_belief_estimate():
    env = tag_env.TagMetrics(TimeVaryingTagEnv())
    env.reset(seed=0, options={"robot": 0, "opponent": 9})
    b = np.zeros(env.unwrapped.n_states)
    b[0 * 30 + 9] = 1.0
    assert env.log_estimate(b) == 0.0


def test_evaluate_same_episodes_for_all_policies():
    from tag_env.baselines import QMDPPolicy, RandomPolicy
    seeds = tag_env.EVAL_SEEDS[:5]
    starts = []
    for pol in (RandomPolicy(0), QMDPPolicy()):
        recs, summ = tag_env.evaluate(pol, "Tag-TV-v0", seeds=seeds)
        assert [r["seed"] for r in recs] == list(seeds) and summ["episodes"] == 5
        env = gym.make("Tag-TV-v0").unwrapped
        starts.append([env.reset(seed=s)[1]["state"] for s in seeds])
    assert starts[0] == starts[1]
    assert not np.isnan(summ["estimation_error"])   # QMDP logs its belief
