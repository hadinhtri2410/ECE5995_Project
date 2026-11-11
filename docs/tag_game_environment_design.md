# Tag Game Environment Design Notes

These notes describe the environment for Sec. IV of the final report and the choices made
where the report left things open. Implementation: `ModeTagEnv` in `tag_env/mode_tag.py`.
The original Pineau et al. (2003) Tag game (`TimeVaryingTagEnv`) is kept only as a legacy
reference.

## 1. Core POMDP Components

Model the environment as

\[
\mathcal{M} = \{S, A, O, \mu_0, P_t, Z_t, R, \gamma\}.
\]

### State \(S\)

\[
s_t = (x_t, y_t, m_t),
\]

where \(x_t \in \{0,\dots,99\}\) is the robot's cell, \(y_t \in \{0,\dots,99\}\) the
opponent's cell, and \(m_t \in \{\text{up},\text{down},\text{left},\text{right}\}\) the
opponent's mode, on an open 10 × 10 grid (cell \(= 10\,y + x\)). That gives
\(|S| = 100 \cdot 100 \cdot 4 = 40{,}000\) states. The simulator knows the full state; the
policy does not.

### Action \(A\)

\[
A = \{\text{Up, Down, Left, Right}\}.
\]

There is no Tag action: capture happens by moving onto the opponent's cell.

### Initial Distribution \(\mu_0\)

Robot and opponent cells are independent and uniform, with the opponent re-drawn until it
differs from the robot (so an episode never starts already captured). The mode is uniform.

### Transition Model \(P_t\)

\[
s_{t+1} \sim P_t(\cdot \mid s_t, a_t), \quad \text{in this order:}
\]

1. **Robot:** moves deterministically by \(a_t\); a move into the boundary means staying.
2. **Opponent:** ignores the robot. With probability \(p_o(t)\) it moves in the direction
   of \(m_t\). Otherwise it stays or moves in one of the other three directions, each with
   probability \((1-p_o)/4\). A move into the boundary means staying.
3. **Mode:** \(m_{t+1} = m_t\) with probability \(p_m(t)\), otherwise one of the other three
   modes uniformly. Alternatively, a fixed schedule \(m(t)\) (`mode_schedule`).

The hidden-state dynamics are time-varying because the hidden mode keeps changing, and
optionally because \(p_o(t)\) and \(p_m(t)\) follow a schedule.

**Meaning of the knobs.** These correct the wording in the report draft:

- larger \(p_o\) ⇒ the opponent follows its mode more reliably ⇒ **more predictable**;
- larger \(p_m\) ⇒ the mode stays the same more often ⇒ it **switches less often**
  (expected run length \(1/(1-p_m)\) steps).

### Observation \(O\)

\[
o_t = (x_t, \tilde{y}_t), \qquad
\tilde{y}_t =
\begin{cases}
y_t, & \text{opponent visible},\\
-1\ (\text{hidden}), & \text{otherwise}.
\end{cases}
\]

The robot's own cell is always observed. The mode and \(\varepsilon_t\) are never observed.

### Observation Function \(Z_t\)

Two visibility rules, one per experiment:

- **A (`observation="row_col"`):** visible iff robot and opponent share a row or a column.
- **B (`observation="radius"`):** visible iff \(d(x_t, y_t) \le \varepsilon_t\), with
  \(d\) = Manhattan distance and an inclusive \(\le\).

**\(\varepsilon_t\) is a placeholder for now.** The default is a simple deterministic
function, \(\varepsilon_t = 3 + 2\sin(2\pi t/50)\) (range 1 to 5, two cycles per episode),
until its final form is decided. The constant-baseline range \([d_1, d_2]\) and the
"band of thickness \(2T\)" trajectory bank from the report are **not implemented yet**.
\(\varepsilon_t\) can be any number, schedule or function `fn(t, rng)`.

### Reward \(R\)

Computed on the post-move state \(s_{t+1}\):

- capture (\(x_{t+1} = y_{t+1}\)): \(+10\);
- otherwise, `reward="sparse"` (default, used first): \(-1\);
- otherwise, `reward="dense"` (optional): \(-d(x_{t+1}, y_{t+1})\).

### Termination

Capture is checked **after both agents move**. On capture the episode terminates.
Otherwise it is truncated after **100 steps**.

### Discount Factor

\(\gamma = 0.95\). The environment returns raw rewards; \(\gamma\) is used by the exact model
(QMDP) and the discounted-return metric.

---

## 2. Most Important Implementation Rule

Keep the **true state** separate from the **agent observation**.

The policy receives only \(o_t = (x_t, \tilde{y}_t)\), plus optionally \(t\). The opponent's
cell when hidden, the mode, \(p_o\), \(p_m\) and \(\varepsilon_t\) appear only in the
privileged `info` dict, which is for teacher–student / asymmetric-critic training and
evaluation bookkeeping. A test checks that the observation doesn't change when the hidden
mode or \(\varepsilon\) changes.

---

## 3. Environment Step Structure

\[
(s_t,a_t,t) \rightarrow s_{t+1} \rightarrow r_t,\text{done} \rightarrow o_{t+1}.
\]

Modular functions (the only definitions of the model; `step()` and the exact model both use
them):

```text
reset()
transition(state, action, t, rng)      # P_t
reward(state, action, next_state)      # R
observation_probs(state, action, t)    # Z_t
observe(state, action, t, rng)         # sample from Z_t
```

Exact model for planners: since the robot is observed and the opponent ignores it, the
belief covers only \(h = (y, m)\) (400 values):
`opponent_kernel(t)` \(= K_t[h, h']\), `initial_belief`, `belief_update`, `reward_table`.

---

## 4. Time-Varying Components

The two kinds of non-stationarity are controlled independently.

### Time-Varying Hidden-State Dynamics (experiment A)

The hidden mode \(m_t\) switches according to \(p_m\), and the opponent follows it according
to \(p_o\). Both can also be schedules \(p_o(t), p_m(t)\), or the mode can be a fixed
schedule \(m(t)\).

### Time-Varying Observation Function (experiment B)

The visibility radius \(\varepsilon_t\) (placeholder function above). The agent never knows
\(\varepsilon_t\).

---

## 5. Main Experimental Difficulty Axes

1. **Opponent predictability:** \(p_o\).
2. **State-dynamics non-stationarity:** how often the mode switches, \(1 - p_m\).
3. **Observability:** row/column rule (A), or the level of \(\varepsilon_t\) (B).
4. **Observation non-stationarity:** how fast \(\varepsilon_t\) changes (B; to be designed).
5. **Memory requirement:** how long the policy must carry the last sighting and the
   inferred mode (follows from the observability and \(p_m\)).

---

## 6. Experiment Groups

### A — Time-varying hidden-state dynamics

`ModeTag-A-v0`: row/column observation, sweep \(p_o \times p_m\)
(`examples/sweep_mode_tag.py`, default \(p_o \in \{0.4, 0.6, 0.8, 1.0\}\),
\(p_m \in \{0.5, 0.8, 0.9, 0.95, 1.0\}\)).

### B — Time-varying observation function

`ModeTag-B-v0`: radius observation with \(\varepsilon_t\), opponent dynamics fixed at
\(p_o = 0.8, p_m = 0.9\), so only the observation side varies. The \(\varepsilon_t\) family
(constant baseline, bounded trajectories, shared trajectory bank) is still to be designed.

---

## 7. Metrics to Log

Logged by the `TagMetrics` wrapper, for every episode:

- cumulative return and discounted return
- capture rate, and time to capture (length of captured episodes)
- episode length
- fraction of decision steps where the opponent is visible
- opponent-position estimation error \(\|\hat{y}_t - y_t\|\) (Euclidean, grid units), if
  the method reports an estimate
- mode accuracy (most likely mode = true mode), if the method reports a belief over
  \((y, m)\)

The estimation metrics separate bad hidden-state estimation from bad control.

---

## 8. Fair Comparison Between Algorithms

All methods are evaluated with `evaluate()` on the shared seeds `EVAL_SEEDS`
(100000–100999, never used for training). A seed fixes the start state, the initial mode and
the schedule realisation, so every method faces the same episodes. Keep these identical
across methods:

- evaluation seeds
- initial-state distribution
- \(p_o, p_m\) and \(\varepsilon_t\) settings
- reward type (sparse / dense)
- episode length (100) and termination rule

---

## 9. Main Goal of the Environment

The environment should make it possible to answer:

> How well do existing RL/POMDP methods handle a setting where the hidden-state dynamics
> (experiment A) and the observation function (experiment B) change over time?

---

## 10. Open Items

- Design of \(\varepsilon_t\) for experiment B: constant-baseline range, bounded / banded
  trajectories, and a trajectory bank shared across algorithms.
- Difficulty: with the current defaults the chase-last-seen heuristic captures 94–100% of
  the time across the whole \(p_o \times p_m\) sweep. The grid size, horizon or
  observability may need tuning so that the methods differ.
- Learning baselines (DQN with frame stacking, DRQN / recurrent PPO, DynaMITE-RL).
