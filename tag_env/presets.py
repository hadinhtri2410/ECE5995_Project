"""Named configurations, registered as Gymnasium ids in ``tag_env/__init__.py``.

Final-report game (ModeTagEnv, 10 x 10 open grid, opponent driven by a hidden mode):

    ModeTag-A-v0    Exp. A: time-varying hidden state (mode switching, p_o / p_m),
                    opponent seen in the same row or column
    ModeTag-B-v0    Exp. B: time-varying observation radius eps(t) (placeholder
                    tag_env.mode_tag.simple_eps), agent never sees eps(t)

Legacy Pineau et al. (2003) Tag (TimeVaryingTagEnv):

    Tag-Pineau-v0   original static Tag (opponent seen only in the same cell)
    Tag-Static-v0   opponent never moves (stationary hidden state)
    Tag-TVObs-v0    time-varying observation function d(t) (+ imperfect detection)
    Tag-TVOpp-v0    time-varying hidden-state dynamics (opponent speed / move prob)
    Tag-TV-v0       both time-varying observation and dynamics
"""
from .mode_tag import simple_eps
from .schedules import IID, RegimeSwitching, Sinusoid

MODE_PRESETS = {
    "ModeTag-A-v0": dict(observation="row_col", p_o=0.8, p_m=0.9, reward="sparse"),
    "ModeTag-B-v0": dict(observation="radius", eps=simple_eps, p_o=0.8, p_m=0.9, reward="sparse"),
}

PRESETS = {
    "Tag-Pineau-v0": dict(
        visibility_radius=0.5, detection_prob=1.0, opponent_speed=1, opponent_move_prob=0.8,
    ),
    "Tag-Static-v0": dict(
        visibility_radius=0.5, opponent_speed=0,
    ),
    "Tag-TVObs-v0": dict(
        # Sensor range switches between blind / short / medium / long regimes.
        visibility_radius=RegimeSwitching([0.5, 1.5, 3.0, 5.0], hazard=0.05),
        detection_prob=IID(low=0.7, high=1.0),
        opponent_speed=1, opponent_move_prob=0.8,
    ),
    "Tag-TVOpp-v0": dict(
        visibility_radius=0.5,
        # Opponent alternates between frozen, normal and fast regimes.
        opponent_speed=RegimeSwitching([0, 1, 2], hazard=0.05),
        opponent_move_prob=Sinusoid(mean=0.6, amplitude=0.3, period=40, low=0.0, high=1.0),
    ),
    "Tag-TV-v0": dict(
        visibility_radius=RegimeSwitching([0.5, 1.5, 3.0, 5.0], hazard=0.05),
        detection_prob=IID(low=0.7, high=1.0),
        opponent_speed=RegimeSwitching([0, 1, 2], hazard=0.05),
        opponent_move_prob=Sinusoid(mean=0.6, amplitude=0.3, period=40, low=0.0, high=1.0),
    ),
}
