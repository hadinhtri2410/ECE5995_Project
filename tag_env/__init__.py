from gymnasium.envs.registration import register, registry

from .env import POMDPModel, TagState, TimeVaryingTagEnv
from .grid import ACTION_NAMES, DOWN, LEFT, RIGHT, TAG, UP, TagGrid
from .mode_tag import ModeTagEnv, ModeTagState, simple_eps
from .presets import MODE_PRESETS, PRESETS
from .metrics import TagMetrics, summarize
from .evaluation import EVAL_SEEDS, Policy, evaluate
from .schedules import (IID, Constant, FunctionSchedule, PerEpisode, Piecewise, RandomWalk,
                        RegimeSwitching, Schedule, Sinusoid)

# max_steps is enforced inside the envs (truncated=True), so no TimeLimit wrapper.
for _id, _kwargs in MODE_PRESETS.items():
    if _id not in registry:
        register(id=_id, entry_point="tag_env.mode_tag:ModeTagEnv", kwargs=_kwargs)
for _id, _kwargs in PRESETS.items():
    if _id not in registry:
        register(id=_id, entry_point="tag_env.env:TimeVaryingTagEnv", kwargs=_kwargs)
