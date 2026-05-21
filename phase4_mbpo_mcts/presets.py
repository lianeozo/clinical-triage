"""SMOKE and STANDARD presets for Phase 4 MBPO-MCTS runs.

Each preset is a dict of OuterLoopConfig-shaped knobs plus the seed list.
The CLI (phase4_mbpo_mcts.train) reads these into an OuterLoopConfig and
runs each seed. SMOKE finishes in a few minutes on CPU; STANDARD targets
a full A100 run (~7-8h) and mirrors phase3_iql's STANDARD shape.
"""
from __future__ import annotations

SMOKE = {
    "n_outer_iters": 2,
    "n_episodes_per_iter": 5,
    "n_model_steps_per_iter": 50,
    "n_pv_steps_per_iter": 50,
    "n_eval_episodes": 10,
    "seeds": [0, 1],
}

STANDARD = {
    "n_outer_iters": 15,
    "n_episodes_per_iter": 30,
    "n_model_steps_per_iter": 500,
    "n_pv_steps_per_iter": 500,
    "n_eval_episodes": 50,
    "seeds": [0, 1, 2],
}

PRESETS = {"smoke": SMOKE, "standard": STANDARD}
