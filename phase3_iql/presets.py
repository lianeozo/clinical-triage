"""SMOKE_IQL and STANDARD_IQL presets for Part 3 runs."""
from __future__ import annotations

SMOKE_IQL = {
    "total_grad_steps": 20_000,
    "seeds": [0, 1, 2],
    "eval_cadence": 4_000,
    "n_eval_episodes": 50,
}

STANDARD_IQL = {
    "total_grad_steps": 200_000,
    "seeds": [0, 1, 2, 3, 4],
    "eval_cadence": 25_000,
    "n_eval_episodes": 50,
}

PRESETS = {"smoke": SMOKE_IQL, "standard": STANDARD_IQL}
