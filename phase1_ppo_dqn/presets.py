"""SMOKE and STANDARD presets for Phase 1 runs."""
from __future__ import annotations

SMOKE = {
    "total_env_steps": 1000,
    "seeds": [0, 1],
    "eval_cadence": 100,
    "n_eval_episodes": 20,
}

STANDARD = {
    "total_env_steps": 500_000,
    "seeds": [0, 1, 2, 3, 4],
    "eval_cadence": 25_000,
    "n_eval_episodes": 50,
}

PRESETS = {"smoke": SMOKE, "standard": STANDARD}
