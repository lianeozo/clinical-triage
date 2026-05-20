"""SMOKE_SAC and STANDARD_SAC presets for Phase 2 runs."""
from __future__ import annotations

SMOKE_SAC = {
    "total_env_steps": 100_000,
    "seeds": [0, 1, 2],
    "eval_cadence": 10_000,
    "n_eval_episodes": 50,
}

STANDARD_SAC = {
    "total_env_steps": 500_000,
    "seeds": [0, 1, 2, 3, 4],
    "eval_cadence": 25_000,
    "n_eval_episodes": 50,
}

PRESETS = {"smoke": SMOKE_SAC, "standard": STANDARD_SAC}
