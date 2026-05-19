"""Integration smoke test: 1 seed x 5K env-steps for DQN and PPO end-to-end.

Verifies that all five instrumentation buckets are produced with content,
under 120 seconds wall-clock per algo. Run via: pytest tests/test_integration_smoke.py -v
"""
import json
import time
from pathlib import Path

import pandas as pd
import pytest

from phase1_ppo_dqn.train import run_one_seed
from phase1_ppo_dqn.presets import PRESETS


@pytest.fixture
def tiny_preset(monkeypatch):
    """Override SMOKE so the integration test finishes in <120s."""
    tiny = {"total_env_steps": 5000, "seeds": [0], "eval_cadence": 2500, "n_eval_episodes": 5}
    monkeypatch.setitem(PRESETS, "smoke", tiny)
    return tiny


def _assert_all_buckets_present(out_dir: Path, algo: str):
    # Bucket #1: train_episodes
    df = pd.read_json(out_dir / "train_episodes.jsonl", lines=True)
    assert len(df) > 0
    assert set(["step", "return", "length", "terminal_reason"]).issubset(df.columns)

    # Bucket #3: eval_checkpoints. Should contain algo rows + reference rows (random + noop at step=0).
    cps = pd.read_json(out_dir / "eval_checkpoints.jsonl", lines=True)
    algos = set(cps["algo"].unique())
    assert algo in algos, f"expected {algo} in eval_checkpoints, got {algos}"
    assert "random" in algos and "noop" in algos

    # Bucket #2: per-step trajectories (at least one checkpoint produced).
    traj_files = list((out_dir / "eval_trajectories").glob("step_*.jsonl"))
    assert len(traj_files) >= 1
    with open(traj_files[0]) as f:
        lines = f.readlines()
    assert len(lines) >= 1
    rec = json.loads(lines[0])
    for k in ("agent_actions", "executed_actions", "rewards_raw", "socs", "num_abnormal_vitals"):
        assert k in rec

    # Bucket #4: training internals.
    ints = pd.read_json(out_dir / "training_internals.jsonl", lines=True)
    assert len(ints) > 0

    # Bucket #5: action_hist_by_soc_x_abnormal embedded in eval_checkpoints.
    sample = json.loads((out_dir / "eval_checkpoints.jsonl").read_text().splitlines()[-1])
    assert "action_hist_by_soc_x_abnormal" in sample
    assert isinstance(sample["action_hist_by_soc_x_abnormal"], dict)
    assert len(sample["action_hist_by_soc_x_abnormal"]) > 0


def test_dqn_integration_smoke(tmp_path, tiny_preset):
    t0 = time.time()
    out_dir = run_one_seed("dqn", "smoke", seed=0, out_root=tmp_path,
                           eval_only=False, run_name="itest", device="cpu")
    elapsed = time.time() - t0
    assert elapsed < 120, f"DQN integration smoke took {elapsed:.1f}s; expected <120s"
    _assert_all_buckets_present(out_dir, algo="dqn")


def test_ppo_integration_smoke(tmp_path, tiny_preset):
    t0 = time.time()
    out_dir = run_one_seed("ppo", "smoke", seed=0, out_root=tmp_path,
                           eval_only=False, run_name="itest", device="cpu")
    elapsed = time.time() - t0
    assert elapsed < 120, f"PPO integration smoke took {elapsed:.1f}s; expected <120s"
    _assert_all_buckets_present(out_dir, algo="ppo")
