import json
import time
from pathlib import Path

import pandas as pd
import pytest

from triage_rl.logger import Logger


def test_logger_writes_episode_rows(tmp_path):
    log = Logger(tmp_path)
    log.log_episode(step=10, return_=1.5, length=3, terminal_reason="discharge")
    log.log_episode(step=20, return_=-0.5, length=5, terminal_reason="death")
    log.close()

    df = pd.read_json(tmp_path / "train_episodes.jsonl", lines=True)
    assert list(df.columns) == ["step", "return", "length", "terminal_reason", "wall_time"]
    assert df.shape == (2, 5)
    assert df["return"].iloc[0] == 1.5
    assert df["terminal_reason"].iloc[1] == "death"


def test_logger_writes_checkpoint_and_internals(tmp_path):
    log = Logger(tmp_path)
    log.log_checkpoint(step=100, aggregates={"algo": "dqn", "reward_mean": 1.0})
    log.log_internals(step=50, metrics={"loss": 0.3, "q_mean": 0.1})
    log.close()

    cps = pd.read_json(tmp_path / "eval_checkpoints.jsonl", lines=True)
    ints = pd.read_json(tmp_path / "training_internals.jsonl", lines=True)
    assert cps["reward_mean"].iloc[0] == pytest.approx(1.0)
    assert ints["loss"].iloc[0] == pytest.approx(0.3)
    assert "wall_time" in cps.columns and "wall_time" in ints.columns


def test_logger_writes_eval_trajectory(tmp_path):
    log = Logger(tmp_path)
    traj = {"episode_idx": 0, "init_seed": 42, "states": [[0.0]*8]*2,
            "agent_actions": [0, 1], "executed_actions": [0, 1],
            "clamped_steps": [0, 0], "rewards_raw": [0.1, -0.2],
            "socs": [0, 1], "num_abnormal_vitals": [0, 1],
            "terminal_reason": "discharge", "length": 2}
    log.log_eval_trajectory(step=100, episode_idx=0, trajectory=traj)
    log.close()

    p = tmp_path / "eval_trajectories" / "step_100.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["episode_idx"] == 0
    assert rows[0]["terminal_reason"] == "discharge"
