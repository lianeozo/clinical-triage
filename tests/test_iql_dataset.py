"""Unit tests for phase3_iql.dataset (offline dataset assembly)."""
import json
from pathlib import Path

import numpy as np
import pytest

from phase3_iql.dataset import assemble_offline_dataset, _trajectory_to_transitions


def _make_traj_record(algo: str, length: int, terminal_reason: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "algo": algo,
        "episode_idx": 0,
        "init_seed": seed,
        "states":            [rng.standard_normal(8).astype(np.float32).tolist() for _ in range(length)],
        "agent_actions":     [int(rng.integers(0, 32)) for _ in range(length)],
        "executed_actions":  [int(rng.integers(0, 32)) for _ in range(length)],
        "clamped_steps":     [0 for _ in range(length)],
        "rewards_raw":       [float(rng.standard_normal()) for _ in range(length)],
        "socs":              [int(rng.integers(0, 4)) for _ in range(length)],
        "num_abnormal_vitals": [int(rng.integers(0, 5)) for _ in range(length)],
        "terminal_reason": terminal_reason,
        "length": length,
    }


def _write_trajectory_file(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_trajectory_to_transitions_count_and_shape():
    """A length-5 episode yields 5 transitions."""
    rec = _make_traj_record("sac", length=5, terminal_reason="discharge")
    rows = _trajectory_to_transitions(rec)
    assert len(rows) == 5
    for row in rows:
        assert row["obs"].shape == (8,)
        assert row["next_obs"].shape == (8,)
        assert isinstance(row["action"], int)
        assert isinstance(row["reward"], float)
        assert isinstance(row["terminated"], bool)


def test_terminated_flag_at_last_step_discharge():
    """Last transition has terminated=True for discharge endings."""
    rec = _make_traj_record("sac", length=3, terminal_reason="discharge")
    rows = _trajectory_to_transitions(rec)
    assert rows[0]["terminated"] is False
    assert rows[1]["terminated"] is False
    assert rows[2]["terminated"] is True


def test_terminated_flag_at_last_step_death():
    rec = _make_traj_record("sac", length=2, terminal_reason="death")
    rows = _trajectory_to_transitions(rec)
    assert rows[-1]["terminated"] is True


def test_terminated_flag_at_last_step_timeout():
    """Spec §6.1: timeout endings also treated as terminated=True (offline-RL standard handling)."""
    rec = _make_traj_record("sac", length=2, terminal_reason="timeout")
    rows = _trajectory_to_transitions(rec)
    assert rows[-1]["terminated"] is True


def test_next_obs_at_last_step_is_self():
    """Last step has no recorded next state; we set next_obs := obs (terminated zeros bootstrap anyway)."""
    rec = _make_traj_record("sac", length=2, terminal_reason="discharge")
    rows = _trajectory_to_transitions(rec)
    np.testing.assert_array_equal(rows[-1]["obs"], rows[-1]["next_obs"])


def test_uses_executed_actions(tmp_path):
    """Action field uses executed_actions (post-F4a-clamp), not agent_actions."""
    rec = _make_traj_record("sac", length=3, terminal_reason="discharge")
    rec["agent_actions"] = [8, 8, 8]
    rec["executed_actions"] = [0, 0, 0]
    rows = _trajectory_to_transitions(rec)
    assert [r["action"] for r in rows] == [0, 0, 0]


def test_assemble_with_supplement(tmp_path):
    """assemble_offline_dataset combines trajectory files + random transitions."""
    pull_root = tmp_path / "phase1_ppo_dqn" / "_modal_pull"
    for seed in (0, 1):
        path = pull_root / "2026-01-01T00-00-standard-sac" / f"seed_{seed}" / "eval_trajectories" / "step_500000.jsonl"
        recs = [
            _make_traj_record("sac", length=5, terminal_reason="discharge", seed=seed),
            _make_traj_record("sac", length=3, terminal_reason="death", seed=seed + 10),
        ]
        _write_trajectory_file(path, recs)

    data = assemble_offline_dataset(pull_root, run_pattern="*-standard-*",
                                    n_random_transitions=100, random_seed=0)

    # 2 seeds × (5 + 3 transitions) = 16 from trained, plus 100 random = 116 total
    assert len(data['action']) == 116
    assert data['obs'].shape == (116, 8)
    assert data['next_obs'].shape == (116, 8)
    assert data['action'].dtype == np.int64
    assert data['reward'].dtype == np.float32
    assert data['terminated'].dtype == np.bool_
    assert len(data['source_algo']) == 116
    assert (data['source_algo'] == "random").sum() == 100
    assert (data['source_algo'] == "sac").sum() == 16
