"""Unit tests for analysis.aggregate."""
import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.aggregate import aggregate_run_dirs


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_checkpoint_row(algo: str, step: int, soc_abn_hist: dict | None = None) -> dict:
    return {
        "step": step, "algo": algo, "n_episodes": 50,
        "reward_mean": float(-1000 * step / 1000),
        "reward_std": 100.0,
        "mortality_rate": 0.5 - step / 1_000_000,
        "discharge_rate": 0.3 + step / 1_000_000,
        "timeout_rate": 0.2,
        "ep_length_mean": 25.0,
        "clamp_rate": 0.01,
        "action_hist": [0] * 32,
        "soc_dwell_fractions": [0.25, 0.25, 0.25, 0.25],
        "soc_transition_counts": [[1, 0, 0, 0]] * 4,
        "action_hist_by_soc_x_abnormal": soc_abn_hist or {"soc_0_abn_0": [0] * 32},
        "wall_time": 1700000000.0,
    }


def test_learning_curves_long_form(tmp_path):
    """One row per (run, algo, seed, step)."""
    run_dir = tmp_path / "phase1_ppo_dqn" / "2026-01-01T00-00-standard-dqn"
    for seed in (0, 1):
        rows = [
            _make_checkpoint_row("random", 0),
            _make_checkpoint_row("noop", 0),
            _make_checkpoint_row("dqn", 10000),
            _make_checkpoint_row("dqn", 20000),
        ]
        _write_jsonl(run_dir / f"seed_{seed}" / "eval_checkpoints.jsonl", rows)

    lc, ad = aggregate_run_dirs(tmp_path / "phase1_ppo_dqn", run_pattern="*-standard-*")

    assert len(lc) == 8  # 2 seeds × (2 algo checkpoints + 2 references)
    expected_cols = {"run_id", "algo", "seed", "step", "n_episodes",
                     "reward_mean", "reward_std", "mortality_rate",
                     "discharge_rate", "timeout_rate", "ep_length_mean", "clamp_rate"}
    assert expected_cols.issubset(set(lc.columns))
    assert set(lc["algo"].unique()) == {"random", "noop", "dqn"}
    assert set(lc["seed"].unique()) == {0, 1}
    # References at step=0 only.
    refs = lc[lc["algo"].isin(["random", "noop"])]
    assert (refs["step"] == 0).all()


def test_action_distributions_fractions_sum_to_one(tmp_path):
    """fraction must sum to 1.0 per (run, seed, step, soc, num_abnormal) group."""
    # Build a histogram where action 5 has all the mass for (soc=2, abn=3) at step=10000.
    hist = [0] * 32
    hist[5] = 7
    soc_abn = {"soc_2_abn_3": hist, "soc_0_abn_0": [1] * 32}  # second key has uniform fractions
    rows = [_make_checkpoint_row("dqn", 10000, soc_abn_hist=soc_abn)]

    run_dir = tmp_path / "phase1_ppo_dqn" / "2026-01-01T00-00-standard-dqn"
    _write_jsonl(run_dir / "seed_0" / "eval_checkpoints.jsonl", rows)

    lc, ad = aggregate_run_dirs(tmp_path / "phase1_ppo_dqn", run_pattern="*-standard-*")

    expected_cols = {"run_id", "algo", "seed", "step", "soc",
                     "num_abnormal", "action_idx", "count", "fraction"}
    assert expected_cols.issubset(set(ad.columns))

    # All fractions sum to 1.0 within each group.
    grouped = ad.groupby(["run_id", "seed", "step", "soc", "num_abnormal"])["fraction"].sum()
    for total in grouped:
        assert abs(total - 1.0) < 1e-6, f"fractions sum to {total}, expected 1.0"

    # The (soc=2, abn=3) action_idx=5 row should have fraction = 1.0 (all mass on action 5).
    row = ad[(ad["soc"] == 2) & (ad["num_abnormal"] == 3) & (ad["action_idx"] == 5)].iloc[0]
    assert row["count"] == 7
    assert abs(row["fraction"] - 1.0) < 1e-6


def test_reference_rows_tagged_with_seed(tmp_path):
    """Reference rows must carry the training seed of the run they belong to."""
    run_dir = tmp_path / "phase1_ppo_dqn" / "2026-01-01T00-00-standard-dqn"
    for seed in (0, 1, 2):
        rows = [_make_checkpoint_row("random", 0), _make_checkpoint_row("noop", 0)]
        _write_jsonl(run_dir / f"seed_{seed}" / "eval_checkpoints.jsonl", rows)

    lc, _ = aggregate_run_dirs(tmp_path / "phase1_ppo_dqn", run_pattern="*-standard-*")

    rand_rows = lc[lc["algo"] == "random"].sort_values("seed")
    assert list(rand_rows["seed"]) == [0, 1, 2]
    noop_rows = lc[lc["algo"] == "noop"].sort_values("seed")
    assert list(noop_rows["seed"]) == [0, 1, 2]
