"""End-to-end smoke test for the Phase 4 OuterLoopTrainer.

This is intentionally minimal — a single outer iteration on a tiny config —
because the trainer is an orchestrator over five already-tested components
(EnsembleModel, PiVNet, run_mcts, MBPOAgent, TrajectoryBuffer + Env).

What we assert:

* The trainer's ``run()`` returns without error on a 1-iter config.
* The buffer grows by the number of transitions actually collected (>= 1).
* ``eval_checkpoints.jsonl`` has at least 2 records (i=0 baseline + i=1 post-iter).
* ``training_internals.jsonl`` has at least 1 record (post-iter internals).
* Each eval record carries an ``outer_iter`` field so the aggregator can
  build a Phase 4 learning curve with outer-iter on the x-axis.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from phase4_mbpo_mcts.trainers.outer_loop import OuterLoopTrainer
from triage_rl.config import (
    EnsembleModelConfig,
    MCTSConfig,
    OuterLoopConfig,
    PiVNetConfig,
)


_REAL_DATASET_PATH = "results/phase1_ppo_dqn/aggregated/offline_dataset.npz"


def _smoke_cfg(tmp_path: Path) -> OuterLoopConfig:
    """Tiny config that finishes in well under 5 minutes on CPU."""
    return OuterLoopConfig(
        seed_dataset_path=Path(_REAL_DATASET_PATH),
        out_dir=tmp_path / "run",
        n_outer_iters=1,
        n_episodes_per_iter=2,
        n_model_steps_per_iter=4,
        n_pv_steps_per_iter=4,
        batch_size=8,
        segment_len=3,
        seed=0,
        n_eval_episodes=2,
    )


def test_one_outer_iter_grows_buffer_and_writes_logs(tmp_path):
    if not os.path.exists(_REAL_DATASET_PATH):
        pytest.skip(f"{_REAL_DATASET_PATH} not present")

    cfg = _smoke_cfg(tmp_path)
    model_cfg = EnsembleModelConfig(n_members=2, hidden_dim=16)
    pi_v_cfg = PiVNetConfig(hidden_dim=16)
    mcts_cfg = MCTSConfig(n_simulations=5)

    trainer = OuterLoopTrainer(
        outer_cfg=cfg,
        model_cfg=model_cfg,
        pi_v_cfg=pi_v_cfg,
        mcts_cfg=mcts_cfg,
        no_mcts=False,
        algo_name="mbpo_mcts_main",
        device="cpu",
    )
    n_seed_transitions = trainer.run()

    # The trainer should have grown the buffer beyond the seed-load size.
    assert trainer.buffer.n_transitions > n_seed_transitions, (
        f"buffer did not grow: started at {n_seed_transitions}, "
        f"ended at {trainer.buffer.n_transitions}"
    )

    # Eval checkpoints: 1 baseline (i=0) + 1 post-iter (i=1) = 2 rows.
    ckpt_path = cfg.out_dir / "eval_checkpoints.jsonl"
    assert ckpt_path.exists(), f"missing {ckpt_path}"
    rows = [json.loads(line) for line in ckpt_path.read_text().splitlines() if line.strip()]
    assert len(rows) >= 2, f"expected >= 2 eval rows, got {len(rows)}"

    # outer_iter must appear in each row so the aggregator can group by it.
    assert all("outer_iter" in r for r in rows), \
        f"missing outer_iter in some rows: {[r.keys() for r in rows]}"
    outer_iters = sorted({int(r["outer_iter"]) for r in rows})
    assert 0 in outer_iters
    assert 1 in outer_iters

    # Internals jsonl: at least one row after outer iter 1.
    internals_path = cfg.out_dir / "training_internals.jsonl"
    assert internals_path.exists(), f"missing {internals_path}"
    int_rows = [json.loads(line) for line in internals_path.read_text().splitlines()
                if line.strip()]
    assert len(int_rows) >= 1
    # Expected diagnostic keys per spec §6 internals bullet.
    expected_keys = {"model_nll_heldout", "mcts_mean_depth", "policy_entropy",
                     "value_mean"}
    assert expected_keys.issubset(set(int_rows[-1].keys())), \
        f"internals missing keys: have {set(int_rows[-1].keys())}, want {expected_keys}"
