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

from phase4_mbpo_mcts.trainers.outer_loop import (
    OuterLoopTrainer,
    _env_full_reward_fn,
)
from sepsisSimDiabetes.Action import Action
from triage_rl.config import (
    EnsembleModelConfig,
    MCTSConfig,
    OuterLoopConfig,
    PiVNetConfig,
)


# ----------------------------------------------------------------------------
# env_full_reward_fn — exact correctness against MDP.calculateReward semantics.
#
# Obs vector layout (must match sepsisSimDiabetes/State.py):
#   [hr, sysbp, percoxyg, glucose, antibiotic, vaso, vent, soc]
# Normal bins: hr=1, sysbp=1, percoxyg=1, glucose=2. NUM_SOC=4.
# ----------------------------------------------------------------------------

_NORMAL_OBS = np.array([1, 1, 1, 2, 0, 0, 0, 0], dtype=np.int64)


def _action_idx(antibiotic: int, ventilation: int, vasopressors: int, soc: int) -> int:
    """Mirror ``Action.get_action_idx`` without going through the dict ctor."""
    a = Action(selected_actions={})  # all-zero treatments, soc=0
    a.antibiotic = antibiotic
    a.ventilation = ventilation
    a.vasopressors = vasopressors
    a.soc = soc
    return a.get_action_idx()


def _noop_action() -> int:
    """An action that keeps soc unchanged and turns off all treatments."""
    return _action_idx(0, 0, 0, 0)


def test_full_reward_discharge_terminal():
    """next_obs with num_abnormal==0 and no active treatments => +10000."""
    prev = _NORMAL_OBS.copy()
    prev[0] = 0  # hr abnormal so prev is not itself terminal
    nxt = _NORMAL_OBS.copy()
    r = _env_full_reward_fn(prev, _noop_action(), nxt)
    assert r == 10_000.0, f"expected +10000 discharge reward, got {r}"


def test_full_reward_death_terminal():
    """next_obs with num_abnormal>=3 => -10000."""
    prev = _NORMAL_OBS.copy()
    nxt = _NORMAL_OBS.copy()
    nxt[0] = 0  # hr abnormal
    nxt[1] = 0  # sysbp abnormal
    nxt[2] = 0  # percoxyg abnormal => 3 abnormal => death
    r = _env_full_reward_fn(prev, _noop_action(), nxt)
    assert r == -10_000.0, f"expected -10000 death reward, got {r}"


def test_full_reward_vitals_improvement_and_treatment_cost():
    """Improving from 2 abnormal to 1 with antibiotic on:
       change_in_abnormal=+1 -> +100, antibiotic -> -10. Total = +90.
    """
    prev = _NORMAL_OBS.copy()
    prev[0] = 0  # hr abnormal
    prev[1] = 0  # sysbp abnormal => 2 abnormal
    nxt = _NORMAL_OBS.copy()
    nxt[0] = 0  # 1 abnormal in next
    nxt[4] = 1  # antibiotic on in next obs (treatment applied)
    # action: antibiotic on, soc unchanged
    a = _action_idx(antibiotic=1, ventilation=0, vasopressors=0, soc=0)
    r = _env_full_reward_fn(prev, a, nxt)
    assert r == 90.0, f"expected +90 (vitals +100, antibiotic -10), got {r}"


def test_full_reward_soc_change_cost():
    """Soc change with no other changes: -50 flat cost.
       prev abn=1, next abn=1 => no vitals reward.
       next_abn != 0 so no escalation-with-zero-abnormal penalty.
       next_abn < 2 so no de-escalation severity penalty.
    """
    prev = _NORMAL_OBS.copy()
    prev[0] = 0  # 1 abnormal
    nxt = prev.copy()
    nxt[7] = 1  # soc changed 0 -> 1
    a = _action_idx(antibiotic=0, ventilation=0, vasopressors=0, soc=1)
    r = _env_full_reward_fn(prev, a, nxt)
    assert r == -50.0, f"expected -50 for soc change alone, got {r}"


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
