"""Seed-100K assembly + TrajectoryBuffer tests (Phase 4 T6)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from phase4_mbpo_mcts.dataset import TrajectoryBuffer, assemble_seed_100k


# ---- assemble_seed_100k --------------------------------------------------


_REAL_DATASET_PATH = "results/phase1_ppo_dqn/aggregated/offline_dataset.npz"


def test_assemble_seed_100k_sizes_on_real_dataset():
    """If the Phase 3 offline_dataset.npz is present, assert the 25K+75K mix."""
    if not os.path.exists(_REAL_DATASET_PATH):
        pytest.skip(f"{_REAL_DATASET_PATH} not present")
    data = assemble_seed_100k(src_path=_REAL_DATASET_PATH, seed=0)
    assert set(data.keys()) >= {"obs", "action", "reward", "next_obs",
                                "terminated", "source_algo"}
    assert len(data["obs"]) == 100_000
    sa = np.asarray(data["source_algo"])
    assert int((sa == "random").sum()) == 25_000
    assert int((sa == "sac_kl_f").sum()) == 75_000
    # All per-key arrays have a consistent leading dim.
    for k in ("action", "reward", "next_obs", "terminated", "source_algo"):
        assert len(data[k]) == 100_000


def test_assemble_seed_100k_synthetic(tmp_path):
    """Synthetic .npz with known per-source counts: exercise the slicer
    without depending on the real (large) dataset."""
    rng = np.random.default_rng(0)
    n_random = 25_000
    n_sac_kl_f = 200_000  # > 75K so subsampling kicks in
    n_other = 50_000
    n = n_random + n_sac_kl_f + n_other

    obs = rng.integers(0, 3, (n, 8)).astype(np.float32)
    action = rng.integers(0, 32, n).astype(np.int64)
    reward = rng.standard_normal(n).astype(np.float32)
    next_obs = rng.integers(0, 3, (n, 8)).astype(np.float32)
    terminated = (rng.random(n) < 0.05)
    source_algo = np.concatenate([
        np.full(n_random, "random", dtype="<U20"),
        np.full(n_sac_kl_f, "sac_kl_f", dtype="<U20"),
        np.full(n_other, "other", dtype="<U20"),
    ])

    src = tmp_path / "fake_offline.npz"
    np.savez(
        src,
        obs=obs, action=action, reward=reward, next_obs=next_obs,
        terminated=terminated, source_algo=source_algo,
    )

    data = assemble_seed_100k(src_path=str(src), seed=0)
    assert len(data["obs"]) == 100_000
    sa = np.asarray(data["source_algo"])
    assert int((sa == "random").sum()) == 25_000
    assert int((sa == "sac_kl_f").sum()) == 75_000
    assert int((sa == "other").sum()) == 0  # 'other' source must be excluded

    # Determinism: same seed -> same indices selected (compare a stable
    # projection of the assembled tensor).
    data2 = assemble_seed_100k(src_path=str(src), seed=0)
    np.testing.assert_array_equal(data["action"], data2["action"])
    np.testing.assert_array_equal(data["source_algo"], data2["source_algo"])

    # Different seed -> different sample (with overwhelming probability).
    data3 = assemble_seed_100k(src_path=str(src), seed=1)
    assert not np.array_equal(data["action"], data3["action"])


def test_assemble_seed_100k_smaller_random(tmp_path):
    """If fewer than 25K Random transitions exist, take all of them."""
    rng = np.random.default_rng(0)
    n_random = 1_000
    n_sac_kl_f = 200_000
    n = n_random + n_sac_kl_f
    obs = rng.integers(0, 3, (n, 8)).astype(np.float32)
    action = rng.integers(0, 32, n).astype(np.int64)
    reward = rng.standard_normal(n).astype(np.float32)
    next_obs = rng.integers(0, 3, (n, 8)).astype(np.float32)
    terminated = (rng.random(n) < 0.05)
    source_algo = np.concatenate([
        np.full(n_random, "random", dtype="<U20"),
        np.full(n_sac_kl_f, "sac_kl_f", dtype="<U20"),
    ])
    src = tmp_path / "small_random.npz"
    np.savez(src, obs=obs, action=action, reward=reward,
             next_obs=next_obs, terminated=terminated, source_algo=source_algo)
    data = assemble_seed_100k(src_path=str(src), seed=0)
    sa = np.asarray(data["source_algo"])
    # All 1K random taken; SAC-KL-F still subsampled to 75K.
    assert int((sa == "random").sum()) == n_random
    assert int((sa == "sac_kl_f").sum()) == 75_000
    assert len(data["obs"]) == n_random + 75_000


# ---- TrajectoryBuffer ----------------------------------------------------


def _make_episode(rng, length, action, terminal_reward):
    """Build a length-``length`` episode whose last step has ``terminated=True``."""
    out = []
    for t in range(length):
        obs = rng.integers(0, 2, 8).astype(np.int64)
        nxt = rng.integers(0, 2, 8).astype(np.int64)
        if t == length - 1:
            out.append((obs, int(action), float(terminal_reward), nxt, True))
        else:
            out.append((obs, int(action), 0.0, nxt, False))
    return out


def test_buffer_append_and_sample_segments():
    buf = TrajectoryBuffer(obs_dim=8, n_actions=32)
    rng = np.random.default_rng(0)
    ep1 = _make_episode(rng, length=5, action=5, terminal_reward=1.0)
    ep2 = _make_episode(rng, length=4, action=1, terminal_reward=-1.0)
    buf.append_episode(ep1)
    buf.append_episode(ep2)

    assert buf.n_episodes == 2
    assert buf.n_transitions == 9

    seg = buf.sample_segments(n=4, segment_len=3, rng=rng)
    assert seg["obs_hist"].shape == (4, 3, 8)
    assert seg["act_hist"].shape == (4, 3)
    assert seg["next_obs"].shape == (4, 8)
    assert seg["obs_hist"].dtype == np.int64
    assert seg["act_hist"].dtype == np.int64
    assert seg["next_obs"].dtype == np.int64


def test_buffer_sample_steps_shapes():
    buf = TrajectoryBuffer(obs_dim=8, n_actions=32)
    rng = np.random.default_rng(1)
    buf.append_episode(_make_episode(rng, length=6, action=2, terminal_reward=0.5))
    buf.append_episode(_make_episode(rng, length=4, action=3, terminal_reward=-0.25))

    out = buf.sample_steps(n=5, rng=rng)
    assert out["obs"].shape == (5, 8)
    assert out["action"].shape == (5,)
    assert out["reward"].shape == (5,)
    assert out["next_obs"].shape == (5, 8)
    assert out["done"].shape == (5,)
    assert out["action"].dtype == np.int64
    assert out["done"].dtype == np.bool_


def test_buffer_sample_segments_only_uses_valid_starts():
    """All sampled segments must come from episodes long enough to fit
    ``segment_len`` (i.e., len(ep) >= segment_len + 1, since ``next_obs`` is
    the obs at position start + segment_len)."""
    buf = TrajectoryBuffer(obs_dim=8, n_actions=32)
    rng = np.random.default_rng(2)
    # ep_short has length 2 — too short for segment_len=4 (need >= 5).
    buf.append_episode(_make_episode(rng, length=2, action=0, terminal_reward=0.0))
    # ep_long has length 10 — plenty of valid starts.
    buf.append_episode(_make_episode(rng, length=10, action=7, terminal_reward=1.0))

    seg = buf.sample_segments(n=50, segment_len=4, rng=rng)
    # Every action in every sampled segment must be 7 — the long episode's
    # constant action — because the short episode can't yield any valid
    # segment of length 4.
    assert np.all(seg["act_hist"] == 7), seg["act_hist"]


def test_buffer_bulk_load_from_dict_via_terminated():
    """``bulk_load_from_dict`` should infer episode boundaries from
    ``terminated`` when ``episode_boundaries`` is None."""
    buf = TrajectoryBuffer(obs_dim=4, n_actions=8)
    rng = np.random.default_rng(3)
    # 3 episodes of lengths 3, 2, 4 -> 9 transitions total.
    lengths = [3, 2, 4]
    n = sum(lengths)
    obs = rng.integers(0, 2, (n, 4)).astype(np.int64)
    action = rng.integers(0, 8, n).astype(np.int64)
    reward = rng.standard_normal(n).astype(np.float32)
    next_obs = rng.integers(0, 2, (n, 4)).astype(np.int64)
    terminated = np.zeros(n, dtype=bool)
    cursor = 0
    for L in lengths:
        terminated[cursor + L - 1] = True
        cursor += L

    data = {
        "obs": obs, "action": action, "reward": reward,
        "next_obs": next_obs, "terminated": terminated,
    }
    buf.bulk_load_from_dict(data)
    assert buf.n_episodes == 3
    assert buf.n_transitions == 9

    # Explicit boundaries should also work.
    buf2 = TrajectoryBuffer(obs_dim=4, n_actions=8)
    buf2.bulk_load_from_dict(data, episode_boundaries=[(0, 3), (3, 5), (5, 9)])
    assert buf2.n_episodes == 3
    assert buf2.n_transitions == 9
