import numpy as np
import pytest

from triage_rl.buffers.replay import ReplayBuffer


def test_replay_buffer_push_and_sample():
    buf = ReplayBuffer(capacity=100, obs_dim=8)
    for i in range(50):
        obs = np.full(8, i, dtype=np.float32)
        next_obs = np.full(8, i + 1, dtype=np.float32)
        buf.push(obs, action=i % 32, scaled_reward=float(i) * 0.1, next_obs=next_obs, terminated=(i % 10 == 0))
    assert buf.size == 50

    batch = buf.sample(batch_size=16)
    for key in ("obs", "action", "scaled_reward", "next_obs", "terminated"):
        assert key in batch
        assert len(batch[key]) == 16
    assert batch["obs"].shape == (16, 8)
    assert batch["next_obs"].shape == (16, 8)
    assert batch["action"].dtype.kind in ("i", "u")
    assert batch["terminated"].dtype == np.bool_


def test_replay_buffer_capacity_overflow_evicts_oldest():
    buf = ReplayBuffer(capacity=5, obs_dim=2)
    for i in range(10):
        buf.push(np.array([i, i], dtype=np.float32), i, float(i),
                 np.array([i+1, i+1], dtype=np.float32), False)
    assert buf.size == 5
    # The oldest items (i=0..4) should have been overwritten by i=5..9.
    obs = buf._obs[:buf.size]
    # All retained obs first-components should be >= 5
    assert obs[:, 0].min() >= 5
