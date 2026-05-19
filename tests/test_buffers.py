import numpy as np
import pytest

from triage_rl.buffers.replay import ReplayBuffer
from triage_rl.buffers.rollout import compute_gae, RolloutBuffer


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


def test_compute_gae_toy_three_step():
    """Hand-verifiable GAE on a 3-step rollout. gamma=1.0, lam=1.0 → advantage = sum-of-future-rewards - value."""
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    next_values = np.array([0.5, 0.5, 0.0], dtype=np.float32)  # last next_value = 0 (terminal)
    terminated = np.array([False, False, True], dtype=np.bool_)

    adv, returns = compute_gae(rewards, values, next_values, terminated, gamma=1.0, lam=1.0)

    # delta[2] = 1 + 1*0*(1-1) - 0.5 = 0.5; gae[2] = 0.5
    # delta[1] = 1 + 1*0.5*(1-0) - 0.5 = 1.0; gae[1] = 1.0 + 1*1*0.5 = 1.5
    # delta[0] = 1 + 1*0.5*(1-0) - 0.5 = 1.0; gae[0] = 1.0 + 1*1*1.5 = 2.5
    assert np.allclose(adv, [2.5, 1.5, 0.5], atol=1e-5)
    assert np.allclose(returns, adv + values, atol=1e-5)


def test_rollout_buffer_fill_and_minibatches():
    obs = [np.zeros(4, dtype=np.float32) for _ in range(5)]
    next_obs = [np.ones(4, dtype=np.float32) for _ in range(5)]
    actions = [0, 1, 2, 0, 1]
    rewards_raw = [10.0, 0.0, -5.0, 0.0, 100.0]
    terminated = [False, False, False, False, True]
    log_probs = [-0.5] * 5
    values = [0.0] * 5
    next_values = [0.0] * 5

    buf = RolloutBuffer()
    buf.fill(obs=obs, actions=actions, rewards_raw=rewards_raw,
             terminated=terminated, log_probs=log_probs, values=values,
             next_values=next_values, gamma=0.99, lam=0.95, reward_scale=1e-4)

    mbs = list(buf.minibatches(batch_size=2))
    assert sum(mb["obs"].shape[0] for mb in mbs) == 5
    for mb in mbs:
        for k in ("obs", "action", "old_logprob", "advantage", "return_"):
            assert k in mb
