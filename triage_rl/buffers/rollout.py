"""Rollout buffer with GAE for on-policy training (PPO)."""
from __future__ import annotations

from typing import Iterator

import numpy as np


def compute_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray,
                terminated: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE(lambda) advantages and returns.

    Bootstrapping rule: when terminated[t] is True, the next-value term is zeroed
    (the trajectory ended in an absorbing state). Truncated (timeout) episodes
    should NOT have terminated=True here, so the next_value bootstrap is used.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - float(terminated[t])
        delta = rewards[t] + gamma * next_values[t] * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


class RolloutBuffer:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._reset_storage()

    def _reset_storage(self) -> None:
        self.obs: np.ndarray | None = None
        self.action: np.ndarray | None = None
        self.old_logprob: np.ndarray | None = None
        self.advantage: np.ndarray | None = None
        self.return_: np.ndarray | None = None

    def fill(self, *, obs, actions, rewards_raw, terminated, log_probs,
             values, next_values, gamma: float, lam: float, reward_scale: float) -> None:
        rewards = np.array(rewards_raw, dtype=np.float32) * float(reward_scale)
        values_arr = np.array(values, dtype=np.float32)
        next_values_arr = np.array(next_values, dtype=np.float32)
        terminated_arr = np.array(terminated, dtype=np.bool_)
        adv, ret = compute_gae(rewards, values_arr, next_values_arr,
                               terminated_arr, gamma=gamma, lam=lam)
        # Advantage normalization
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.obs = np.stack(obs).astype(np.float32)
        self.action = np.array(actions, dtype=np.int64)
        self.old_logprob = np.array(log_probs, dtype=np.float32)
        self.advantage = adv.astype(np.float32)
        self.return_ = ret.astype(np.float32)

    def minibatches(self, batch_size: int) -> Iterator[dict[str, np.ndarray]]:
        assert self.obs is not None, "must call fill() before minibatches()"
        n = self.obs.shape[0]
        perm = self._rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            yield {
                "obs": self.obs[idx],
                "action": self.action[idx],
                "old_logprob": self.old_logprob[idx],
                "advantage": self.advantage[idx],
                "return_": self.return_[idx],
            }

    def clear(self) -> None:
        self._reset_storage()
