"""Ring-buffer replay for off-policy training."""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int | None = None) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros(capacity, dtype=np.int64)
        self._scaled_reward = np.zeros(capacity, dtype=np.float32)
        self._terminated = np.zeros(capacity, dtype=np.bool_)
        self._write_idx = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)

    def push(self, obs: np.ndarray, action: int, scaled_reward: float,
             next_obs: np.ndarray, terminated: bool) -> None:
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"obs shape {obs.shape} != ({self.obs_dim},)")
        if next_obs.shape != (self.obs_dim,):
            raise ValueError(f"next_obs shape {next_obs.shape} != ({self.obs_dim},)")
        i = self._write_idx
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._action[i] = int(action)
        self._scaled_reward[i] = float(scaled_reward)
        self._terminated[i] = bool(terminated)
        self._write_idx = (self._write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise RuntimeError("buffer is empty")
        idx = self._rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self._obs[idx],
            "action": self._action[idx],
            "scaled_reward": self._scaled_reward[idx],
            "next_obs": self._next_obs[idx],
            "terminated": self._terminated[idx],
        }
