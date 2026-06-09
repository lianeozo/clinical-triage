"""DQN agent for the 32-action triage POMDP.

4-hidden-layer MLP (LeakyReLU), target net with soft updates (tau),
epsilon-greedy with exponential decay parameterized by total_env_steps,
Huber loss, Adam, gradient clip.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from triage_rl.agents.base import Agent
from triage_rl.config import DQNAgentConfig


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    assert n_layers >= 2
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.LeakyReLU()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class DQNAgent(Agent):
    def __init__(self, obs_dim: int, n_actions: int, total_env_steps: int,
                 config: DQNAgentConfig, seed: int = 0, device: str = "cpu") -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.cfg = config
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.total_env_steps = int(total_env_steps)
        self.policy_net = _make_mlp(obs_dim, config.hidden_dim, n_actions, config.n_layers).to(self.device)
        self.target_net = _make_mlp(obs_dim, config.hidden_dim, n_actions, config.n_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self._rng = np.random.default_rng(seed)
        self._steps_done = 0
        # Decay schedule: eps_end reached at config.eps_decay_fraction of total budget.
        self._decay_horizon = max(1, int(self.total_env_steps * config.eps_decay_fraction))

    def _epsilon(self) -> float:
        if self._steps_done >= self._decay_horizon:
            return self.cfg.eps_end
        frac = self._steps_done / self._decay_horizon
        return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * math.exp(-3.0 * frac)

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode:
            self._steps_done += 1
            if self._rng.random() < self._epsilon():
                return int(self._rng.integers(0, self.n_actions))
        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            q = self.policy_net(x).squeeze(0)
            return int(q.argmax().item())

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.from_numpy(batch["obs"]).to(self.device)
        action = torch.from_numpy(batch["action"]).long().to(self.device).unsqueeze(1)
        reward = torch.from_numpy(batch["scaled_reward"]).to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).to(self.device)
        terminated = torch.from_numpy(batch["terminated"]).to(self.device)

        q = self.policy_net(obs).gather(1, action).squeeze(1)
        with torch.no_grad():
            if self.cfg.double_dqn:
                best_a = self.policy_net(next_obs).argmax(dim=1, keepdim=True)
                target_q_next = self.target_net(next_obs).gather(1, best_a).squeeze(1)
            else:
                target_q_next = self.target_net(next_obs).max(dim=1).values
            target = reward + (1.0 - terminated.float()) * self.cfg.gamma * target_q_next

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100.0)
        self.optimizer.step()

        # Soft target update.
        with torch.no_grad():
            # Pre-update gap: this is the lag that produced today's TD target.
            drift_sq = 0.0
            for p_pol, p_tgt in zip(self.policy_net.parameters(), self.target_net.parameters()):
                drift_sq += ((p_pol.data - p_tgt.data) ** 2).sum().item()
                p_tgt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p_pol.data)
        return {
            "loss": float(loss.item()),
            "q_mean": float(q.mean().item()),
            "q_max": float(q.max().item()),
            "target_drift_l2": float(drift_sq ** 0.5),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "steps_done": self._steps_done,
        }, path)

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self._steps_done = int(ckpt["steps_done"])
