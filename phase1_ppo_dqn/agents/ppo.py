"""PPO agent: actor-critic with clipped surrogate, GAE handled by RolloutBuffer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from triage_rl.agents.base import Agent
from triage_rl.config import PPOAgentConfig


class _ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class PPOAgent(Agent):
    def __init__(self, obs_dim: int, n_actions: int, config: PPOAgentConfig,
                 seed: int = 0, device: str = "cpu") -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.cfg = config
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.model = _ActorCritic(obs_dim, config.hidden_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            logits, _ = self.model(x)
            if eval_mode:
                return int(logits.argmax(dim=-1).item())
            dist = Categorical(logits=logits)
            return int(dist.sample().item())

    def get_logp_value(self, obs: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return per-row log-prob of `actions` under current policy and per-row value."""
        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32)).to(self.device)
            a = torch.from_numpy(actions.astype(np.int64)).to(self.device)
            logits, value = self.model(x)
            logp = Categorical(logits=logits).log_prob(a)
            return logp.cpu().numpy(), value.cpu().numpy()

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.from_numpy(batch["obs"]).to(self.device)
        action = torch.from_numpy(batch["action"]).long().to(self.device)
        old_logp = torch.from_numpy(batch["old_logprob"]).to(self.device)
        adv = torch.from_numpy(batch["advantage"]).to(self.device)
        ret = torch.from_numpy(batch["return_"]).to(self.device)

        logits, value = self.model(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(action)
        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon)
        pg_loss = -torch.min(ratio * adv, clipped * adv).mean()
        v_loss = F.mse_loss(value, ret)
        entropy = dist.entropy().mean()
        loss = pg_loss + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            # Schulman positive estimator: (ratio - 1) - log(ratio)
            logratio = logp - old_logp
            approx_kl = ((ratio - 1.0) - logratio).mean().item()
            clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_epsilon).float().mean().item()
        return {
            "pg_loss": float(pg_loss.item()),
            "v_loss": float(v_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(approx_kl),
            "clip_frac": float(clip_frac),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
