"""QACAgent: SAC-Discrete with twin-Q critics, auto-tuned entropy temperature.

Inherits from triage_rl.agents.base.Agent. Uses the existing ReplayBuffer
(obs, action, scaled_reward, next_obs, terminated). Closed-form expectations
over the 32 discrete actions — no MC sampling.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from triage_rl.agents.base import Agent
from triage_rl.config import QACAgentConfig


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """ReLU MLP with `n_layers` hidden layers."""
    assert n_layers >= 1
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class QACAgent(Agent):
    def __init__(self, obs_dim: int, n_actions: int, config: QACAgentConfig,
                 seed: int = 0, device: str = "cpu") -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.cfg = config
        self.device = torch.device(device)
        self.n_actions = n_actions

        self.actor = _make_mlp(obs_dim, config.hidden_dim, n_actions, config.n_layers).to(self.device)
        self.q1 = _make_mlp(obs_dim, config.hidden_dim, n_actions, config.n_layers).to(self.device)
        self.q2 = _make_mlp(obs_dim, config.hidden_dim, n_actions, config.n_layers).to(self.device)
        self.target_q1 = _make_mlp(obs_dim, config.hidden_dim, n_actions, config.n_layers).to(self.device)
        self.target_q2 = _make_mlp(obs_dim, config.hidden_dim, n_actions, config.n_layers).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        for p in self.target_q1.parameters():
            p.requires_grad_(False)
        for p in self.target_q2.parameters():
            p.requires_grad_(False)

        # Auto-tuned temperature α (log-parameterized for stability).
        self.log_alpha = torch.tensor(math.log(config.initial_alpha),
                                       device=self.device, requires_grad=True)
        self.alpha = math.exp(self.log_alpha.item())
        self.target_entropy = -config.target_entropy_fraction * math.log(n_actions)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q_opt = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=config.critic_lr)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=config.alpha_lr)

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            logits = self.actor(x).squeeze(0)
            if eval_mode:
                return int(logits.argmax().item())
            return int(Categorical(logits=logits).sample().item())

    def _compute_actor_loss(self, obs: torch.Tensor, probs: torch.Tensor,
                            logp: torch.Tensor, q_min: torch.Tensor
                            ) -> tuple[torch.Tensor, dict[str, float]]:
        """Base actor loss (SAC-Discrete). Subclasses override to add regularizers.

        Returns (loss_scalar, extra_metrics_dict). The extras are merged into
        the returned metrics dict by update().
        """
        sac_term = (probs * (self.alpha * logp - q_min)).sum(dim=-1).mean()
        return sac_term, {}

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.from_numpy(batch["obs"]).to(self.device)
        action = torch.from_numpy(batch["action"]).long().to(self.device)
        scaled_reward = torch.from_numpy(batch["scaled_reward"]).to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).to(self.device)
        terminated = torch.from_numpy(batch["terminated"]).to(self.device)

        # --- Critic update ---
        with torch.no_grad():
            next_logits = self.actor(next_obs)
            next_probs = F.softmax(next_logits, dim=-1)
            next_logp = F.log_softmax(next_logits, dim=-1)
            q1_t = self.target_q1(next_obs)
            q2_t = self.target_q2(next_obs)
            q_min_next = torch.min(q1_t, q2_t)
            soft_v_next = (next_probs * (q_min_next - self.alpha * next_logp)).sum(dim=-1)
            y = scaled_reward + (1.0 - terminated.float()) * self.cfg.gamma * soft_v_next

        q1_sa = self.q1(obs).gather(1, action.unsqueeze(1)).squeeze(1)
        q2_sa = self.q2(obs).gather(1, action.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q1_sa, y) + F.mse_loss(q2_sa, y)

        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.cfg.max_grad_norm)
        self.q_opt.step()

        # --- Actor update ---
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1)
        logp = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            q_min = torch.min(self.q1(obs), self.q2(obs))

        actor_loss, extra_metrics = self._compute_actor_loss(obs, probs, logp, q_min)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_opt.step()

        # --- Temperature update ---
        with torch.no_grad():
            entropy = -(probs * logp).sum(dim=-1).mean()
        alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()
        # Note: when entropy is BELOW target, (entropy + target_entropy) is negative (target_entropy < 0
        # by construction); -log_α*neg = positive, gradient descends → log_alpha increases →
        # α increases → more exploration. Sign convention is standard SAC.
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # --- Soft target update ---
        with torch.no_grad():
            drift_sq = 0.0
            for tgt, src in [(self.target_q1, self.q1), (self.target_q2, self.q2)]:
                for p_tgt, p_src in zip(tgt.parameters(), src.parameters()):
                    drift_sq += ((p_src.data - p_tgt.data) ** 2).sum().item()
                    p_tgt.data.mul_(1.0 - self.cfg.tau).add_(p_src.data, alpha=self.cfg.tau)

        metrics = {
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha),
            "entropy": float(entropy.item()),
            "q1_mean": float(q1_sa.mean().item()),
            "q2_mean": float(q2_sa.mean().item()),
            "target_drift_l2": float(drift_sq ** 0.5),
        }
        metrics.update(extra_metrics)
        return metrics

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }, path)

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.target_q1.load_state_dict(ckpt["target_q1"])
        self.target_q2.load_state_dict(ckpt["target_q2"])
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha = self.log_alpha.exp().item()
        # Rebuild alpha_opt because log_alpha is a new tensor.
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)
