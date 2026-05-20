"""IQLAgent: offline IQL with twin-Q + V-network + expectile regression + AWR policy extraction.

Reference: Kostrikov et al. 2021 "Offline Reinforcement Learning with Implicit Q-Learning".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from triage_rl.agents.base import Agent
from triage_rl.config import IQLAgentConfig


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """ReLU MLP with `n_layers` hidden layers."""
    assert n_layers >= 1
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class IQLAgent(Agent):
    def __init__(self, obs_dim: int, n_actions: int, config: IQLAgentConfig,
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
        self.v_net = _make_mlp(obs_dim, config.hidden_dim, 1, config.n_layers).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q_opt = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=config.critic_lr)
        self.v_opt = optim.Adam(self.v_net.parameters(), lr=config.v_lr)

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            logits = self.actor(x).squeeze(0)
            if eval_mode:
                return int(logits.argmax().item())
            return int(Categorical(logits=logits).sample().item())

    def _compute_policy_loss(self, logits: torch.Tensor, log_p: torch.Tensor,
                             awr_weight: torch.Tensor, log_p_taken: torch.Tensor
                             ) -> tuple[torch.Tensor, dict[str, float]]:
        """Base AWR loss: -E_D[ exp(beta*A(s,a)) * log pi(a|s) ]. Subclasses override to add regularizers."""
        return -(awr_weight * log_p_taken).mean(), {}

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.from_numpy(batch["obs"]).to(self.device)
        action = torch.from_numpy(batch["action"]).long().to(self.device)
        scaled_reward = torch.from_numpy(batch["scaled_reward"]).to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).to(self.device)
        terminated = torch.from_numpy(batch["terminated"]).to(self.device)

        # --- 1. V update (asymmetric expectile loss) ---
        with torch.no_grad():
            q1_t_sa = self.target_q1(obs).gather(1, action.unsqueeze(1)).squeeze(1)
            q2_t_sa = self.target_q2(obs).gather(1, action.unsqueeze(1)).squeeze(1)
            y_v = torch.min(q1_t_sa, q2_t_sa)

        v_pred = self.v_net(obs).squeeze(-1)
        u = y_v - v_pred
        weight_lam = torch.where(u > 0,
                                  torch.tensor(self.cfg.expectile_lambda, device=self.device),
                                  torch.tensor(1.0 - self.cfg.expectile_lambda, device=self.device))
        v_loss = (weight_lam * u.pow(2)).mean()

        if v_loss.requires_grad:
            self.v_opt.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(self.v_net.parameters(), self.cfg.max_grad_norm)
            self.v_opt.step()

        # --- 2. Q update (target uses V, NOT max-Q) ---
        with torch.no_grad():
            v_next = self.v_net(next_obs).squeeze(-1)
            y_q = scaled_reward + (1.0 - terminated.float()) * self.cfg.gamma * v_next

        q1_pred = self.q1(obs).gather(1, action.unsqueeze(1)).squeeze(1)
        q2_pred = self.q2(obs).gather(1, action.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q1_pred, y_q) + F.mse_loss(q2_pred, y_q)

        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.cfg.max_grad_norm)
        self.q_opt.step()

        # --- 3. Policy update (AWR) ---
        with torch.no_grad():
            q_min_sa = torch.min(self.q1(obs), self.q2(obs)).gather(
                1, action.unsqueeze(1)).squeeze(1)
            advantage = q_min_sa - self.v_net(obs).squeeze(-1)
            awr_weight = torch.exp(
                torch.clamp(self.cfg.awr_beta * advantage,
                            max=self.cfg.awr_weight_clip))

        logits = self.actor(obs)
        log_p = F.log_softmax(logits, dim=-1)
        log_p_taken = log_p.gather(1, action.unsqueeze(1)).squeeze(1)

        policy_loss, extra = self._compute_policy_loss(logits, log_p, awr_weight, log_p_taken)

        self.actor_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_opt.step()

        # --- 4. Soft target Q update ---
        with torch.no_grad():
            drift_sq = 0.0
            for tgt, src in [(self.target_q1, self.q1), (self.target_q2, self.q2)]:
                for p_t, p_s in zip(tgt.parameters(), src.parameters()):
                    drift_sq += ((p_s.data - p_t.data) ** 2).sum().item()
                    p_t.data.mul_(1.0 - self.cfg.tau).add_(p_s.data, alpha=self.cfg.tau)

        metrics = {
            "v_loss": float(v_loss.item()),
            "q_loss": float(q_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "v_mean": float(v_pred.mean().item()),
            "q1_mean": float(q1_pred.mean().item()),
            "q2_mean": float(q2_pred.mean().item()),
            "advantage_mean": float(advantage.mean().item()),
            "weight_mean": float(awr_weight.mean().item()),
            "target_drift_l2": float(drift_sq ** 0.5),
        }
        metrics.update(extra)
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
            "v_net": self.v_net.state_dict(),
        }, path)

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.target_q1.load_state_dict(ckpt["target_q1"])
        self.target_q2.load_state_dict(ckpt["target_q2"])
        self.v_net.load_state_dict(ckpt["v_net"])
