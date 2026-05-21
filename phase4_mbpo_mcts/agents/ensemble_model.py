"""PETS-style ensemble of recurrent transition models with per-variable categorical heads."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from triage_rl.config import EnsembleModelConfig


def _onehot(idx: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(idx.long(), num_classes=n).float()


def _multi_var_onehot(obs: torch.Tensor, obs_dims: Sequence[int]) -> torch.Tensor:
    # obs: [..., len(obs_dims)] int; returns [..., sum(obs_dims)] float
    parts = []
    for i, d in enumerate(obs_dims):
        parts.append(_onehot(obs[..., i], d))
    return torch.cat(parts, dim=-1)


class _Member(nn.Module):
    def __init__(self, obs_dims: Sequence[int], n_actions: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.obs_dims = tuple(obs_dims)
        self.n_actions = int(n_actions)
        in_dim = sum(obs_dims) + self.n_actions
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, d) for d in obs_dims])

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> list[torch.Tensor]:
        # obs_hist: [B, T, V] int; act_hist: [B, T] int
        oh = _multi_var_onehot(obs_hist, self.obs_dims)          # [B, T, sum_d]
        ah = _onehot(act_hist, self.n_actions)                   # [B, T, n_actions]
        x = torch.cat([oh, ah], dim=-1)                          # [B, T, in_dim]
        h_all, _ = self.gru(x)
        h_final = h_all[:, -1, :]                                # [B, hidden]
        return [head(h_final) for head in self.heads]


class EnsembleModel:
    """K-member ensemble of recurrent transition models.

    Not an ``nn.Module`` — a plain Python container of per-member modules and
    optimizers. Each member is constructed under its own ``torch.manual_seed``
    so init differs across members; each has its own ``torch.optim.Adam``.

    ``predict`` returns ``list[torch.Tensor]`` — one logits tensor per obs
    variable, shape ``[B, dim_v]`` — derived from the GRU's final hidden
    state on the supplied ``(o_t, a_t)`` history.
    """

    def __init__(self, obs_dims: Sequence[int], n_actions: int,
                 config: EnsembleModelConfig, seed: int, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.cfg = config
        self.obs_dims = tuple(obs_dims)
        self.n_actions = int(n_actions)
        self.members: list[_Member] = []
        self.optimizers: list[torch.optim.Optimizer] = []
        for k in range(config.n_members):
            torch.manual_seed(seed * 1000 + k)
            m = _Member(obs_dims, n_actions, config.hidden_dim, config.n_layers).to(self.device)
            self.members.append(m)
            self.optimizers.append(
                torch.optim.Adam(m.parameters(), lr=config.lr, weight_decay=config.weight_decay))

    def predict(self, obs_hist: torch.Tensor, act_hist: torch.Tensor, member_idx: int) -> list[torch.Tensor]:
        obs_hist = obs_hist.to(self.device).long()
        act_hist = act_hist.to(self.device).long()
        return self.members[member_idx](obs_hist, act_hist)

    def compute_loss(self, obs_hist: torch.Tensor, act_hist: torch.Tensor,
                     target_next: torch.Tensor, member_idx: int) -> torch.Tensor:
        logits_list = self.predict(obs_hist, act_hist, member_idx)
        target_next = target_next.to(self.device).long()
        total = logits_list[0].new_zeros(())
        for i, logits in enumerate(logits_list):
            total = total + F.cross_entropy(logits, target_next[:, i])
        return total

    def state_dict_all(self) -> dict:
        return {"members": [m.state_dict() for m in self.members]}

    def load_state_dict_all(self, sd: dict) -> None:
        for m, m_sd in zip(self.members, sd["members"]):
            m.load_state_dict(m_sd)
