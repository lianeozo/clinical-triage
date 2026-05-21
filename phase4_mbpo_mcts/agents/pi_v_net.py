"""Policy / value network with a separate recurrent encoder.

Per Phase 4 spec §3b: the AlphaZero policy and value heads sit on top of their
OWN GRU encoder, distinct from the dynamics ensemble. The policy head emits raw
logits over actions (no in-module softmax); the value head emits a scalar
per sample. A single Adam optimizer covers the encoder and both heads.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from triage_rl.config import PiVNetConfig


def _onehot(idx: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(idx.long(), num_classes=n).float()


def _multi_var_onehot(obs: torch.Tensor, obs_dims: Sequence[int]) -> torch.Tensor:
    # obs: [..., len(obs_dims)] int; returns [..., sum(obs_dims)] float
    parts = []
    for i, d in enumerate(obs_dims):
        parts.append(_onehot(obs[..., i], d))
    return torch.cat(parts, dim=-1)


class PiVNet(nn.Module):
    """Separate-encoder policy + value network.

    Architecture (matches spec §3b):
        encoder: GRU(input_dim = sum(obs_dims) + n_actions, hidden_dim, n_layers)
        pi_head: Linear(hidden_dim, n_actions)  -> raw logits
        v_head:  Linear(hidden_dim, 1)          -> scalar, squeezed to [B]

    Unlike ``EnsembleModel`` (a plain container of K independent members),
    ``PiVNet`` IS an ``nn.Module`` — a single network with one optimizer over
    all params.
    """

    def __init__(
        self,
        obs_dims: Sequence[int],
        n_actions: int,
        config: PiVNetConfig,
        seed: int,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.cfg = config
        self.obs_dims = tuple(obs_dims)
        self.n_actions = int(n_actions)
        in_dim = sum(self.obs_dims) + self.n_actions

        # Seeded init so the seed kwarg reproducibly determines weights
        # (mirrors how EnsembleModel seeds each member at construction).
        torch.manual_seed(seed)
        self.encoder = nn.GRU(
            in_dim, config.hidden_dim, num_layers=config.n_layers, batch_first=True
        )
        self.pi_head = nn.Linear(config.hidden_dim, self.n_actions)
        self.v_head = nn.Linear(config.hidden_dim, 1)

        self.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    def forward(
        self, obs_hist: torch.Tensor, act_hist: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode (obs_hist, act_hist) and return (logits, value).

        Args:
            obs_hist: [B, T, V] int — per-variable categorical indices.
            act_hist: [B, T] int — action indices.

        Returns:
            logits: [B, n_actions] — RAW policy logits (no softmax applied).
            value:  [B] — scalar value per sample (v_head Linear(.,1) squeezed).
        """
        obs_hist = obs_hist.to(self.device).long()
        act_hist = act_hist.to(self.device).long()
        oh = _multi_var_onehot(obs_hist, self.obs_dims)   # [B, T, sum_d]
        ah = _onehot(act_hist, self.n_actions)            # [B, T, n_actions]
        x = torch.cat([oh, ah], dim=-1)                   # [B, T, in_dim]
        h_all, _ = self.encoder(x)
        h_final = h_all[:, -1, :]                         # [B, hidden]
        logits = self.pi_head(h_final)                    # [B, n_actions]
        value = self.v_head(h_final).squeeze(-1)          # [B]
        return logits, value
