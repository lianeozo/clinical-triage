"""FactorizedPPO: hierarchical actor that samples SOC first, then treatment conditioned on SOC."""
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
from phase2_sac.feasibility import FEASIBILITY_MASK

# action_idx = treat_idx * 4 + soc; treat_idx = antibiotic*4 + ventilation*2 + vasopressors
_TREAT_FEASIBLE_NP: np.ndarray = np.array(
	[[FEASIBILITY_MASK[t * 4 + s] for t in range(8)] for s in range(4)],
	dtype=bool,
)


def _apply_feasibility_mask(
	treat_logits: torch.Tensor, soc_batch: torch.Tensor, feasible_t: torch.Tensor,
) -> torch.Tensor:
	mask = feasible_t[soc_batch]
	return treat_logits.masked_fill(~mask, -1e9)


class FactorizedActorCritic(nn.Module):
	"""Shared encoder → SOC head + SOC-conditioned treatment head + separate critic."""

	def __init__(self, obs_dim: int, hidden_dim: int) -> None:
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(obs_dim, hidden_dim), nn.LeakyReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
		)
		self.soc_head = nn.Linear(hidden_dim, 4)
		self.treat_head = nn.Linear(hidden_dim + 4, 8)
		self.critic = nn.Sequential(
			nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
			nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
			nn.Linear(hidden_dim, 1),
		)
		self.register_buffer('eye4', torch.eye(4))
		self.register_buffer('treat_feasible', torch.from_numpy(_TREAT_FEASIBLE_NP))

	def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		batch = obs.shape[0]
		h = self.encoder(obs)

		soc_logits = self.soc_head(h)

		soc_oh = self.eye4.unsqueeze(0).expand(batch, 4, 4)
		h_tiled = h.unsqueeze(1).expand(batch, 4, h.shape[-1])
		treat_inp = torch.cat([h_tiled, soc_oh], dim=-1)
		treat_logits_all = self.treat_head(
			treat_inp.reshape(batch * 4, -1)
		).reshape(batch, 4, 8)

		value = self.critic(obs).squeeze(-1)
		return soc_logits, treat_logits_all, value


class FactorizedPPOAgent(Agent):
	"""PPO with factorized actor: soc ~ π_soc(s), treat ~ π_treat(s, soc)."""

	def __init__(
		self,
		obs_dim: int,
		n_actions: int,
		config: PPOAgentConfig,
		seed: int = 0,
		device: str = "cpu",
	) -> None:
		super().__init__()
		torch.manual_seed(seed)
		self.cfg = config
		self.device = torch.device(device)
		self.n_actions = n_actions
		self.model = FactorizedActorCritic(obs_dim, config.hidden_dim).to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

	def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
		with torch.no_grad():
			x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
			soc_logits, treat_logits_all, _ = self.model(x)
			if eval_mode:
				soc = soc_logits.argmax(dim=-1)
			else:
				soc = Categorical(logits=soc_logits).sample()
			treat_logits = treat_logits_all[torch.arange(soc.shape[0], device=self.device), soc]
			treat_logits = _apply_feasibility_mask(treat_logits, soc, self.model.treat_feasible)
			if eval_mode:
				treat = treat_logits.argmax(dim=-1)
			else:
				treat = Categorical(logits=treat_logits).sample()
			return int(treat[0].item() * 4 + soc[0].item())

	def act_with_logp_value(self, obs: np.ndarray) -> tuple[int, float, float]:
		with torch.no_grad():
			x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
			soc_logits, treat_logits_all, value = self.model(x)
			soc_dist = Categorical(logits=soc_logits)
			soc = soc_dist.sample()
			treat_logits = treat_logits_all[torch.arange(soc.shape[0], device=self.device), soc]
			treat_logits = _apply_feasibility_mask(treat_logits, soc, self.model.treat_feasible)
			treat_dist = Categorical(logits=treat_logits)
			treat = treat_dist.sample()
			log_prob = soc_dist.log_prob(soc) + treat_dist.log_prob(treat)
			action_idx = int(treat[0].item() * 4 + soc[0].item())
			return action_idx, float(log_prob[0].item()), float(value[0].item())

	def value_only(self, obs: np.ndarray) -> float:
		with torch.no_grad():
			x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
			_, _, value = self.model(x)
			return float(value[0].item())

	def _evaluate_actions(
		self,
		obs: torch.Tensor,
		soc: torch.Tensor,
		treat: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		batch = obs.shape[0]
		soc_logits, treat_logits_all, value = self.model(obs)

		soc_dist = Categorical(logits=soc_logits)
		treat_logits = treat_logits_all[torch.arange(batch, device=obs.device), soc]
		treat_logits = _apply_feasibility_mask(treat_logits, soc, self.model.treat_feasible)
		treat_dist = Categorical(logits=treat_logits)

		log_prob = soc_dist.log_prob(soc) + treat_dist.log_prob(treat)
		return log_prob, soc_dist.entropy(), treat_dist.entropy(), value, soc_logits

	def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
		obs = torch.from_numpy(batch["obs"]).to(self.device)
		action = torch.from_numpy(batch["action"]).long().to(self.device)
		old_logp = torch.from_numpy(batch["old_logprob"]).to(self.device)
		adv = torch.from_numpy(batch["advantage"]).to(self.device)
		ret = torch.from_numpy(batch["return_"]).to(self.device)

		soc = action % 4
		treat = action // 4

		log_prob, soc_ent, treat_ent, value, soc_logits = self._evaluate_actions(obs, soc, treat)

		ratio = torch.exp(log_prob - old_logp)
		clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon)
		pg_loss = -torch.min(ratio * adv, clipped * adv).mean()
		v_loss = F.mse_loss(value, ret)
		entropy = (soc_ent + treat_ent).mean()
		loss = pg_loss + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * entropy

		self.optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
		self.optimizer.step()

		with torch.no_grad():
			logratio = log_prob - old_logp
			approx_kl = ((ratio - 1.0) - logratio).mean().item()
			clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_epsilon).float().mean().item()
			soc_probs = soc_logits.softmax(dim=-1).mean(dim=0)

		return {
			"pg_loss": float(pg_loss.item()),
			"v_loss": float(v_loss.item()),
			"entropy": float(entropy.item()),
			"soc_entropy": float(soc_ent.mean().item()),
			"treat_entropy": float(treat_ent.mean().item()),
			"approx_kl": float(approx_kl),
			"clip_frac": float(clip_frac),
			"soc_dist_0": float(soc_probs[0].item()),
			"soc_dist_1": float(soc_probs[1].item()),
			"soc_dist_2": float(soc_probs[2].item()),
			"soc_dist_3": float(soc_probs[3].item()),
		}

	def save(self, path: Path) -> None:
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(self.model.state_dict(), path)

	def load(self, path: Path) -> None:
		self.model.load_state_dict(
			torch.load(path, map_location=self.device, weights_only=True)
		)
