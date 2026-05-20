"""SACKLFAgent: SACAgent + infeasible-mass penalty in actor loss.

The actor loss adds `β * E_π[1 - feasibility_mask]` — a linear penalty on
policy mass placed on infeasible actions. Returns an extra `infeasible_mass`
metric for logging.
"""
from __future__ import annotations

import numpy as np
import torch

from phase2_sac.agents.sac import SACAgent
from phase2_sac.feasibility import FEASIBILITY_MASK
from triage_rl.config import SACKLFAgentConfig


class SACKLFAgent(SACAgent):
    def __init__(self, obs_dim: int, n_actions: int, config: SACKLFAgentConfig,
                 seed: int = 0, device: str = "cpu") -> None:
        super().__init__(obs_dim, n_actions, config, seed, device)
        self._feasibility = torch.from_numpy(
            FEASIBILITY_MASK.astype(np.float32)).to(self.device)
        self.beta = config.feasibility_beta

    def _compute_actor_loss(self, obs, probs, logp, q_min):
        sac_term = (probs * (self.alpha * logp - q_min)).sum(dim=-1).mean()
        # Mass placed on infeasible actions (1 - mask broadcasts over 32 actions).
        infeasible_mass = (probs * (1.0 - self._feasibility)).sum(dim=-1).mean()
        actor_loss = sac_term + self.beta * infeasible_mass
        return actor_loss, {"infeasible_mass": float(infeasible_mass.item())}
