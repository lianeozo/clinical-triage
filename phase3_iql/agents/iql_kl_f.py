"""IQLKLFAgent: IQLAgent + infeasible-mass penalty in policy extraction loss.

The AWR policy loss is augmented with the same feasibility regularizer as SAC-KL-F:
  L_pi = -E_D[ exp(beta*A) * log pi(a|s) ] + beta_feas * E_s[ sum_a pi(a|s) * (1 - feasibility_mask[a]) ]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from phase2_sac.feasibility import FEASIBILITY_MASK
from phase3_iql.agents.iql import IQLAgent
from triage_rl.config import IQLKLFAgentConfig


class IQLKLFAgent(IQLAgent):
    def __init__(self, obs_dim: int, n_actions: int, config: IQLKLFAgentConfig,
                 seed: int = 0, device: str = "cpu") -> None:
        super().__init__(obs_dim, n_actions, config, seed, device)
        self._feasibility = torch.from_numpy(
            FEASIBILITY_MASK.astype(np.float32)).to(self.device)
        self.beta_feas = config.feasibility_beta

    def _compute_policy_loss(self, logits, log_p, awr_weight, log_p_taken):
        awr_term = -(awr_weight * log_p_taken).mean()
        probs = F.softmax(logits, dim=-1)
        infeasible_mass = (probs * (1.0 - self._feasibility)).sum(dim=-1).mean()
        loss = awr_term + self.beta_feas * infeasible_mass
        return loss, {"infeasible_mass": float(infeasible_mass.item())}
