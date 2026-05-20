"""SACKLPPOAgent: SACAgent + KL(π_SAC || π_PPO_ref) penalty in actor loss.

Loads a Phase-1 PPO actor as a frozen reference at construction time
(seed-paired). The reference is identified by --ppo-run-dir at the CLI level
and a seed index. Closed-form KL over 32 discrete actions, added to the
SAC-Discrete actor loss with weight kl_beta.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from phase2_sac.agents.sac import SACAgent
from phase2_sac.ppo_reference import load_ppo_actor
from triage_rl.config import PPOAgentConfig, SACKLPPOAgentConfig


class SACKLPPOAgent(SACAgent):
    def __init__(self, obs_dim: int, n_actions: int, config: SACKLPPOAgentConfig,
                 seed: int = 0, device: str = "cpu",
                 ppo_run_dir: Path | None = None) -> None:
        if ppo_run_dir is None:
            raise ValueError("SACKLPPOAgent requires ppo_run_dir at construction time")
        super().__init__(obs_dim, n_actions, config, seed, device)
        self._ppo_ref = load_ppo_actor(
            ppo_run_dir=Path(ppo_run_dir), seed=seed,
            obs_dim=obs_dim, n_actions=n_actions,
            ppo_cfg=PPOAgentConfig(), device=device,
        )
        self.kappa = config.kl_beta

    def _compute_actor_loss(self, obs, probs, logp, q_min):
        sac_term = (probs * (self.alpha * logp - q_min)).sum(dim=-1).mean()
        with torch.no_grad():
            ref_logits, _ = self._ppo_ref(obs)
            ref_logp = F.log_softmax(ref_logits, dim=-1)
        # Forward KL closed-form over 32 actions: sum_a π(a) * (log π(a) - log π_ref(a)).
        kl = (probs * (logp - ref_logp)).sum(dim=-1).mean()
        actor_loss = sac_term + self.kappa * kl
        return actor_loss, {"kl_to_ppo": float(kl.item())}
