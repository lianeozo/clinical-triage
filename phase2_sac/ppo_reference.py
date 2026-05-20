"""Load a Phase-1 PPO actor as a frozen reference policy.

Given a Phase-1 PPO run directory and a seed index, find that seed's final
checkpoint .pt file, load the actor portion, freeze all parameters, and return
a module that maps obs -> (logits, value).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from phase1_ppo_dqn.agents.ppo import _ActorCritic
from triage_rl.config import PPOAgentConfig


def find_final_checkpoint(seed_dir: Path) -> Path:
    """Return the seed's latest .pt checkpoint.

    Prefers step_<K>_final.pt (the I2-fix forced-final checkpoint) when present;
    otherwise the highest-step checkpoint.
    """
    ckpts = list((seed_dir / "checkpoints").glob("step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints under {seed_dir}/checkpoints/")
    final = [c for c in ckpts if c.stem.endswith("_final")]
    if final:
        return final[0]
    return max(ckpts, key=lambda p: int(p.stem.split("_")[1]))


def load_ppo_actor(ppo_run_dir: Path, seed: int, obs_dim: int, n_actions: int,
                   ppo_cfg: PPOAgentConfig | None = None,
                   device: str = "cpu") -> nn.Module:
    """Load and freeze the PPO actor-critic for the given seed.

    Returns a frozen _ActorCritic module. Caller uses the logits output for KL.
    All parameters have requires_grad=False; module is in eval() mode.
    """
    if ppo_cfg is None:
        ppo_cfg = PPOAgentConfig()
    seed_dir = Path(ppo_run_dir) / f"seed_{seed}"
    ckpt_path = find_final_checkpoint(seed_dir)
    model = _ActorCritic(obs_dim, ppo_cfg.hidden_dim, n_actions).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
