"""Config dataclasses for env, trainers, and agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EnvConfig:
    p_diabetes: float = 0.2
    max_steps: int = 100


@dataclass
class OffPolicyConfig:
    total_env_steps: int
    seed: int
    out_dir: Path
    reward_scale: float = 1e-4
    warmup_steps: int = 1000
    batch_size: int = 128
    eval_cadence: int = 25_000
    n_eval_episodes: int = 50
    internals_log_every: int = 1000


@dataclass
class OnPolicyConfig:
    total_env_steps: int
    seed: int
    out_dir: Path
    reward_scale: float = 1e-4
    rollout_episodes: int = 64
    max_steps_per_episode: int = 100
    update_epochs: int = 4
    minibatch_size: int = 128
    eval_cadence: int = 25_000
    n_eval_episodes: int = 50


@dataclass
class DQNAgentConfig:
    hidden_dim: int = 256
    n_layers: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    eps_start: float = 0.9
    eps_end: float = 0.01
    eps_decay_fraction: float = 0.8


@dataclass
class PPOAgentConfig:
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
