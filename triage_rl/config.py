"""Config dataclasses for env, trainers, and agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EnvConfig:
    p_diabetes: float = 0.2
    max_steps: int = 100
    reward_variant: int = 0  # 0–4; see sepsisSimDiabetes/MDP.py REWARD_VARIANTS


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
    double_dqn: bool = False


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


@dataclass
class SACAgentConfig:
    hidden_dim: int = 256
    n_layers: int = 2                        # SAC trunk depth
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    initial_alpha: float = 0.2
    target_entropy_fraction: float = 0.98   # target = -fraction * log(n_actions)
    max_grad_norm: float = 1.0


@dataclass
class SACKLFAgentConfig(SACAgentConfig):
    feasibility_beta: float = 0.5            # weight on infeasible-mass regularizer


@dataclass
class SACKLPPOAgentConfig(SACAgentConfig):
    kl_beta: float = 0.5                     # weight on KL(π_SAC || π_PPO_ref)


@dataclass
class IQLAgentConfig:
    hidden_dim: int = 256
    n_layers: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    v_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005                  # Polyak update rate for target Q (NOT the expectile coefficient)
    expectile_lambda: float = 0.8       # λ — asymmetric expectile coefficient
    awr_beta: float = 3.0               # β — advantage temperature in AWR weight
    awr_weight_clip: float = 10.0       # max exponent value to prevent overflow
    max_grad_norm: float = 1.0


@dataclass
class IQLKLFAgentConfig(IQLAgentConfig):
    feasibility_beta: float = 0.5       # same default as SAC-KL-F's β_feas


@dataclass
class OfflineConfig:
    total_grad_steps: int
    seed: int
    out_dir: Path
    dataset_path: Path
    reward_scale: float = 1e-4
    batch_size: int = 128
    eval_cadence: int = 25_000
    n_eval_episodes: int = 50
    internals_log_every: int = 1000


@dataclass
class EnsembleModelConfig:
    n_members: int = 5
    hidden_dim: int = 128
    n_layers: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class PiVNetConfig:
    hidden_dim: int = 128
    n_layers: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class MCTSConfig:
    n_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    gamma: float = 0.99
    temperature_steps: int = 15
    temperature_initial: float = 1.0
    temperature_final: float = 0.0


@dataclass
class OuterLoopConfig:
    seed_dataset_path: Path
    out_dir: Path
    n_outer_iters: int = 15
    n_episodes_per_iter: int = 30
    n_model_steps_per_iter: int = 500
    n_pv_steps_per_iter: int = 500
    batch_size: int = 256
    segment_len: int = 8
    seed: int = 0
    eval_cadence_iters: int = 1
    n_eval_episodes: int = 50
    reward_scale: float = 1e-4
    internals_log_every: int = 1  # every outer iter
