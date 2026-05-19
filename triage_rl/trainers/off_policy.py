"""Off-policy training loop (DQN, future QAC/IQL)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from triage_rl.agents.base import Agent
from triage_rl.buffers.replay import ReplayBuffer
from triage_rl.config import OffPolicyConfig
from triage_rl.env import Env
from triage_rl.evaluator import Evaluator
from triage_rl.logger import Logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_eval_pool(seed: int, n: int) -> list[int]:
    rng = np.random.default_rng(seed)
    # Use a large range so pool entries don't collide with reset seeds used during training.
    return [int(x) for x in rng.integers(10_000_000, 2**31 - 1, size=n)]


class OffPolicyTrainer:
    def __init__(self, env: Env, agent: Agent, buffer: ReplayBuffer,
                 evaluator: Evaluator, logger: Logger, config: OffPolicyConfig,
                 algo_name: str) -> None:
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.evaluator = evaluator
        self.logger = logger
        self.cfg = config
        self.algo_name = algo_name

    def run(self) -> None:
        seed_everything(self.cfg.seed)
        eval_pool = make_eval_pool(self.cfg.seed, n=self.cfg.n_eval_episodes)
        (self.cfg.out_dir / "eval_pool.json").write_text(json.dumps(eval_pool))
        # Patch the evaluator's pool (it was constructed with a placeholder before training).
        self.evaluator.set_eval_pool(eval_pool)
        self.evaluator.evaluate_references_once()

        train_rng = np.random.default_rng(self.cfg.seed)
        obs, _info = self.env.reset(seed=int(train_rng.integers(0, 2**31 - 1)))
        ep_return = 0.0
        ep_len = 0

        for step in range(1, self.cfg.total_env_steps + 1):
            action = self.agent.act(obs, eval_mode=False)
            next_obs, raw_reward, terminated, truncated, info = self.env.step(action)
            self.buffer.push(obs, action, raw_reward * self.cfg.reward_scale, next_obs, terminated)
            ep_return += raw_reward
            ep_len += 1

            if self.buffer.size >= self.cfg.warmup_steps:
                metrics = self.agent.update(self.buffer.sample(self.cfg.batch_size))
                if step % self.cfg.internals_log_every == 0:
                    self.logger.log_internals(step, metrics)

            if terminated or truncated:
                self.logger.log_episode(step, ep_return, ep_len, info["terminal_reason"])
                obs, _info = self.env.reset(seed=int(train_rng.integers(0, 2**31 - 1)))
                ep_return = 0.0
                ep_len = 0
            else:
                obs = next_obs

            if step % self.cfg.eval_cadence == 0:
                aggs = self.evaluator.evaluate(self.agent, step=step, algo_name=self.algo_name)
                self.logger.log_checkpoint(step, aggs)
                ckpt_dir = self.cfg.out_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.agent.save(ckpt_dir / f"step_{step}.pt")
