"""Offline training loop. No env interaction during training — only at eval cadence boundaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from triage_rl.agents.base import Agent
from triage_rl.config import OfflineConfig
from triage_rl.evaluator import Evaluator
from triage_rl.logger import Logger
from triage_rl.trainers.off_policy import make_eval_pool, seed_everything


class OfflineTrainer:
    def __init__(self, agent: Agent, evaluator: Evaluator, logger: Logger,
                 config: OfflineConfig, algo_name: str) -> None:
        self.agent = agent
        self.evaluator = evaluator
        self.logger = logger
        self.cfg = config
        self.algo_name = algo_name

    def run(self) -> None:
        seed_everything(self.cfg.seed)
        pool = make_eval_pool(self.cfg.seed, n=self.cfg.n_eval_episodes)
        (self.cfg.out_dir / "eval_pool.json").write_text(json.dumps(pool))
        self.evaluator.set_eval_pool(pool)
        self.evaluator.evaluate_references_once()

        data_np = np.load(self.cfg.dataset_path)
        n_total = len(data_np['action'])
        if n_total == 0:
            raise RuntimeError(f"empty dataset at {self.cfg.dataset_path}")
        print(f"[offline] dataset loaded: {n_total} transitions from {self.cfg.dataset_path}")

        # Preload everything to GPU as torch tensors. Dataset is ~70MB total — trivial for any GPU.
        device = self.agent.device
        data = {
            'obs':        torch.from_numpy(data_np['obs']).to(device),
            'action':     torch.from_numpy(data_np['action']).long().to(device),
            'reward':     torch.from_numpy(data_np['reward']).to(device),
            'next_obs':   torch.from_numpy(data_np['next_obs']).to(device),
            'terminated': torch.from_numpy(data_np['terminated']).to(device),
        }
        reward_scale = float(self.cfg.reward_scale)

        rng = np.random.default_rng(self.cfg.seed)
        for step in range(1, self.cfg.total_grad_steps + 1):
            idx_np = rng.integers(0, n_total, size=self.cfg.batch_size)
            idx = torch.from_numpy(idx_np).to(device)
            batch = {
                'obs':           data['obs'][idx],
                'action':        data['action'][idx],
                'next_obs':      data['next_obs'][idx],
                'terminated':    data['terminated'][idx],
                'scaled_reward': data['reward'][idx] * reward_scale,
            }
            metrics = self.agent.update(batch)
            if step % self.cfg.internals_log_every == 0:
                self.logger.log_internals(step, metrics)
            if step % self.cfg.eval_cadence == 0:
                aggs = self.evaluator.evaluate(self.agent, step=step, algo_name=self.algo_name)
                self.logger.log_checkpoint(step, aggs)
                ckpt_dir = self.cfg.out_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.agent.save(ckpt_dir / f"step_{step}.pt")
