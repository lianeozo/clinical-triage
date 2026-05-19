"""On-policy training loop (PPO)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from triage_rl.buffers.rollout import RolloutBuffer
from triage_rl.config import OnPolicyConfig
from triage_rl.env import Env
from triage_rl.evaluator import Evaluator
from triage_rl.logger import Logger
from triage_rl.trainers.off_policy import seed_everything, make_eval_pool
from phase1_ppo_dqn.agents.ppo import PPOAgent


class OnPolicyTrainer:
    def __init__(self, env: Env, agent: PPOAgent, buffer: RolloutBuffer,
                 evaluator: Evaluator, logger: Logger, config: OnPolicyConfig,
                 algo_name: str = "ppo") -> None:
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.evaluator = evaluator
        self.logger = logger
        self.cfg = config
        self.algo_name = algo_name

    def _collect_rollouts(self, train_rng: np.random.Generator) -> dict:
        """Collect cfg.rollout_episodes episodes; return packed lists for the buffer.

        Returns dict with keys: obs, next_obs, actions, rewards_raw, terminated,
        log_probs, values, next_values, ep_returns, ep_lengths, terminal_reasons.
        """
        obs_list, next_obs_list = [], []
        actions, rewards_raw, term_flags = [], [], []
        log_probs, values_list, next_values_list = [], [], []
        ep_returns, ep_lengths, ep_terms = [], [], []

        for _ in range(self.cfg.rollout_episodes):
            obs, _ = self.env.reset(seed=int(train_rng.integers(0, 2**31 - 1)))
            ep_ret = 0.0
            ep_len = 0
            term_reason = "timeout"
            for _step in range(self.cfg.max_steps_per_episode):
                # Sample action + record value + log_prob in one go.
                a = self.agent.act(obs, eval_mode=False)
                logp_arr, val_arr = self.agent.get_logp_value(
                    obs[np.newaxis, :], np.array([a], dtype=np.int64))
                next_obs, raw_reward, terminated, truncated, info = self.env.step(a)
                _, nval_arr = self.agent.get_logp_value(
                    next_obs[np.newaxis, :], np.array([0], dtype=np.int64))
                obs_list.append(obs)
                next_obs_list.append(next_obs)
                actions.append(int(a))
                rewards_raw.append(float(raw_reward))
                term_flags.append(bool(terminated))
                log_probs.append(float(logp_arr[0]))
                values_list.append(float(val_arr[0]))
                next_values_list.append(float(nval_arr[0]) if not terminated else 0.0)
                ep_ret += raw_reward
                ep_len += 1
                obs = next_obs
                if terminated or truncated:
                    term_reason = info["terminal_reason"]
                    break
            ep_returns.append(ep_ret)
            ep_lengths.append(ep_len)
            ep_terms.append(term_reason)

        return {
            "obs": obs_list, "next_obs": next_obs_list,
            "actions": actions, "rewards_raw": rewards_raw,
            "terminated": term_flags, "log_probs": log_probs,
            "values": values_list, "next_values": next_values_list,
            "ep_returns": ep_returns, "ep_lengths": ep_lengths,
            "terminal_reasons": ep_terms,
        }

    def run(self) -> None:
        seed_everything(self.cfg.seed)
        eval_pool = make_eval_pool(self.cfg.seed, n=self.cfg.n_eval_episodes)
        (self.cfg.out_dir / "eval_pool.json").write_text(json.dumps(eval_pool))
        self.evaluator.set_eval_pool(eval_pool)
        self.evaluator.evaluate_references_once()

        train_rng = np.random.default_rng(self.cfg.seed)
        env_steps_done = 0
        next_eval_at = self.cfg.eval_cadence

        while env_steps_done < self.cfg.total_env_steps:
            roll = self._collect_rollouts(train_rng)
            env_steps_done += sum(roll["ep_lengths"])

            self.buffer.fill(
                obs=roll["obs"], actions=roll["actions"], rewards_raw=roll["rewards_raw"],
                terminated=roll["terminated"], log_probs=roll["log_probs"],
                values=roll["values"], next_values=roll["next_values"],
                gamma=self.agent.cfg.gamma, lam=self.agent.cfg.gae_lambda,
                reward_scale=self.cfg.reward_scale,
            )

            agg_metrics: dict[str, list[float]] = {}
            for _ in range(self.cfg.update_epochs):
                for mb in self.buffer.minibatches(self.cfg.minibatch_size):
                    m = self.agent.update(mb)
                    for k, v in m.items():
                        agg_metrics.setdefault(k, []).append(v)
            mean_metrics = {k: float(np.mean(v)) for k, v in agg_metrics.items()}
            self.logger.log_internals(env_steps_done, mean_metrics)

            for ep_ret, ep_len, ep_term in zip(roll["ep_returns"], roll["ep_lengths"], roll["terminal_reasons"]):
                self.logger.log_episode(env_steps_done, ep_ret, ep_len, ep_term)

            if env_steps_done >= next_eval_at:
                aggs = self.evaluator.evaluate(self.agent, step=env_steps_done, algo_name=self.algo_name)
                self.logger.log_checkpoint(env_steps_done, aggs)
                ckpt_dir = self.cfg.out_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.agent.save(ckpt_dir / f"step_{env_steps_done}.pt")
                next_eval_at = ((env_steps_done // self.cfg.eval_cadence) + 1) * self.cfg.eval_cadence

            self.buffer.clear()
