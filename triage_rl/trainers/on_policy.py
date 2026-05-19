"""On-policy training loop (PPO)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from triage_rl.agents.base import OnPolicyAgent
from triage_rl.buffers.rollout import RolloutBuffer
from triage_rl.config import OnPolicyConfig
from triage_rl.env import Env
from triage_rl.evaluator import Evaluator
from triage_rl.logger import Logger
from triage_rl.trainers.off_policy import seed_everything, make_eval_pool


class OnPolicyTrainer:
    def __init__(self, env: Env, agent: OnPolicyAgent, buffer: RolloutBuffer,
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

        Per env-step: 1 forward pass via act_with_logp_value. Per episode:
        1 additional value-only forward to bootstrap next_value when not terminated.
        """
        obs_list, next_obs_list = [], []
        actions, rewards_raw, term_flags = [], [], []
        log_probs, values_list, next_values_list = [], [], []
        ep_returns, ep_lengths, ep_terms = [], [], []

        for _ in range(self.cfg.rollout_episodes):
            obs, _ = self.env.reset(seed=int(train_rng.integers(0, 2**31 - 1)))
            ep_obs, ep_next_obs = [], []
            ep_actions, ep_rewards, ep_term_flags = [], [], []
            ep_log_probs, ep_values = [], []
            ep_ret = 0.0
            term_reason = "timeout"
            ended_terminated = False
            for _step in range(self.cfg.max_steps_per_episode):
                a, logp, val = self.agent.act_with_logp_value(obs)
                next_obs, raw_reward, terminated, truncated, info = self.env.step(a)
                ep_obs.append(obs)
                ep_next_obs.append(next_obs)
                ep_actions.append(int(a))
                ep_rewards.append(float(raw_reward))
                ep_term_flags.append(bool(terminated))
                ep_log_probs.append(float(logp))
                ep_values.append(float(val))
                ep_ret += raw_reward
                obs = next_obs
                if terminated or truncated:
                    term_reason = info["terminal_reason"]
                    ended_terminated = bool(terminated)
                    break

            ep_len = len(ep_obs)
            # next_values[t] = values[t+1] for t<L-1, and at t=L-1 it's 0 if terminated
            # else V(final next_obs) via one value-only forward.
            if ep_len > 0:
                tail = 0.0 if ended_terminated else self.agent.value_only(ep_next_obs[-1])
                ep_next_values = ep_values[1:] + [float(tail)]
            else:
                ep_next_values = []

            obs_list.extend(ep_obs)
            next_obs_list.extend(ep_next_obs)
            actions.extend(ep_actions)
            rewards_raw.extend(ep_rewards)
            term_flags.extend(ep_term_flags)
            log_probs.extend(ep_log_probs)
            values_list.extend(ep_values)
            next_values_list.extend(ep_next_values)
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

        # Note: PPO eval steps may overshoot cadence boundaries by up to one rollout's
        # worth of env-steps. For SMOKE (rollouts ~6K, cadence 10K) and STANDARD
        # (rollouts ~6K, cadence 25K) this is at most a one-rollout drift. Cross-algo
        # plots should align by checkpoint *index*, not by exact step. The final eval
        # (after the while loop exits) is guaranteed at the actual env_steps_done.
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

        # Guaranteed final eval at run end (avoids missing eval if the last rollout's
        # env_steps_done happened to fall in the same cadence bucket as the prior eval).
        last_eval_step = next_eval_at - self.cfg.eval_cadence  # most recent eval step boundary
        if env_steps_done > last_eval_step:
            aggs = self.evaluator.evaluate(self.agent, step=env_steps_done, algo_name=self.algo_name)
            self.logger.log_checkpoint(env_steps_done, aggs)
            ckpt_dir = self.cfg.out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.agent.save(ckpt_dir / f"step_{env_steps_done}_final.pt")
