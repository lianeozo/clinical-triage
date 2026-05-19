"""Evaluator: deterministic rollouts on a fixed eval pool; instrumentation buckets 2, 3, 5."""
from __future__ import annotations

from typing import Callable

import numpy as np

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.agents.base import Agent
from triage_rl.env import Env
from triage_rl.logger import Logger


class Evaluator:
    def __init__(self, eval_pool: list[int], env_factory: Callable[[], Env],
                 logger: Logger,
                 reference_agents: dict[str, Agent] | None = None) -> None:
        self.eval_pool = list(eval_pool)
        self.env_factory = env_factory
        self.logger = logger
        self.reference_agents = reference_agents or {}

    def set_eval_pool(self, eval_pool: list[int]) -> None:
        """Replace the eval pool. Trainer calls this once after generating the pool from its seed."""
        self.eval_pool = list(eval_pool)

    def evaluate(self, agent: Agent, step: int, algo_name: str = "agent") -> dict:
        env = self.env_factory()
        rewards, lengths, terminals, clamp_counts = [], [], [], []
        action_hist = np.zeros(Action.NUM_ACTIONS_TOTAL, dtype=np.int64)
        soc_dwell = np.zeros(State.NUM_SOC, dtype=np.int64)
        soc_trans = np.zeros((State.NUM_SOC, State.NUM_SOC), dtype=np.int64)
        # bucket #5: action_hist conditioned on (soc, num_abnormal_vitals).
        action_hist_sxa: dict[str, np.ndarray] = {}

        for ep_idx, init_seed in enumerate(self.eval_pool):
            obs, info_reset = env.reset(seed=init_seed)
            prev_soc = int(info_reset["soc"])
            ep_return = 0.0
            ep_len = 0
            ep_clamped = 0
            traj = {"episode_idx": ep_idx, "init_seed": int(init_seed),
                    "states": [], "agent_actions": [], "executed_actions": [],
                    "clamped_steps": [], "rewards_raw": [], "socs": [],
                    "num_abnormal_vitals": []}
            num_abn_now = int(info_reset["num_abnormal_vitals"])
            while True:
                a = agent.act(obs, eval_mode=True)
                # bucket #5 keying must use pre-step (soc, num_abn).
                key = f"soc_{prev_soc}_abn_{num_abn_now}"
                if key not in action_hist_sxa:
                    action_hist_sxa[key] = np.zeros(Action.NUM_ACTIONS_TOTAL, dtype=np.int64)
                action_hist_sxa[key][a] += 1
                action_hist[a] += 1

                traj["states"].append(obs.tolist())
                traj["agent_actions"].append(int(a))
                soc_dwell[prev_soc] += 1
                obs, raw_reward, terminated, truncated, info = env.step(a)
                ep_return += raw_reward
                ep_len += 1
                ep_clamped += int(info["clamped"])
                new_soc = int(info["soc"])
                soc_trans[prev_soc, new_soc] += 1
                prev_soc = new_soc
                num_abn_now = int(info["num_abnormal_vitals"])

                traj["executed_actions"].append(int(info["executed_action"]))
                traj["clamped_steps"].append(int(info["clamped"]))
                traj["rewards_raw"].append(float(raw_reward))
                traj["socs"].append(int(info["soc"]))
                traj["num_abnormal_vitals"].append(int(info["num_abnormal_vitals"]))
                if terminated or truncated:
                    traj["terminal_reason"] = info["terminal_reason"]
                    traj["length"] = ep_len
                    self.logger.log_eval_trajectory(step, ep_idx, traj)
                    rewards.append(ep_return)
                    lengths.append(ep_len)
                    terminals.append(info["terminal_reason"])
                    clamp_counts.append(ep_clamped)
                    break

        n = len(rewards)
        total_steps = sum(lengths)
        n_term = {"discharge": 0, "death": 0, "timeout": 0}
        for t in terminals:
            n_term[t] = n_term.get(t, 0) + 1

        aggregates = {
            "algo": algo_name,
            "n_episodes": n,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "ep_length_mean": float(np.mean(lengths)),
            "mortality_rate": n_term["death"] / max(n, 1),
            "discharge_rate": n_term["discharge"] / max(n, 1),
            "timeout_rate": n_term["timeout"] / max(n, 1),
            "clamp_rate": float(sum(clamp_counts) / max(total_steps, 1)),
            "action_hist": action_hist.tolist(),
            "soc_dwell_fractions": (soc_dwell / max(soc_dwell.sum(), 1)).tolist(),
            "soc_transition_counts": soc_trans.tolist(),
            "action_hist_by_soc_x_abnormal": {k: v.tolist() for k, v in action_hist_sxa.items()},
        }
        return aggregates

    def evaluate_references_once(self) -> None:
        """Run each reference agent through the eval pool once and write checkpoint rows at step=0."""
        for name, agent in self.reference_agents.items():
            aggs = self.evaluate(agent, step=0, algo_name=name)
            self.logger.log_checkpoint(step=0, aggregates=aggs)
