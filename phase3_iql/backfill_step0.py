"""One-off: produce step-0 eval records for Phase 3 IQL runs that completed before
step-0 eval was added to OfflineTrainer.run() in commit (this commit).

Replicates the trainer's init sequence exactly (seed -> agent build -> pool ->
evaluator) and appends ONE step-0 record to each existing eval_checkpoints.jsonl.

Usage:
    python -m phase3_iql.backfill_step0 \
        --run-dir results/phase1_ppo_dqn/_modal_pull/2026-05-21T00-07-standard-iql \
        --algo iql --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.agents.noop import NoOpAgent
from triage_rl.agents.random import RandomAgent
from triage_rl.config import EnvConfig, IQLAgentConfig, IQLKLFAgentConfig
from triage_rl.env import Env
from triage_rl.evaluator import Evaluator
from triage_rl.logger import Logger
from triage_rl.trainers.off_policy import make_eval_pool, seed_everything
from phase3_iql.agents.iql import IQLAgent
from phase3_iql.agents.iql_kl_f import IQLKLFAgent


def backfill_seed(algo: str, seed: int, seed_dir: Path,
                  n_eval_episodes: int, device: str) -> None:
    env_cfg = EnvConfig()
    logger = Logger(seed_dir)

    if algo == "iql":
        agent_cfg = IQLAgentConfig()
        agent = IQLAgent(obs_dim=State.NUM_STATE_VARS,
                         n_actions=Action.NUM_ACTIONS_TOTAL,
                         config=agent_cfg, seed=seed, device=device)
    elif algo == "iql_kl_f":
        agent_cfg = IQLKLFAgentConfig()
        agent = IQLKLFAgent(obs_dim=State.NUM_STATE_VARS,
                            n_actions=Action.NUM_ACTIONS_TOTAL,
                            config=agent_cfg, seed=seed, device=device)
    else:
        raise SystemExit(f"unknown algo {algo!r}; expected iql or iql_kl_f")

    seed_everything(seed)
    pool = make_eval_pool(seed, n=n_eval_episodes)

    evaluator = Evaluator(
        eval_pool=pool,
        env_factory=lambda: Env(p_diabetes=env_cfg.p_diabetes,
                                 max_steps=env_cfg.max_steps),
        logger=logger,
        reference_agents={"random": RandomAgent(seed=seed), "noop": NoOpAgent()},
    )
    aggs = evaluator.evaluate(agent, step=0, algo_name=algo)
    logger.log_checkpoint(0, aggs)
    logger.close()
    print(f"[backfilled] {algo}/seed_{seed}: reward={aggs['reward_mean']:.1f} "
          f"mortality={aggs['mortality_rate']:.3f} clamp={aggs['clamp_rate']:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True,
                   help="run directory containing seed_<N> subdirs")
    p.add_argument("--algo", required=True, choices=["iql", "iql_kl_f"])
    p.add_argument("--seeds", default="0,1,2,3,4",
                   help="comma-separated seed ints")
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    for seed in seeds:
        seed_dir = args.run_dir / f"seed_{seed}"
        if not seed_dir.exists():
            raise SystemExit(f"missing seed dir: {seed_dir}")
        backfill_seed(args.algo, seed, seed_dir,
                      n_eval_episodes=args.n_eval_episodes,
                      device=args.device)


if __name__ == "__main__":
    main()
