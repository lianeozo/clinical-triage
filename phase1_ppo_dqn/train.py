"""Entry point: python -m phase1_ppo_dqn.train --algo {dqn,ppo,random,noop} --preset {smoke,standard} ..."""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path

import torch

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.agents.noop import NoOpAgent
from triage_rl.agents.random import RandomAgent
from triage_rl.agents.soc_heuristic import SocHeuristicAgent
from triage_rl.buffers.replay import ReplayBuffer
from triage_rl.buffers.rollout import RolloutBuffer
from triage_rl.config import (DQNAgentConfig, EnvConfig, OffPolicyConfig,
                              OnPolicyConfig, PPOAgentConfig)
from triage_rl.env import Env
from triage_rl.evaluator import Evaluator
from triage_rl.logger import Logger
from triage_rl.trainers.off_policy import OffPolicyTrainer, make_eval_pool
from triage_rl.trainers.on_policy import OnPolicyTrainer
from phase1_ppo_dqn.agents.dqn import DQNAgent
from phase1_ppo_dqn.agents.fact_ppo import FactorizedPPOAgent
from phase1_ppo_dqn.agents.ppo import PPOAgent
from phase1_ppo_dqn.presets import PRESETS


def _run_name(preset: str, algo: str, tag: str | None) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%dT%H-%M")
    base = f"{now}-{preset}-{algo}"
    return f"{base}-{tag}" if tag else base


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _serialize_config(obj) -> dict:
    """Recursively turn dataclasses and Paths into JSON-compatible structures."""
    if dataclasses.is_dataclass(obj):
        return {k: _serialize_config(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize_config(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_config(x) for x in obj]
    return obj


def _run_eval_agent(
    agent, algo: str, preset_name: str, seed: int,
    out_dir: Path, preset: dict, make_env, logger, reward_variant: int,
) -> Path:
    pool = make_eval_pool(seed=seed, n=preset["n_eval_episodes"])
    (out_dir / "eval_pool.json").write_text(json.dumps(pool))
    evaluator = Evaluator(eval_pool=pool, env_factory=make_env,
                          logger=logger, reference_agents={})
    aggs = evaluator.evaluate(agent, step=0, algo_name=algo)
    logger.log_checkpoint(0, aggs)
    (out_dir / "config.json").write_text(json.dumps(
        {"algo": algo, "preset": preset_name, "seed": seed,
         "reward_variant": reward_variant, "eval_only": True}, indent=2))
    logger.close()
    return out_dir


def _run_on_policy(
    agent_class, algo: str, preset_name: str, seed: int,
    out_dir: Path, preset: dict, env, make_env, logger, refs,
    env_cfg, device: str,
) -> None:
    agent_cfg = PPOAgentConfig()
    agent = agent_class(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                        config=agent_cfg, seed=seed, device=device)
    buf = RolloutBuffer(seed=seed)
    trainer_cfg = OnPolicyConfig(
        total_env_steps=preset["total_env_steps"],
        seed=seed,
        out_dir=out_dir,
        eval_cadence=preset["eval_cadence"],
        n_eval_episodes=preset["n_eval_episodes"],
    )
    evaluator = Evaluator(eval_pool=[], env_factory=make_env,
                          logger=logger, reference_agents=refs)
    trainer = OnPolicyTrainer(env, agent, buf, evaluator, logger, trainer_cfg, algo_name=algo)
    (out_dir / "config.json").write_text(json.dumps({
        "algo": algo, "preset": preset_name, "seed": seed,
        "env": _serialize_config(env_cfg),
        "agent": _serialize_config(agent_cfg),
        "trainer": _serialize_config(trainer_cfg),
    }, indent=2))
    trainer.run()


def run_one_seed(algo: str, preset_name: str, seed: int, out_root: Path,
                 eval_only: bool, run_name: str, device: str,
                 reward_variant: int = 0) -> Path:
    preset = PRESETS[preset_name]
    out_dir = out_root / run_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = EnvConfig(reward_variant=reward_variant)
    def _make_env():
        return Env(p_diabetes=env_cfg.p_diabetes, max_steps=env_cfg.max_steps,
                   reward_variant=env_cfg.reward_variant)
    env = _make_env()
    logger = Logger(out_dir)

    # Reference agents are always Random + NoOp.
    refs = {"random": RandomAgent(seed=seed), "noop": NoOpAgent()}

    if eval_only and algo in refs:
        return _run_eval_agent(refs[algo], algo, preset_name, seed, out_dir,
                               preset, _make_env, logger, reward_variant)

    if algo == "heuristic":
        return _run_eval_agent(SocHeuristicAgent(), "heuristic", preset_name, seed,
                               out_dir, preset, _make_env, logger, reward_variant)

    # Trained run.
    if algo in ("dqn", "ddqn"):
        agent_cfg = DQNAgentConfig(double_dqn=(algo == "ddqn"))
        agent = DQNAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                         total_env_steps=preset["total_env_steps"], config=agent_cfg,
                         seed=seed, device=device)
        buf = ReplayBuffer(capacity=preset["total_env_steps"],
                           obs_dim=State.NUM_STATE_VARS, seed=seed)
        trainer_cfg = OffPolicyConfig(
            total_env_steps=preset["total_env_steps"],
            seed=seed,
            out_dir=out_dir,
            eval_cadence=preset["eval_cadence"],
            n_eval_episodes=preset["n_eval_episodes"],
        )
        evaluator = Evaluator(eval_pool=[], env_factory=_make_env,
                              logger=logger, reference_agents=refs)
        trainer = OffPolicyTrainer(env, agent, buf, evaluator, logger, trainer_cfg, algo_name=algo)
        (out_dir / "config.json").write_text(json.dumps({
            "algo": algo, "preset": preset_name, "seed": seed,
            "env": _serialize_config(env_cfg),
            "agent": _serialize_config(agent_cfg),
            "trainer": _serialize_config(trainer_cfg),
        }, indent=2))
        trainer.run()

    elif algo in ("ppo", "factppo"):
        agent_class = PPOAgent if algo == "ppo" else FactorizedPPOAgent
        _run_on_policy(agent_class, algo, preset_name, seed, out_dir, preset,
                       env, _make_env, logger, refs, env_cfg, device)

    else:
        raise SystemExit(f"unknown algo {algo!r} for training; use --eval-only for reference agents")

    logger.close()
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["dqn", "ddqn", "ppo", "factppo", "random", "noop", "heuristic"])
    p.add_argument("--preset", required=True, choices=list(PRESETS.keys()))
    p.add_argument("--seed", type=int, default=None,
                   help="single seed; mutually exclusive with --all-seeds")
    p.add_argument("--all-seeds", action="store_true",
                   help="iterate over preset['seeds'] sequentially")
    p.add_argument("--eval-only", action="store_true",
                   help="run a reference agent (random|noop) through the eval pool once; no training")
    p.add_argument("--tag", default=None, help="optional suffix on run name")
    p.add_argument("--reward-variant", type=int, default=0, choices=[0, 1, 2, 3, 4],
                   help="reward variant 0–4 (default 0); see MDP.REWARD_VARIANTS")
    p.add_argument("--run-name", default=None,
                   help="override auto-generated run name (used by Modal to fix per-seed timestamps)")
    p.add_argument("--out-root", default="results/phase1_ppo_dqn",
                   help="output root (gitignored)")
    p.add_argument("--device", default="auto",
                   help="torch device: auto (default — picks cuda > mps > cpu), cpu, cuda, mps")
    args = p.parse_args()

    if args.all_seeds and args.seed is not None:
        raise SystemExit("--seed and --all-seeds are mutually exclusive")

    seeds = PRESETS[args.preset]["seeds"] if args.all_seeds else [args.seed or 0]
    out_root = Path(args.out_root)
    tag = args.tag if args.tag is not None else f"reward{args.reward_variant}"
    run_name = args.run_name if args.run_name else _run_name(args.preset, args.algo, tag)
    device = _resolve_device(args.device)
    print(f"[device] using {device}  [reward_variant] {args.reward_variant}")
    for s in seeds:
        out_dir = run_one_seed(args.algo, args.preset, s, out_root,
                               eval_only=args.eval_only, run_name=run_name, device=device,
                               reward_variant=args.reward_variant)
        print(f"[done] seed={s} -> {out_dir}")


if __name__ == "__main__":
    main()
