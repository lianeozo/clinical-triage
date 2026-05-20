"""Entry point: python -m phase2_sac.train --algo {sac,sac_kl_f,sac_kl_ppo,random,noop} --preset {smoke,standard} ..."""
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
from triage_rl.buffers.replay import ReplayBuffer
from triage_rl.config import (EnvConfig, OffPolicyConfig,
                              SACAgentConfig, SACKLFAgentConfig, SACKLPPOAgentConfig)
from triage_rl.env import Env
from triage_rl.evaluator import Evaluator
from triage_rl.logger import Logger
from triage_rl.trainers.off_policy import OffPolicyTrainer, make_eval_pool
from phase2_sac.agents.sac import SACAgent
from phase2_sac.agents.sac_kl_f import SACKLFAgent
from phase2_sac.agents.sac_kl_ppo import SACKLPPOAgent
from phase2_sac.presets import PRESETS


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


def _serialize_config(obj):
    if dataclasses.is_dataclass(obj):
        return {k: _serialize_config(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize_config(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_config(x) for x in obj]
    return obj


def run_one_seed(algo: str, preset_name: str, seed: int, out_root: Path,
                 eval_only: bool, run_name: str, device: str,
                 ppo_run_dir: Path | None = None) -> Path:
    preset = PRESETS[preset_name]
    out_dir = out_root / run_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = EnvConfig()
    env = Env(p_diabetes=env_cfg.p_diabetes, max_steps=env_cfg.max_steps)
    logger = Logger(out_dir)
    refs = {"random": RandomAgent(seed=seed), "noop": NoOpAgent()}

    if eval_only:
        assert algo in refs, f"--eval-only requires algo in {list(refs.keys())}"
        agent = refs[algo]
        pool = make_eval_pool(seed=seed, n=preset["n_eval_episodes"])
        (out_dir / "eval_pool.json").write_text(json.dumps(pool))
        evaluator = Evaluator(eval_pool=pool,
                              env_factory=lambda: Env(p_diabetes=env_cfg.p_diabetes,
                                                       max_steps=env_cfg.max_steps),
                              logger=logger, reference_agents={})
        aggs = evaluator.evaluate(agent, step=0, algo_name=algo)
        logger.log_checkpoint(0, aggs)
        (out_dir / "config.json").write_text(json.dumps(
            {"algo": algo, "preset": preset_name, "seed": seed, "eval_only": True}, indent=2))
        logger.close()
        return out_dir

    # Build trained agent.
    if algo == "sac":
        agent_cfg = SACAgentConfig()
        agent = SACAgent(obs_dim=State.NUM_STATE_VARS,
                         n_actions=Action.NUM_ACTIONS_TOTAL,
                         config=agent_cfg, seed=seed, device=device)
    elif algo == "sac_kl_f":
        agent_cfg = SACKLFAgentConfig()
        agent = SACKLFAgent(obs_dim=State.NUM_STATE_VARS,
                           n_actions=Action.NUM_ACTIONS_TOTAL,
                           config=agent_cfg, seed=seed, device=device)
    elif algo == "sac_kl_ppo":
        if ppo_run_dir is None:
            raise SystemExit("--algo sac_kl_ppo requires --ppo-run-dir")
        agent_cfg = SACKLPPOAgentConfig()
        agent = SACKLPPOAgent(obs_dim=State.NUM_STATE_VARS,
                           n_actions=Action.NUM_ACTIONS_TOTAL,
                           config=agent_cfg, seed=seed, device=device,
                           ppo_run_dir=ppo_run_dir)
    else:
        raise SystemExit(f"unknown algo {algo!r}; valid: sac, sac_kl_f, sac_kl_ppo, random, noop")

    buf = ReplayBuffer(capacity=preset["total_env_steps"],
                       obs_dim=State.NUM_STATE_VARS, seed=seed)
    trainer_cfg = OffPolicyConfig(
        total_env_steps=preset["total_env_steps"],
        seed=seed, out_dir=out_dir,
        eval_cadence=preset["eval_cadence"],
        n_eval_episodes=preset["n_eval_episodes"],
    )
    evaluator = Evaluator(
        eval_pool=[],
        env_factory=lambda: Env(p_diabetes=env_cfg.p_diabetes, max_steps=env_cfg.max_steps),
        logger=logger, reference_agents=refs,
    )
    trainer = OffPolicyTrainer(env, agent, buf, evaluator, logger, trainer_cfg, algo_name=algo)
    (out_dir / "config.json").write_text(json.dumps({
        "algo": algo, "preset": preset_name, "seed": seed,
        "env": _serialize_config(env_cfg),
        "agent": _serialize_config(agent_cfg),
        "trainer": _serialize_config(trainer_cfg),
        "ppo_run_dir": str(ppo_run_dir) if ppo_run_dir else None,
    }, indent=2))
    trainer.run()
    logger.close()
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["sac", "sac_kl_f", "sac_kl_ppo", "random", "noop"])
    p.add_argument("--preset", required=True, choices=list(PRESETS.keys()))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--all-seeds", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-root", default="results/phase1_ppo_dqn",
                   help="default points at phase1's results root (shared output dir, see Phase 2 spec §3.3)")
    p.add_argument("--device", default="auto")
    p.add_argument("--ppo-run-dir", type=Path, default=None,
                   help="required for --algo sac_kl_ppo; path to a Phase-1 PPO run dir containing seed_<N>/checkpoints/*.pt")
    args = p.parse_args()

    if args.all_seeds and args.seed is not None:
        raise SystemExit("--seed and --all-seeds are mutually exclusive")
    if args.algo == "sac_kl_ppo" and not args.eval_only and args.ppo_run_dir is None:
        raise SystemExit("--algo sac_kl_ppo requires --ppo-run-dir")

    seeds = PRESETS[args.preset]["seeds"] if args.all_seeds else [args.seed or 0]
    out_root = Path(args.out_root)
    run_name = _run_name(args.preset, args.algo, args.tag)
    device = _resolve_device(args.device)
    print(f"[device] using {device}")
    for s in seeds:
        out_dir = run_one_seed(args.algo, args.preset, s, out_root,
                               eval_only=args.eval_only, run_name=run_name,
                               device=device, ppo_run_dir=args.ppo_run_dir)
        print(f"[done] seed={s} -> {out_dir}")


if __name__ == "__main__":
    main()
