"""Entry point: ``python -m phase4_mbpo_mcts.train --condition {main,no_mcts}
--preset {smoke,standard} ...``.

Mirrors the layout of ``phase3_iql/train.py``: a ``run_one_seed`` helper
that constructs the four Phase 4 configs from the preset, instantiates
``OuterLoopTrainer`` with the chosen ablation switch, writes ``config.json``,
and calls ``trainer.run()``. ``--all-seeds`` iterates the preset's seed list.

The ``run_name`` carries the ``mbpo_mcts_`` prefix so downstream aggregators
distinguish Phase 4 algos from Phases 1-3.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path

import torch

from phase4_mbpo_mcts.presets import PRESETS
from phase4_mbpo_mcts.trainers.outer_loop import OuterLoopTrainer
from triage_rl.config import (
    EnsembleModelConfig,
    EnvConfig,
    MCTSConfig,
    OuterLoopConfig,
    PiVNetConfig,
)


_VALID_CONDITIONS = ("main", "no_mcts")


def _run_name(preset: str, condition: str, tag: str | None) -> str:
    """Format: ``{YYYY-MM-DDTHH-MM}-{preset}-mbpo_mcts_{condition}[-{tag}]``."""
    now = dt.datetime.now().strftime("%Y-%m-%dT%H-%M")
    base = f"{now}-{preset}-mbpo_mcts_{condition}"
    return f"{base}-{tag}" if tag else base


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
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


def run_one_seed(
    condition: str,
    preset_name: str,
    seed: int,
    out_root: Path,
    run_name: str,
    device: str,
    dataset_path: Path,
) -> Path:
    """Build configs + OuterLoopTrainer for one (condition, seed) and run.

    Returns the seed output directory.
    """
    if condition not in _VALID_CONDITIONS:
        raise SystemExit(
            f"unknown condition {condition!r}; valid: {list(_VALID_CONDITIONS)}"
        )

    preset = PRESETS[preset_name]
    out_dir = out_root / run_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = EnvConfig()
    model_cfg = EnsembleModelConfig()
    pi_v_cfg = PiVNetConfig()
    mcts_cfg = MCTSConfig()
    outer_cfg = OuterLoopConfig(
        seed_dataset_path=Path(dataset_path),
        out_dir=out_dir,
        n_outer_iters=preset["n_outer_iters"],
        n_episodes_per_iter=preset["n_episodes_per_iter"],
        n_model_steps_per_iter=preset["n_model_steps_per_iter"],
        n_pv_steps_per_iter=preset["n_pv_steps_per_iter"],
        n_eval_episodes=preset["n_eval_episodes"],
        seed=seed,
    )

    algo_name = f"mbpo_mcts_{condition}"
    no_mcts = condition == "no_mcts"

    trainer = OuterLoopTrainer(
        outer_cfg=outer_cfg,
        model_cfg=model_cfg,
        pi_v_cfg=pi_v_cfg,
        mcts_cfg=mcts_cfg,
        no_mcts=no_mcts,
        algo_name=algo_name,
        device=device,
    )

    (out_dir / "config.json").write_text(json.dumps({
        "algo": algo_name,
        "condition": condition,
        "preset": preset_name,
        "seed": seed,
        "device": device,
        "env": _serialize_config(env_cfg),
        "model": _serialize_config(model_cfg),
        "pi_v": _serialize_config(pi_v_cfg),
        "mcts": _serialize_config(mcts_cfg),
        "outer": _serialize_config(outer_cfg),
        "dataset_path": str(dataset_path),
    }, indent=2))

    trainer.run()
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=list(_VALID_CONDITIONS))
    p.add_argument("--preset", required=True, choices=list(PRESETS.keys()))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--all-seeds", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("results/phase1_ppo_dqn/aggregated/offline_dataset.npz"),
    )
    p.add_argument("--out-root", default="results/phase1_ppo_dqn",
                   help="default points at phase1's results root (shared output dir)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    if args.all_seeds and args.seed is not None:
        raise SystemExit("--seed and --all-seeds are mutually exclusive")

    seeds = PRESETS[args.preset]["seeds"] if args.all_seeds else [args.seed or 0]
    out_root = Path(args.out_root)
    run_name = _run_name(args.preset, args.condition, args.tag)
    device = _resolve_device(args.device)
    print(f"[device] using {device}")
    for s in seeds:
        out_dir = run_one_seed(
            condition=args.condition,
            preset_name=args.preset,
            seed=s,
            out_root=out_root,
            run_name=run_name,
            device=device,
            dataset_path=args.dataset_path,
        )
        print(f"[done] seed={s} -> {out_dir}")


if __name__ == "__main__":
    main()
