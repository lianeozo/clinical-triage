"""Scan results run directories, emit two tidy parquet tables.

learning_curves.parquet: long-form per (run, algo, seed, step) — one row per
eval_checkpoints.jsonl record (algo + Random + NoOp).

action_distributions.parquet: long-form per (run, seed, step, soc, num_abnormal,
action_idx). One row per non-zero action count in each checkpoint's
action_hist_by_soc_x_abnormal dict.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


_LEARNING_COLS = [
    "run_id", "algo", "seed", "step", "n_episodes",
    "reward_mean", "reward_std",
    "mortality_rate", "discharge_rate", "timeout_rate",
    "ep_length_mean", "clamp_rate",
]

# Keys in action_hist_by_soc_x_abnormal look like "soc_<i>_abn_<j>".
_SOC_ABN_KEY_RE = re.compile(r"soc_(\d+)_abn_(\d+)")


def _parse_seed_dir_name(name: str) -> int | None:
    """Extract integer from 'seed_<N>' or return None if it doesn't match."""
    if not name.startswith("seed_"):
        return None
    try:
        return int(name[len("seed_"):])
    except ValueError:
        return None


def _aggregate_one_seed(run_id: str, seed: int, seed_dir: Path) -> tuple[list[dict], list[dict]]:
    """Parse one seed_<N>/eval_checkpoints.jsonl into long-form rows for both tables."""
    cp_path = seed_dir / "eval_checkpoints.jsonl"
    if not cp_path.exists():
        return [], []

    lc_rows: list[dict] = []
    ad_rows: list[dict] = []
    with open(cp_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # learning_curves row.
            lc_row = {
                "run_id": run_id,
                "algo": rec["algo"],
                "seed": seed,
                "step": int(rec["step"]),
                "n_episodes": int(rec["n_episodes"]),
                "reward_mean": float(rec["reward_mean"]),
                "reward_std": float(rec["reward_std"]),
                "mortality_rate": float(rec["mortality_rate"]),
                "discharge_rate": float(rec["discharge_rate"]),
                "timeout_rate": float(rec["timeout_rate"]),
                "ep_length_mean": float(rec["ep_length_mean"]),
                "clamp_rate": float(rec["clamp_rate"]),
            }
            lc_rows.append(lc_row)

            # action_distributions rows from action_hist_by_soc_x_abnormal.
            sxa = rec.get("action_hist_by_soc_x_abnormal", {})
            for key, hist in sxa.items():
                m = _SOC_ABN_KEY_RE.match(key)
                if not m:
                    continue
                soc = int(m.group(1))
                num_abnormal = int(m.group(2))
                total = sum(hist)
                if total == 0:
                    continue
                for action_idx, count in enumerate(hist):
                    if count == 0:
                        continue
                    ad_rows.append({
                        "run_id": run_id,
                        "algo": rec["algo"],
                        "seed": seed,
                        "step": int(rec["step"]),
                        "soc": soc,
                        "num_abnormal": num_abnormal,
                        "action_idx": int(action_idx),
                        "count": int(count),
                        "fraction": float(count) / float(total),
                    })
    return lc_rows, ad_rows


def aggregate_run_dirs(results_root: Path, run_pattern: str = "*") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk results_root/<run-pattern>/seed_*/ and return (learning_curves_df, action_distributions_df)."""
    results_root = Path(results_root)
    all_lc: list[dict] = []
    all_ad: list[dict] = []
    for run_dir in sorted(results_root.glob(run_pattern)):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        for seed_dir in sorted(run_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            seed = _parse_seed_dir_name(seed_dir.name)
            if seed is None:
                continue
            lc_rows, ad_rows = _aggregate_one_seed(run_id, seed, seed_dir)
            all_lc.extend(lc_rows)
            all_ad.extend(ad_rows)

    lc_df = pd.DataFrame(all_lc, columns=_LEARNING_COLS) if all_lc else pd.DataFrame(columns=_LEARNING_COLS)
    ad_df = pd.DataFrame(all_ad) if all_ad else pd.DataFrame(
        columns=["run_id", "algo", "seed", "step", "soc", "num_abnormal", "action_idx", "count", "fraction"])
    return lc_df, ad_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("results/phase1_ppo_dqn"))
    p.add_argument("--run-pattern", default="*-standard-*",
                   help="glob pattern (relative to results-root) for run dirs to include")
    p.add_argument("--out-dir", type=Path, default=Path("results/phase1_ppo_dqn/aggregated"))
    args = p.parse_args()

    lc_df, ad_df = aggregate_run_dirs(args.results_root, run_pattern=args.run_pattern)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lc_path = args.out_dir / "learning_curves.parquet"
    ad_path = args.out_dir / "action_distributions.parquet"
    lc_df.to_parquet(lc_path, index=False)
    ad_df.to_parquet(ad_path, index=False)
    print(f"learning_curves: {len(lc_df)} rows -> {lc_path}")
    print(f"action_distributions: {len(ad_df)} rows -> {ad_path}")


if __name__ == "__main__":
    main()
