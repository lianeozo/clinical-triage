# evaluation/aggregate_eval_checkpoints.py

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REWARD_VERSION = "reward3"  # change this to reward2 / reward3 / reward0

RUNS = [
    {
        "run_name": f"dqn_{REWARD_VERSION}",
        "algo": "dqn",
        "reward_version": REWARD_VERSION,
        "path": f"sample_runs/{REWARD_VERSION}/standard-dqn-{REWARD_VERSION}",
    },
    {
        "run_name": f"ppo_{REWARD_VERSION}",
        "algo": "ppo",
        "reward_version": REWARD_VERSION,
        "path": f"sample_runs/{REWARD_VERSION}/standard-ppo-{REWARD_VERSION}",
    },
    {
        "run_name": f"sac_{REWARD_VERSION}",
        "algo": "sac",
        "reward_version": REWARD_VERSION,
        "path": f"sample_runs/{REWARD_VERSION}/standard-sac-{REWARD_VERSION}",
    },
    {
        "run_name": f"sac_kl_f_{REWARD_VERSION}",
        "algo": "sac_kl_f",
        "reward_version": REWARD_VERSION,
        "path": f"sample_runs/{REWARD_VERSION}/standard-sac_kl_f-{REWARD_VERSION}",
    },
    {
        "run_name": f"sac_kl_ppo_{REWARD_VERSION}",
        "algo": "sac_kl_ppo",
        "reward_version": REWARD_VERSION,
        "path": f"sample_runs/{REWARD_VERSION}/standard-sac_kl_ppo-{REWARD_VERSION}",
    },
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_seed(seed_dir: Path) -> int | None:
    if seed_dir.name.startswith("seed_"):
        try:
            return int(seed_dir.name.split("_")[-1])
        except ValueError:
            return None
    return None


def get_soc_rates(row: dict[str, Any]) -> dict[str, float | None]:
    soc = row.get("soc_dwell_fractions")

    if not isinstance(soc, list) or len(soc) < 4:
        return {
            "async_rate": None,
            "ambulatory_rate": None,
            "facility_rate": None,
            "icu_rate": None,
        }

    return {
        "async_rate": soc[0],
        "ambulatory_rate": soc[1],
        "facility_rate": soc[2],
        "icu_rate": soc[3],
    }


def action_hist_stats(row: dict[str, Any]) -> dict[str, float | None]:
    hist = row.get("action_hist")

    if not isinstance(hist, list) or len(hist) == 0:
        return {
            "action_entropy": None,
            "top_action_fraction": None,
        }

    total = sum(hist)
    if total <= 0:
        return {
            "action_entropy": None,
            "top_action_fraction": None,
        }

    probs = [count / total for count in hist if count > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    top_frac = max(hist) / total

    return {
        "action_entropy": entropy,
        "top_action_fraction": top_frac,
    }


def flatten_eval_row(
    row: dict[str, Any],
    run: dict[str, str],
    seed: int | None,
    eval_idx: int | None,
    seed_dir: Path,
) -> dict[str, Any]:
    soc_rates = get_soc_rates(row)
    action_stats = action_hist_stats(row)

    return {
        "run_name": run["run_name"],
        "algo": run["algo"],
        "reward_version": run["reward_version"],
        "seed": seed,
        "eval_idx": eval_idx,
        "step": row.get("step"),
        "eval_policy": row.get("algo"),

        # Report-friendly metric names
        "avg_return": row.get("reward_mean"),
        "reward_std": row.get("reward_std"),
        "mortality_rate": row.get("mortality_rate"),
        "discharge_rate": row.get("discharge_rate"),
        "timeout_rate": row.get("timeout_rate"),
        "infeasible_action_rate": row.get("clamp_rate"),
        "avg_episode_length": row.get("ep_length_mean"),
        "n_episodes": row.get("n_episodes"),

        # SOC dwell fractions
        **soc_rates,

        # Action-distribution diagnostics
        **action_stats,

        # Debugging source
        "source_path": str(seed_dir),
    }


def load_eval_rows(include_baselines: bool = True) -> pd.DataFrame:
    flat_rows: list[dict[str, Any]] = []

    for run in RUNS:
        run_path = Path(run["path"])
        if not run_path.exists():
            print(f"[WARN] Missing run path: {run_path}")
            continue

        seed_dirs = sorted([p for p in run_path.glob("seed_*") if p.is_dir()])
        if not seed_dirs:
            print(f"[WARN] No seed_* dirs found under {run_path}")
            continue

        for seed_dir in seed_dirs:
            eval_path = seed_dir / "eval_checkpoints.jsonl"
            if not eval_path.exists():
                print(f"[WARN] Missing eval file: {eval_path}")
                continue

            seed = infer_seed(seed_dir)
            rows = read_jsonl(eval_path)

            # Learned policy rows only, e.g. dqn folder keeps only algo == dqn.
            learned_rows = [row for row in rows if row.get("algo") == run["algo"]]

            # Align learned-policy rows by eval_idx, not exact env step.
            for eval_idx, row in enumerate(learned_rows):
                flat_rows.append(
                    flatten_eval_row(
                        row=row,
                        run=run,
                        seed=seed,
                        eval_idx=eval_idx,
                        seed_dir=seed_dir,
                    )
                )

            # Optional random/noop baselines.
            if include_baselines:
                for row in rows:
                    if row.get("algo") in {"random", "noop"}:
                        flat_rows.append(
                            flatten_eval_row(
                                row=row,
                                run=run,
                                seed=seed,
                                eval_idx=None,
                                seed_dir=seed_dir,
                            )
                        )

    df = pd.DataFrame(flat_rows)

    if df.empty:
        raise RuntimeError("No eval rows loaded. Check RUNS paths and eval_checkpoints.jsonl files.")

    return df


def make_final_by_seed(df: pd.DataFrame, learned_only: bool = True) -> pd.DataFrame:
    data = df.copy()

    if learned_only:
        data = data[data["eval_policy"] == data["algo"]].copy()

    group_cols = ["run_name", "algo", "reward_version", "seed", "eval_policy"]

    final_rows = []
    for _, group in data.groupby(group_cols, dropna=False):
        group = group.sort_values(["eval_idx", "step"], na_position="last")
        final_rows.append(group.iloc[-1].to_dict())

    return pd.DataFrame(final_rows)


def make_final_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "avg_return",
        "mortality_rate",
        "discharge_rate",
        "timeout_rate",
        "infeasible_action_rate",
        "avg_episode_length",
        "async_rate",
        "ambulatory_rate",
        "facility_rate",
        "icu_rate",
        "action_entropy",
        "top_action_fraction",
    ]

    summary_rows = []

    for (algo, reward_version, eval_policy), group in final_df.groupby(
        ["algo", "reward_version", "eval_policy"],
        dropna=False,
    ):
        row = {
            "algo": algo,
            "reward_version": reward_version,
            "eval_policy": eval_policy,
            "n_seeds": group["seed"].nunique(dropna=True),
        }

        if row["n_seeds"] == 0:
            row["n_seeds"] = len(group)

        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=f"outputs/eval_{REWARD_VERSION}")
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Do not include random/noop rows in eval_timeseries.csv.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_eval_rows(include_baselines=not args.no_baselines)

    df = df.sort_values(
        ["reward_version", "algo", "seed", "eval_policy", "eval_idx", "step"],
        na_position="last",
    )

    # 1. Full checkpoint-level timeseries.
    #timeseries_out = out_dir / "eval_timeseries.csv"
    timeseries_out = out_dir / f"eval_timeseries_{REWARD_VERSION}.csv"
    df.to_csv(timeseries_out, index=False)

    # 2. Final checkpoint per seed, learned policy only.
    final_by_seed = make_final_by_seed(df, learned_only=True)
    final_by_seed_out = out_dir / f"eval_final_by_seed_{REWARD_VERSION}.csv"
    #final_by_seed_out = out_dir / "eval_final_by_seed.csv"
    final_by_seed.to_csv(final_by_seed_out, index=False)

    # 3. Across-seed final summary, learned policy only.
    final_summary = make_final_summary(final_by_seed)
    #final_summary_out = out_dir / "eval_final_summary.csv"
    final_summary_out = out_dir / f"eval_final_summary_{REWARD_VERSION}.csv"
    final_summary.to_csv(final_summary_out, index=False)

    print(f"Wrote: {timeseries_out}")
    print(f"Wrote: {final_by_seed_out}")
    print(f"Wrote: {final_summary_out}")

    print("\nFinal learned-policy summary:")
    cols = [
        "algo",
        "reward_version",
        "eval_policy",
        "n_seeds",
        "avg_return_mean",
        "mortality_rate_mean",
        "discharge_rate_mean",
        "timeout_rate_mean",
        "infeasible_action_rate_mean",
        "avg_episode_length_mean",
        "async_rate_mean",
        "ambulatory_rate_mean",
        "facility_rate_mean",
        "icu_rate_mean",
        "action_entropy_mean",
        "top_action_fraction_mean",
    ]
    print(final_summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()