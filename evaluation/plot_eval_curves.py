# evaluation/plot_eval_curves.py

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


RUNS = [
    {
        "run_name": "dqn_reward1",
        "algo": "dqn",
        "reward_version": "reward1",
        "path": "sample_runs/reward1/standard-dqn-reward1",
    },
    {
        "run_name": "ppo_reward1",
        "algo": "ppo",
        "reward_version": "reward1",
        "path": "sample_runs/reward1/standard-ppo-reward1",
    },
    {
        "run_name": "sac_reward1",
        "algo": "sac",
        "reward_version": "reward1",
        "path": "sample_runs/reward1/standard-sac-reward1",
    },
    {
        "run_name": "sac_kl_f_reward1",
        "algo": "sac_kl_f",
        "reward_version": "reward1",
        "path": "sample_runs/reward1/standard-sac_kl_f-reward1",
    },
]


METRIC_SPECS = {
    # Core checkpoint metrics
    "avg_return": {
        "source": "reward_mean",
        "title": "Eval Reward (raw) vs env-steps",
        "ylabel": "Eval Reward (raw)",
        "filename": "eval_reward_raw.png",
    },
    "mortality_rate": {
        "source": "mortality_rate",
        "title": "Mortality Rate vs env-steps",
        "ylabel": "Mortality Rate",
        "filename": "mortality_rate.png",
    },
    "discharge_rate": {
        "source": "discharge_rate",
        "title": "Discharge Rate vs env-steps",
        "ylabel": "Discharge Rate",
        "filename": "discharge_rate.png",
    },
    "timeout_rate": {
        "source": "timeout_rate",
        "title": "Timeout Rate vs env-steps",
        "ylabel": "Timeout Rate",
        "filename": "timeout_rate.png",
    },
    "avg_episode_length": {
        "source": "ep_length_mean",
        "title": "Mean Episode Length vs env-steps",
        "ylabel": "Mean Episode Length",
        "filename": "mean_episode_length.png",
    },
    "infeasible_action_rate": {
        "source": "clamp_rate",
        "title": "Infeasible Action Rate vs env-steps",
        "ylabel": "Clamp Rate",
        "filename": "infeasible_action_rate.png",
    },

    # SOC dwell fractions
    "async_rate": {
        "source": "async_rate",
        "title": "Async Dwell Fraction vs env-steps",
        "ylabel": "Async Dwell Fraction",
        "filename": "async_rate.png",
    },
    "ambulatory_rate": {
        "source": "ambulatory_rate",
        "title": "Ambulatory Dwell Fraction vs env-steps",
        "ylabel": "Ambulatory Dwell Fraction",
        "filename": "ambulatory_rate.png",
    },
    "facility_rate": {
        "source": "facility_rate",
        "title": "Facility Dwell Fraction vs env-steps",
        "ylabel": "Facility Dwell Fraction",
        "filename": "facility_rate.png",
    },
    "icu_rate": {
        "source": "icu_rate",
        "title": "ICU Dwell Fraction vs env-steps",
        "ylabel": "ICU Dwell Fraction",
        "filename": "icu_rate.png",
    },

    # Action histogram diagnostics
    "action_entropy": {
        "source": "action_entropy",
        "title": "Action Entropy vs env-steps",
        "ylabel": "Action Entropy",
        "filename": "action_entropy.png",
    },
    "top_action_fraction": {
        "source": "top_action_fraction",
        "title": "Top Action Fraction vs env-steps",
        "ylabel": "Top Action Fraction",
        "filename": "top_action_fraction.png",
    },
}


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
    """
    Compute action distribution diagnostics from action_hist.

    action_entropy:
        Higher = more diverse action usage.
    top_action_fraction:
        Fraction of all actions assigned to the most frequent action.
        Higher = more action concentration / possible action collapse.
    """
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

        # Original checkpoint fields
        "reward_mean": row.get("reward_mean"),
        "reward_std": row.get("reward_std"),
        "mortality_rate": row.get("mortality_rate"),
        "discharge_rate": row.get("discharge_rate"),
        "timeout_rate": row.get("timeout_rate"),
        "ep_length_mean": row.get("ep_length_mean"),
        "clamp_rate": row.get("clamp_rate"),
        "n_episodes": row.get("n_episodes"),

        # Derived checkpoint-level diagnostics
        **soc_rates,
        **action_stats,
    }


def load_eval_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    learned_rows_all = []
    baseline_rows_all = []

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

            # Important: align across seeds by eval_idx, not exact env step.
            for eval_idx, row in enumerate(learned_rows):
                learned_rows_all.append(
                    flatten_eval_row(
                        row=row,
                        run=run,
                        seed=seed,
                        eval_idx=eval_idx,
                    )
                )

            # Baselines, usually only step 0.
            for row in rows:
                if row.get("algo") in {"random", "noop"}:
                    baseline_rows_all.append(
                        flatten_eval_row(
                            row=row,
                            run=run,
                            seed=seed,
                            eval_idx=None,
                        )
                    )

    learned_df = pd.DataFrame(learned_rows_all)
    baseline_df = pd.DataFrame(baseline_rows_all)

    if learned_df.empty:
        raise RuntimeError("No learned eval rows loaded. Check RUNS paths and algo names.")

    return learned_df, baseline_df


def plot_metric(
    learned_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    metric_key: str,
    out_dir: Path,
) -> None:
    spec = METRIC_SPECS[metric_key]
    source_col = spec["source"]

    if source_col not in learned_df.columns:
        print(f"[WARN] Skipping {metric_key}: missing column {source_col}")
        return

    plt.figure(figsize=(11, 6))

    for algo, group in learned_df.groupby("algo"):
        summary = (
            group.groupby("eval_idx")
            .agg(
                step_mean=("step", "mean"),
                mean=(source_col, "mean"),
                std=(source_col, "std"),
                count=(source_col, "count"),
            )
            .reset_index()
            .sort_values("eval_idx")
        )

        # Drop rows where this metric is missing.
        summary = summary.dropna(subset=["mean"])
        if summary.empty:
            continue

        x = summary["step_mean"].to_numpy()
        mean = summary["mean"].to_numpy()
        std = summary["std"].fillna(0.0).to_numpy()

        n_seeds = group["seed"].nunique()
        label = f"{algo.upper()} (mean ± std, n={n_seeds} seeds)"

        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.18)

    # Baseline horizontal lines.
    if not baseline_df.empty and source_col in baseline_df.columns:
        for baseline in ["random", "noop"]:
            base = baseline_df[baseline_df["eval_policy"] == baseline]
            if not base.empty:
                y = base[source_col].dropna().mean()
                if pd.notna(y):
                    plt.axhline(y, linestyle="--", linewidth=1.5, label=baseline)

    plt.title(spec["title"])
    plt.xlabel("env steps")
    plt.ylabel(spec["ylabel"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / spec["filename"]
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/eval_reward1/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    learned_df, baseline_df = load_eval_rows()

    # Save plot source data for debugging.
    source_dir = out_dir.parent
    source_dir.mkdir(parents=True, exist_ok=True)

    learned_csv = source_dir / "eval_plot_source_learned.csv"
    baseline_csv = source_dir / "eval_plot_source_baselines.csv"

    learned_df.to_csv(learned_csv, index=False)
    baseline_df.to_csv(baseline_csv, index=False)

    print(f"Wrote: {learned_csv}")
    print(f"Wrote: {baseline_csv}")

    print("\nLoaded learned eval rows:")
    print(
        learned_df.groupby(["algo"])["seed"]
        .nunique()
        .reset_index(name="n_seeds")
        .to_string(index=False)
    )

    for metric_key in METRIC_SPECS:
        plot_metric(learned_df, baseline_df, metric_key, out_dir)


if __name__ == "__main__":
    main()