# evaluation/plot_eval_curves.py

from __future__ import annotations

import argparse
import json
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


def load_eval_rows() -> pd.DataFrame:
    all_rows = []
    baseline_rows = []

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

            # Baselines, usually only step 0.
            for row in rows:
                if row.get("algo") in {"random", "noop"}:
                    baseline_rows.append(
                        {
                            "run_name": run["run_name"],
                            "algo": run["algo"],
                            "reward_version": run["reward_version"],
                            "seed": seed,
                            "step": row.get("step"),
                            "eval_policy": row.get("algo"),
                            "reward_mean": row.get("reward_mean"),
                            "mortality_rate": row.get("mortality_rate"),
                            "discharge_rate": row.get("discharge_rate"),
                            "ep_length_mean": row.get("ep_length_mean"),
                            "clamp_rate": row.get("clamp_rate"),
                        }
                    )

            # Important: align across seeds by eval_idx, not exact env step.
            # PPO/SAC steps may be 25069, 25063, 25144, etc., so exact step grouping is wrong.
            for eval_idx, row in enumerate(learned_rows):
                all_rows.append(
                    {
                        "run_name": run["run_name"],
                        "algo": run["algo"],
                        "reward_version": run["reward_version"],
                        "seed": seed,
                        "eval_idx": eval_idx,
                        "step": row.get("step"),
                        "eval_policy": row.get("algo"),
                        "reward_mean": row.get("reward_mean"),
                        "mortality_rate": row.get("mortality_rate"),
                        "discharge_rate": row.get("discharge_rate"),
                        "ep_length_mean": row.get("ep_length_mean"),
                        "clamp_rate": row.get("clamp_rate"),
                    }
                )

    learned_df = pd.DataFrame(all_rows)
    baseline_df = pd.DataFrame(baseline_rows)

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

        x = summary["step_mean"].to_numpy()
        mean = summary["mean"].to_numpy()
        std = summary["std"].fillna(0.0).to_numpy()

        n_seeds = group["seed"].nunique()
        label = f"{algo.upper()} (mean ± std, n={n_seeds} seeds)"

        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.18)

    # Baseline horizontal lines.
    # They are averaged across all seed_*/runs for the same reward setting.
    if not baseline_df.empty:
        for baseline in ["random", "noop"]:
            base = baseline_df[baseline_df["eval_policy"] == baseline]
            if not base.empty and source_col in base:
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

    # Quick sanity check: should show 5 seeds for each algo.
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