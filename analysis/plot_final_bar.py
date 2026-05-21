"""Cross-part headline bar chart: final-checkpoint metrics across all 7 trained algos + 2 refs.

Sidesteps the env-step vs grad-step x-axis mismatch by only showing each algo's final value.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_METRICS = [
    ("reward_mean", "final_bar_reward.png", "Eval Reward (raw)"),
    ("mortality_rate", "final_bar_mortality.png", "Mortality Rate"),
    ("discharge_rate", "final_bar_discharge.png", "Discharge Rate"),
    ("ep_length_mean", "final_bar_ep_length.png", "Mean Episode Length (steps)"),
]

_ALGO_ORDER = ["dqn", "ppo", "sac", "sac_kl_f", "sac_kl_ppo",
               "iql", "iql_kl_f",
               "mbpo_mcts_main", "mbpo_mcts_no_mcts",
               "random", "noop"]
_ALGO_COLORS = {
    "dqn":    "#1f77b4",
    "ppo":    "#d62728",
    "sac":    "#2ca02c",
    "sac_kl_f": "#ff7f0e",
    "sac_kl_ppo": "#17becf",
    "iql":    "#8c564b",
    "iql_kl_f": "#e377c2",
    # Note: plot_learning_curves_part4 uses #17becf for main; that clashes with
    # sac_kl_ppo in this combined bar chart, so we substitute a darker teal.
    "mbpo_mcts_main":    "#117a87",  # dark teal
    "mbpo_mcts_no_mcts": "#bcbd22",  # olive
    "random": "#7f7f7f",
    "noop":   "#9467bd",
}


def _final_per_seed(df: pd.DataFrame, algo: str) -> pd.DataFrame:
    """For a trained algo: take the last step's row per seed. For a reference: dedupe by seed."""
    sub = df[df["algo"] == algo]
    if sub.empty:
        return sub
    if algo in ("random", "noop"):
        return sub.drop_duplicates(subset=["seed"])
    return sub.sort_values("step").drop_duplicates(subset=["seed"], keep="last")


def _plot_one(df: pd.DataFrame, metric: str, out_path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    xs = np.arange(len(_ALGO_ORDER))
    means, stds, colors, labels = [], [], [], []
    for algo in _ALGO_ORDER:
        rows = _final_per_seed(df, algo)
        if rows.empty:
            means.append(np.nan); stds.append(0.0)
        else:
            means.append(float(rows[metric].mean()))
            stds.append(float(rows[metric].std(ddof=0) or 0.0))
        colors.append(_ALGO_COLORS[algo])
        labels.append(algo)
    means_arr = np.array(means, dtype=float)
    stds_arr  = np.array(stds, dtype=float)

    ax.bar(xs, means_arr, yerr=stds_arr, color=colors, capsize=4,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Final-checkpoint {ylabel} (mean ± std across seeds, 5 seeds per algo)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aggregated", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/phase1_ppo_dqn/figures"))
    args = p.parse_args()

    df = pd.read_parquet(args.aggregated)
    if df.empty:
        raise SystemExit(f"no rows in {args.aggregated}")

    for metric, filename, ylabel in _METRICS:
        out_path = args.out_dir / filename
        _plot_one(df, metric, out_path, ylabel)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
