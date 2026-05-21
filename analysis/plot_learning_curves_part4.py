"""Part-4 only learning curves. X-axis = MBPO outer iteration.

Renders 4 PNGs (reward, mortality, discharge, ep_length) overlaying
``mbpo_mcts_main`` and ``mbpo_mcts_no_mcts`` with a horizontal dashed
``IQL @ 200K`` reference line per panel (mean over IQL seeds at the last
gradient-step checkpoint).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


_METRICS = [
    ("reward_mean", "learning_part4_reward.png", "Eval Reward (raw)"),
    ("mortality_rate", "learning_part4_mortality.png", "Mortality Rate"),
    ("discharge_rate", "learning_part4_discharge.png", "Discharge Rate"),
    ("ep_length_mean", "learning_part4_ep_length.png", "Mean Episode Length (steps)"),
]

_PART4_ALGOS = ["mbpo_mcts_main", "mbpo_mcts_no_mcts"]
_ALGO_COLORS = {
    "mbpo_mcts_main":    "#17becf",  # teal
    "mbpo_mcts_no_mcts": "#bcbd22",  # olive
}

# Phase 3 IQL trained for 200K gradient steps; we use this as the reference baseline.
_IQL_REF_STEP = 200_000


def _algo_curve_outer(df: pd.DataFrame, algo: str, metric: str) -> pd.DataFrame:
    """Mean±std curve over seeds at each outer_iter for one algo."""
    sub = df[df["algo"] == algo]
    if sub.empty:
        return pd.DataFrame(columns=["outer_iter", "mean", "std", "n"])
    # Drop rows without an outer_iter (Phases 1-3 rows would have NaN, but
    # the algo filter already excludes them — defensive nonetheless).
    sub = sub.dropna(subset=["outer_iter"])
    if sub.empty:
        return pd.DataFrame(columns=["outer_iter", "mean", "std", "n"])
    grouped = sub.groupby("outer_iter")[metric].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.rename(columns={"count": "n"})
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["outer_iter"] = grouped["outer_iter"].astype(int)
    return grouped.sort_values("outer_iter").reset_index(drop=True)


def _iql_reference_mean(original_df: pd.DataFrame, metric: str) -> float | None:
    """Mean of IQL at step==_IQL_REF_STEP across seeds, or None if not present."""
    sub = original_df[(original_df["algo"] == "iql") & (original_df["step"] == _IQL_REF_STEP)]
    if sub.empty:
        return None
    return float(sub[metric].mean())


def _plot_one_metric(
    part4_df: pd.DataFrame, original_df: pd.DataFrame,
    metric: str, out_path: Path, ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo in _PART4_ALGOS:
        curve = _algo_curve_outer(part4_df, algo, metric)
        if curve.empty:
            continue
        color = _ALGO_COLORS[algo]
        n_seeds = int(curve["n"].iloc[0])
        ax.plot(curve["outer_iter"], curve["mean"], color=color, linewidth=2,
                label=f"{algo} (mean ± std, n={n_seeds} seeds)")
        ax.fill_between(curve["outer_iter"], curve["mean"] - curve["std"],
                        curve["mean"] + curve["std"], color=color, alpha=0.2)

    iql_mean = _iql_reference_mean(original_df, metric)
    if iql_mean is not None:
        ax.axhline(y=iql_mean, color="gray", linestyle="--", alpha=0.5,
                   label=f"IQL @ {_IQL_REF_STEP // 1000}K")

    ax.set_xlabel("outer iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Part 4 — {ylabel} vs outer iteration")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aggregated", type=Path, required=True,
                   help="path to learning_curves.parquet")
    p.add_argument("--out-dir", type=Path, default=Path("results/phase1_ppo_dqn/figures"))
    args = p.parse_args()

    original_df = pd.read_parquet(args.aggregated)
    if original_df.empty:
        raise SystemExit(f"no rows in {args.aggregated}")

    part4_df = original_df[original_df["algo"].isin(_PART4_ALGOS)]
    if part4_df.empty:
        print(f"warning: no Phase 4 rows ({_PART4_ALGOS}) in {args.aggregated} — nothing to plot")
        return

    for metric, filename, ylabel in _METRICS:
        out_path = args.out_dir / filename
        _plot_one_metric(part4_df, original_df, metric, out_path, ylabel)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
