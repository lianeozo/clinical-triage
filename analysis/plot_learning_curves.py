"""Render 4 learning-curve PNGs from learning_curves.parquet.

Each PNG overlays DQN + PPO + Random + NoOp:
- Algos: solid line (mean across seeds) + shaded band (±1 std across seeds).
- References: dashed horizontal line + shaded band (deduped by seed, so a single
  unique value per seed regardless of how many algo runs reported it).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_METRICS = [
    ("reward_mean", "learning_reward.png", "Eval Reward (raw)"),
    ("mortality_rate", "learning_mortality.png", "Mortality Rate"),
    ("discharge_rate", "learning_discharge.png", "Discharge Rate"),
    ("ep_length_mean", "learning_ep_length.png", "Mean Episode Length (steps)"),
]

_ALGO_COLORS = {
    "dqn":    "#1f77b4",  # blue
    "ppo":    "#d62728",  # red
    "qac":    "#2ca02c",  # green
    "qac_fk": "#ff7f0e",  # orange
    "qac_kp": "#17becf",  # teal
    "random": "#7f7f7f",  # grey
    "noop":   "#9467bd",  # purple
}


def _algo_curve(df: pd.DataFrame, algo: str, metric: str, cadence: int) -> pd.DataFrame:
    """For DQN/PPO: at each cadence-aligned step bucket, mean and std across seeds.

    Steps are snapped to the nearest multiple of `cadence` before grouping. This
    handles PPO's rollout-overshoot where each seed lands evals at slightly
    different env-step counts (e.g. 25031, 25150, ...) instead of the cadence
    boundary (25000).
    """
    sub = df[df["algo"] == algo]
    if sub.empty:
        return pd.DataFrame(columns=["step", "mean", "std", "n"])
    snapped = sub.copy()
    snapped["step_bin"] = ((snapped["step"] + cadence // 2) // cadence) * cadence
    grouped = snapped.groupby("step_bin")[metric].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.rename(columns={"count": "n", "step_bin": "step"})
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def _ref_band(df: pd.DataFrame, algo: str, metric: str) -> tuple[float, float] | None:
    """For Random/NoOp: dedupe by seed (references are bit-identical across algos at
    the same seed), then mean ± std across the deduped values."""
    sub = df[df["algo"] == algo]
    if sub.empty:
        return None
    deduped = sub.drop_duplicates(subset=["seed"])
    return float(deduped[metric].mean()), float(deduped[metric].std(ddof=0) or 0.0)


def _plot_one_metric(df: pd.DataFrame, metric: str, out_path: Path, ylabel: str, cadence: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    # Determine x range from algo data.
    algo_dfs = {a: _algo_curve(df, a, metric, cadence) for a in ("dqn", "ppo", "qac", "qac_fk", "qac_kp")}
    x_max = 0
    for a, curve in algo_dfs.items():
        if curve.empty:
            continue
        x_max = max(x_max, int(curve["step"].max()))

    # Algos: line + band.
    for algo in ("dqn", "ppo", "qac", "qac_fk", "qac_kp"):
        curve = algo_dfs[algo]
        if curve.empty:
            continue
        color = _ALGO_COLORS[algo]
        ax.plot(curve["step"], curve["mean"], color=color, linewidth=2,
                label=f"{algo.upper()} (mean ± std, n={int(curve['n'].iloc[0])} seeds)")
        ax.fill_between(curve["step"], curve["mean"] - curve["std"], curve["mean"] + curve["std"],
                        color=color, alpha=0.2)

    # References: dashed horizontal + faint band.
    for ref in ("random", "noop"):
        band = _ref_band(df, ref, metric)
        if band is None:
            continue
        m, s = band
        color = _ALGO_COLORS[ref]
        xs = np.array([0, x_max if x_max > 0 else 1])
        ax.axhline(m, color=color, linestyle="--", linewidth=1.5, label=f"{ref}")
        ax.fill_between(xs, m - s, m + s, color=color, alpha=0.1)

    ax.set_xlabel("env steps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs env-steps")
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
    p.add_argument("--eval-cadence", type=int, default=25_000,
                   help="snap step values to nearest multiple of this for cross-seed grouping (default 25000 = STANDARD)")
    args = p.parse_args()

    df = pd.read_parquet(args.aggregated)
    if df.empty:
        raise SystemExit(f"no rows in {args.aggregated}")

    for metric, filename, ylabel in _METRICS:
        out_path = args.out_dir / filename
        _plot_one_metric(df, metric, out_path, ylabel, args.eval_cadence)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
