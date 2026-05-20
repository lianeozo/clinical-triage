"""Part-3 only learning curves. X-axis = gradient steps (Part 3) vs env-steps (Parts 1+2).

Renders 4 PNGs (reward, mortality, discharge, ep_length) overlaying iql + iql_kl_f
with dedup-by-seed Random and NoOp references.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_METRICS = [
    ("reward_mean", "learning_part3_reward.png", "Eval Reward (raw)"),
    ("mortality_rate", "learning_part3_mortality.png", "Mortality Rate"),
    ("discharge_rate", "learning_part3_discharge.png", "Discharge Rate"),
    ("ep_length_mean", "learning_part3_ep_length.png", "Mean Episode Length (steps)"),
]

_ALGO_COLORS = {
    "iql":     "#8c564b",  # brown
    "iql_kl_f": "#e377c2",  # pink
    "random":  "#7f7f7f",  # grey
    "noop":    "#9467bd",  # purple
}


def _algo_curve(df: pd.DataFrame, algo: str, metric: str, cadence: int) -> pd.DataFrame:
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
    sub = df[df["algo"] == algo]
    if sub.empty:
        return None
    deduped = sub.drop_duplicates(subset=["seed"])
    return float(deduped[metric].mean()), float(deduped[metric].std(ddof=0) or 0.0)


def _plot_one_metric(df: pd.DataFrame, metric: str, out_path: Path, ylabel: str, cadence: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    algo_dfs = {a: _algo_curve(df, a, metric, cadence) for a in ("iql", "iql_kl_f")}
    x_max = 0
    for curve in algo_dfs.values():
        if not curve.empty:
            x_max = max(x_max, int(curve["step"].max()))

    for algo in ("iql", "iql_kl_f"):
        curve = algo_dfs[algo]
        if curve.empty:
            continue
        color = _ALGO_COLORS[algo]
        ax.plot(curve["step"], curve["mean"], color=color, linewidth=2,
                label=f"{algo.upper()} (mean ± std, n={int(curve['n'].iloc[0])} seeds)")
        ax.fill_between(curve["step"], curve["mean"] - curve["std"], curve["mean"] + curve["std"],
                        color=color, alpha=0.2)

    for ref in ("random", "noop"):
        band = _ref_band(df, ref, metric)
        if band is None:
            continue
        m, s = band
        color = _ALGO_COLORS[ref]
        xs = np.array([0, x_max if x_max > 0 else 1])
        ax.axhline(m, color=color, linestyle="--", linewidth=1.5, label=f"{ref}")
        ax.fill_between(xs, m - s, m + s, color=color, alpha=0.1)

    ax.set_xlabel("gradient steps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Part 3 — {ylabel} vs gradient-steps")
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
                   help="snap step values to nearest multiple of this for cross-seed grouping")
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
