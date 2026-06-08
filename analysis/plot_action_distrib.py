"""Render 2 action-distribution heatmap PNGs from action_distributions.parquet.

One PNG per trained algo (dqn, ppo). Each PNG is a grid:
- Rows: num_abnormal_vitals in {0..4}
- Columns: 2 columns: step=0 (or first step the algo evaluated at) and the final step
- Cells: grouped bar chart, x-axis = SOC (0..3), y-axis = action fraction (0..1).
  Each cell shows an 8-group stacked bar per SOC, with action groups colored by
  treatment combination (antib/vent/vaso).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Action packing (per sepsisSimDiabetes.Action): idx = 16*antib + 8*vent + 4*vaso + action_soc.
# action_idx ranges 0..31 independent of the state's soc (which is the conditioning variable).
# The 8 treatment combos are the top 3 bits of action_idx: action_idx // 4 = 4*antib + 2*vent + vaso.
_TREATMENT_LABELS = ["none", "vaso", "vent", "vent+vaso",
                     "antib", "antib+vaso", "antib+vent", "antib+vent+vaso"]
_TREATMENT_COLORS = plt.get_cmap("tab10").colors[:8]


def _action_to_treatment_group(action_idx: int) -> int:
    """Return the treatment group index (0..7) for a given action_idx."""
    treat = action_idx // 4
    if not (0 <= treat < 8):
        raise ValueError(f"action_idx {action_idx} → treat {treat}; expected 0..7")
    return treat


def _agg_seed_mean_fractions(df: pd.DataFrame, algo: str, step: int) -> pd.DataFrame:
    """For one (algo, step), aggregate fractions across seeds: mean fraction per (soc, num_abnormal, action_idx)."""
    sub = df[(df["algo"] == algo) & (df["step"] == step)]
    if sub.empty:
        return pd.DataFrame(columns=["soc", "num_abnormal", "action_idx", "fraction"])
    grouped = sub.groupby(["soc", "num_abnormal", "action_idx"])["fraction"].mean().reset_index()
    return grouped


def _stacked_bars_for_step(ax, grouped: pd.DataFrame, title: str) -> None:
    """Render the stacked-bar grid for one (algo, step) panel.

    x-axis: SOC (0..3). For each SOC, stack the 8 treatment groups (each group is
    the sum of fractions over all action_idx in that group at that SOC).
    """
    socs = [0, 1, 2, 3]
    # build matrix: rows = treatment group, cols = soc
    mat = np.zeros((8, len(socs)), dtype=float)
    for _, row in grouped.iterrows():
        soc = int(row["soc"])
        if soc not in socs:
            continue
        action_idx = int(row["action_idx"])
        treat = _action_to_treatment_group(action_idx)
        mat[treat, socs.index(soc)] += float(row["fraction"])
    bottoms = np.zeros(len(socs))
    for treat in range(8):
        ax.bar(socs, mat[treat], bottom=bottoms, color=_TREATMENT_COLORS[treat],
               label=_TREATMENT_LABELS[treat], width=0.8)
        bottoms = bottoms + mat[treat]
    ax.set_ylim(0, 1.05)
    ax.set_xticks(socs)
    ax.set_xticklabels(["ASYNC", "AMB", "FAC", "ICU"])
    ax.set_title(title, fontsize=10.5)


def _plot_one_algo(df: pd.DataFrame, algo: str, out_path: Path) -> None:
    sub = df[df["algo"] == algo]
    if sub.empty:
        print(f"no data for algo={algo}; skipping {out_path}")
        return
    steps = sorted(sub["step"].unique())
    first_step = steps[0]
    last_step = steps[-1]
    rows = sorted(sub["num_abnormal"].unique())
    n_rows = len(rows)

    fig, axes = plt.subplots(n_rows, 2, figsize=(9, 2.0 * n_rows), sharey=True, squeeze=False)
    for ri, num_abn in enumerate(rows):
        for ci, step in enumerate([first_step, last_step]):
            cell = _agg_seed_mean_fractions(
                df[df["num_abnormal"] == num_abn], algo, step)
            ax = axes[ri][ci]
            phase = "untrained" if ci == 0 else "trained"
            title = f"{num_abn} abnormal vital(s) --- {phase}"
            _stacked_bars_for_step(ax, cell, title)
            if ci == 0:
                ax.set_ylabel("action fraction")
    # Legend on the top-right panel.
    axes[0][-1].legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.42, 1.0))
    fig.suptitle(f"{algo.upper()} treatment-action distribution by site of care (within each panel)", y=1.00)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aggregated", type=Path, required=True,
                   help="path to action_distributions.parquet")
    p.add_argument("--out-dir", type=Path, default=Path("results/phase1_ppo_dqn/figures"))
    args = p.parse_args()

    df = pd.read_parquet(args.aggregated)
    if df.empty:
        raise SystemExit(f"no rows in {args.aggregated}")

    for algo in ("dqn", "ppo", "sac", "sac_kl_f", "sac_kl_ppo", "iql", "iql_kl_f",
                 "mbpo_mcts_main", "mbpo_mcts_no_mcts"):
        out_path = args.out_dir / f"action_distrib_{algo}.png"
        _plot_one_algo(df, algo, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
