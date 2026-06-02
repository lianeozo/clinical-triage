#!/usr/bin/env python3
"""Regenerate milestone Figure 1 with IQL / IQL-KL-F rows added.

Dual-panel dot plot:
  LEFT  = mortality rate
  RIGHT = low-abnormal ICU use rate
Y-axis = algorithm (rows); one colored point per reward variant with horizontal
+-1 std error bars.

Online-algo data: results/local_artifacts/milestone/csv/
  eval_final_summary_reward{N}.csv          -> mortality_rate_mean/_std per algo
  trajectory_diagnostic_final_summary_reward{N}.csv -> low_abnormal_icu_rate_mean/_std
IQL data: results/local_artifacts/iql_variant_points.csv (from Step 1).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
CSV_DIR = REPO / "results" / "local_artifacts" / "milestone" / "csv"
IQL_CSV = REPO / "results" / "local_artifacts" / "iql_variant_points.csv"
OUT_PATH = Path(
    os.path.expanduser("~/Documents/CS224R-Final-Project-Poster/figures/figure1_iql.png")
)

REWARDS = ["reward0", "reward1", "reward2", "reward3", "reward4"]
REWARD_COLORS = {
    "reward0": "#1f77b4",
    "reward1": "#ff7f0e",
    "reward2": "#2ca02c",
    "reward3": "#d62728",
    "reward4": "#9467bd",
}

# Row order top -> bottom. matplotlib y increases upward, so the first entry
# gets the largest y.
ROW_ORDER = ["sac_kl_ppo", "sac_kl_f", "sac", "ppo", "dqn", "iql", "iql_kl_f"]
ALGO_PRETTY = {
    "sac_kl_ppo": "SAC-KL-PPO",
    "sac_kl_f": "SAC-KL-F",
    "sac": "SAC",
    "ppo": "PPO",
    "dqn": "DQN",
    "iql": "IQL",
    "iql_kl_f": "IQL-KL-F",
}

ONLINE_ALGOS = {"dqn", "ppo", "sac", "sac_kl_f", "sac_kl_ppo"}


def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str):
    raw = row.get(key, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def load_online():
    """{algo: {reward: (mort_mean, mort_std, lowicu_mean, lowicu_std)}}."""
    data: dict[str, dict[str, tuple]] = {}
    for rew in REWARDS:
        eval_rows = _load_csv(CSV_DIR / f"eval_final_summary_{rew}.csv")
        traj_rows = _load_csv(CSV_DIR / f"trajectory_diagnostic_final_summary_{rew}.csv")
        mort = {r["algo"].strip(): (_f(r, "mortality_rate_mean"), _f(r, "mortality_rate_std")) for r in eval_rows}
        low = {r["algo"].strip(): (_f(r, "low_abnormal_icu_rate_mean"), _f(r, "low_abnormal_icu_rate_std")) for r in traj_rows}
        for algo in set(mort) | set(low):
            if algo not in ONLINE_ALGOS:
                continue
            mm, ms = mort.get(algo, (None, None))
            lm, ls = low.get(algo, (None, None))
            data.setdefault(algo, {})[rew] = (mm, ms, lm, ls)
    return data


def load_iql():
    data: dict[str, dict[str, tuple]] = {}
    for r in _load_csv(IQL_CSV):
        algo = r["algo"].strip()
        rew = r["reward"].strip()
        data.setdefault(algo, {})[rew] = (
            _f(r, "mortality_mean"),
            _f(r, "mortality_std"),
            _f(r, "low_abn_icu_mean"),
            _f(r, "low_abn_icu_std"),
        )
    return data


def main() -> None:
    data = load_online()
    for algo, d in load_iql().items():
        data[algo] = d

    rows = [a for a in ROW_ORDER if a in data]
    # y positions: top row -> highest y.
    y_for = {algo: len(rows) - 1 - i for i, algo in enumerate(rows)}
    # vertical offsets so the 5 reward points don't overlap within a row.
    offsets = {rew: (i - 2) * 0.14 for i, rew in enumerate(REWARDS)}

    fig, (ax_m, ax_l) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    def plot_panel(ax, mean_idx, std_idx, title, xlabel):
        for rew in REWARDS:
            color = REWARD_COLORS[rew]
            xs, ys, xerr = [], [], []
            for algo in rows:
                cell = data[algo].get(rew)
                if cell is None or cell[mean_idx] is None:
                    continue
                xs.append(cell[mean_idx])
                ys.append(y_for[algo] + offsets[rew])
                xerr.append(cell[std_idx] if cell[std_idx] is not None else 0.0)
            if xs:
                ax.errorbar(
                    xs, ys, xerr=xerr, fmt="o", color=color, ecolor=color,
                    elinewidth=1.2, capsize=2.5, markersize=6, label=rew,
                    linestyle="none",
                )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yticks([y_for[a] for a in rows])
        ax.set_yticklabels([ALGO_PRETTY[a] for a in rows])
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        # Light separator above the IQL block.
        if "iql" in y_for:
            sep = y_for["iql"] + 0.5
            ax.axhline(sep, color="0.7", linewidth=0.8, linestyle="--")

    plot_panel(ax_m, 0, 1, "Mortality rate", "mortality rate")
    plot_panel(ax_l, 2, 3, "Low-abnormal ICU use", "low-abnormal ICU rate")

    handles, labels = ax_m.get_legend_handles_labels()
    # order legend by reward index
    order = sorted(range(len(labels)), key=lambda i: REWARDS.index(labels[i]))
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.06))

    fig.suptitle("Final-checkpoint performance across algorithms and reward variants")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
