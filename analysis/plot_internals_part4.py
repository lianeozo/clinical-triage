"""Phase 4 training-internals plot: 2x2 grid of MBPO/MCTS internals vs outer iter.

Reads ``training_internals.jsonl`` from each Phase 4 seed directory and
overlays per-condition mean ± std bands across seeds. Panels:

* ``model_nll_heldout`` — ensemble held-out NLL after each model-fit
* ``mcts_mean_depth``   — average rollout depth (None/0.0 for no_mcts ablation)
* ``policy_entropy``    — π entropy after π/V fit
* ``value_mean``        — V mean after π/V fit

Conditions: ``mbpo_mcts_main`` (teal), ``mbpo_mcts_no_mcts`` (olive). Panels
whose metric is entirely absent are skipped with a warning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_INTERNALS_METRICS = [
    ("model_nll_heldout", "Model NLL (held-out)"),
    ("mcts_mean_depth",   "MCTS mean rollout depth"),
    ("policy_entropy",    "Policy entropy"),
    ("value_mean",        "Value mean"),
]

# Map run-name → canonical condition tag. Phase 4 modal_app uses run_ids like
# "<timestamp>-standard-mbpo_mcts_main" / "...-mbpo_mcts_no_mcts".
_CONDITIONS = ["mbpo_mcts_main", "mbpo_mcts_no_mcts"]
_CONDITION_COLORS = {
    "mbpo_mcts_main":    "#17becf",  # teal
    "mbpo_mcts_no_mcts": "#bcbd22",  # olive
}


def _parse_seed_dir_name(name: str) -> int | None:
    if not name.startswith("seed_"):
        return None
    try:
        return int(name[len("seed_"):])
    except ValueError:
        return None


def _condition_for_run(run_name: str) -> str | None:
    """Return canonical condition tag for a run dir name, or None."""
    for cond in _CONDITIONS:
        if cond in run_name:
            return cond
    return None


def _load_internals_rows(results_root: Path, run_pattern: str) -> pd.DataFrame:
    """Walk results_root/<run-pattern>/seed_*/training_internals.jsonl and return a long-form DF."""
    rows: list[dict] = []
    for run_dir in sorted(results_root.glob(run_pattern)):
        if not run_dir.is_dir():
            continue
        cond = _condition_for_run(run_dir.name)
        if cond is None:
            continue
        for seed_dir in sorted(run_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            seed = _parse_seed_dir_name(seed_dir.name)
            if seed is None:
                continue
            ti_path = seed_dir / "training_internals.jsonl"
            if not ti_path.exists():
                continue
            with open(ti_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    outer_iter = rec.get("outer_iter")
                    if outer_iter is None:
                        # Phase 4 always logs outer_iter; older phases don't ⇒ skip.
                        continue
                    row = {
                        "condition": cond,
                        "run_id": run_dir.name,
                        "seed": seed,
                        "outer_iter": int(outer_iter),
                    }
                    for k, _label in _INTERNALS_METRICS:
                        v = rec.get(k)
                        row[k] = float(v) if v is not None else np.nan
                    rows.append(row)
    return pd.DataFrame(rows)


def _curve_for_metric(df: pd.DataFrame, condition: str, metric: str) -> pd.DataFrame:
    """Per-outer-iter mean±std across seeds for one (condition, metric).

    Rows where the metric is NaN are dropped *before* aggregation, so an
    ablation that never logs a metric yields an empty curve.
    """
    sub = df[df["condition"] == condition]
    if sub.empty:
        return pd.DataFrame(columns=["outer_iter", "mean", "std", "n"])
    sub = sub.dropna(subset=[metric])
    if sub.empty:
        return pd.DataFrame(columns=["outer_iter", "mean", "std", "n"])
    grouped = sub.groupby("outer_iter")[metric].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.rename(columns={"count": "n"})
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped.sort_values("outer_iter").reset_index(drop=True)


def _plot_internals(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for ax, (metric, label) in zip(axes_flat, _INTERNALS_METRICS):
        any_plotted = False
        for condition in _CONDITIONS:
            curve = _curve_for_metric(df, condition, metric)
            if curve.empty:
                continue
            any_plotted = True
            color = _CONDITION_COLORS[condition]
            n_seeds = int(curve["n"].iloc[0])
            ax.plot(curve["outer_iter"], curve["mean"], color=color, linewidth=2,
                    label=f"{condition} (n={n_seeds})")
            ax.fill_between(curve["outer_iter"], curve["mean"] - curve["std"],
                            curve["mean"] + curve["std"], color=color, alpha=0.2)
        ax.set_xlabel("outer iteration")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        if any_plotted:
            ax.legend(loc="best", fontsize=8)
        else:
            print(f"warning: metric {metric!r} not present for any condition — panel will be empty")
            ax.text(0.5, 0.5, f"(no data for {metric})",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray", fontsize=10)

    fig.suptitle("Part 4 — MBPO/MCTS training internals", y=1.00)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path,
                   default=Path("results/phase1_ppo_dqn/_modal_pull"))
    p.add_argument("--run-pattern", default="*-standard-mbpo_mcts*",
                   help="glob (relative to results-root) for Phase 4 run dirs")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/phase1_ppo_dqn/figures"))
    args = p.parse_args()

    df = _load_internals_rows(args.results_root, args.run_pattern)
    if df.empty:
        raise SystemExit(
            f"no training_internals.jsonl rows with outer_iter found under "
            f"{args.results_root}/{args.run_pattern}")

    out_path = args.out_dir / "internals_part4.png"
    _plot_internals(df, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
