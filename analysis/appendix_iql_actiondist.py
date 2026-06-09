"""Per-reward-variant IQL action-distribution figures for the report appendix.

The reward-variant IQL runs are fragmented across timestamp dirs (a run-name artifact);
for each (variant, seed) we select the dir whose eval_checkpoints reaches the highest step
(>= 200k) and read its untrained (step 0) and trained (step 200000) action histograms,
average fractions across seeds, and plot in the same style as the main-body figure.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.plot_action_distrib import (
    _agg_seed_mean_fractions, _stacked_bars_for_step, _TREATMENT_LABELS,
)

_KEY = re.compile(r"soc_(\d+)_abn_(\d+)")
TRAINED_STEP = 200000


def _best_dir_per_seed(modal_root, variant):
    """Return {seed: (dir, max_step)} choosing the highest-step dir per seed (>=200k)."""
    best = {}
    pat = f"{modal_root}/2026-06-02T*-standard-iql-{variant}/seed_*/eval_checkpoints.jsonl"
    for ck in glob.glob(pat):
        seed = int(re.search(r"seed_(\d+)", ck).group(1))
        try:
            steps = [json.loads(l).get("step", 0) for l in open(ck) if l.strip()]
        except Exception:
            continue
        mx = max(steps) if steps else 0
        if mx >= 200000 and (seed not in best or mx > best[seed][1]):
            best[seed] = (Path(ck).parent, mx)
    return best


def _rows_for_record(rec, step, seed):
    out = []
    sxa = rec.get("action_hist_by_soc_x_abnormal", {})
    for key, counts in sxa.items():
        m = _KEY.match(key)
        if not m:
            continue
        soc, abn = int(m.group(1)), int(m.group(2))
        total = sum(counts)
        if total == 0:
            continue
        for a, c in enumerate(counts):
            if c:
                out.append({"algo": "iql", "step": step, "seed": seed,
                            "soc": soc, "num_abnormal": abn,
                            "action_idx": a, "fraction": c / total})
    return out


def build_df(modal_root, variant):
    rows = []
    for seed, (d, _) in _best_dir_per_seed(modal_root, variant).items():
        recs = [json.loads(l) for l in open(d / "eval_checkpoints.jsonl") if l.strip()]
        for step in (0, TRAINED_STEP):
            rec = next((r for r in recs if r.get("step") == step and r.get("algo") == "iql"), None)
            if rec:
                rows.extend(_rows_for_record(rec, step, seed))
    return pd.DataFrame(rows)


def plot_variant(df, variant, out_path):
    abns = [0, 1, 2]
    plt.rcParams.update({
        "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
    })
    fig, axes = plt.subplots(len(abns), 2, figsize=(7.2, 1.7 * len(abns)),
                             sharey=True, squeeze=False)
    for ri, abn in enumerate(abns):
        for ci, step in enumerate((0, TRAINED_STEP)):
            cell = _agg_seed_mean_fractions(df[df["num_abnormal"] == abn], "iql", step)
            phase = "untrained" if ci == 0 else "trained"
            _stacked_bars_for_step(axes[ri][ci], cell, f"{abn} abnormal vital(s) ({phase})")
            if ci == 0:
                axes[ri][ci].set_ylabel("action fraction")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.015), columnspacing=1.4, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modal-root", default="results/phase1_ppo_dqn/_modal_pull")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for v in ("reward1", "reward2", "reward3", "reward4"):
        df = build_df(args.modal_root, v)
        if df.empty:
            print(f"[warn] no data for {v}")
            continue
        plot_variant(df, v, out / f"action_distrib_iql_{v}.png")


if __name__ == "__main__":
    main()
