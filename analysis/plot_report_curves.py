"""Replot online learning curves (mortality + return vs env-steps) for the final report,
paper-grade: sized to the final print width, enlarged fonts, thick lines, std bands, and a
single shared legend placed BELOW the panels so it never overlaps the data.

Input : Liane's eval_timeseries_reward{V}.csv (+ baselines file for random/noop).
Output: a single PNG sized ~full text width.
"""
from __future__ import annotations

import argparse
import csv
import collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRETTY = {
    "heuristic": "Heuristic", "dqn": "DQN", "ddqn": "Double DQN", "ppo": "PPO",
    "factppo": "FactPPO", "sac": "SAC", "sac_kl_f": "SAC-KL-F", "sac_kl_ppo": "SAC-KL-PPO",
}
ORDER = ["sac_kl_f", "sac", "sac_kl_ppo", "ppo", "factppo", "dqn", "ddqn", "heuristic"]
COLORS = {
    "sac_kl_f": "#1f77b4", "sac": "#2ca02c", "sac_kl_ppo": "#9467bd", "ppo": "#ff7f0e",
    "factppo": "#d62728", "dqn": "#8c564b", "ddqn": "#17becf", "heuristic": "#7f7f7f",
}


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    # group by (algo, eval_idx): mean step + mean/std of metric across seeds
    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        st, mo, re_ = f(r["step"]), f(r["mortality_rate"]), f(r["avg_return"])
        if st is None or r.get("eval_idx", "") == "":
            continue
        key = (r["algo"], int(float(r["eval_idx"])))
        by["step"][key].append(st)
        if mo is not None:
            by["mortality"][key].append(mo)
        if re_ is not None:
            by["return"][key].append(re_)
    return by


def series(by, metric, algo):
    keys = sorted((k for k in by[metric] if k[0] == algo and by[metric][k]), key=lambda k: k[1])
    step = np.array([np.mean(by["step"][k]) for k in keys])
    mean = np.array([np.mean(by[metric][k]) for k in keys])
    std = np.array([np.std(by[metric][k]) for k in keys])
    return step, mean, std


def baseline_consts(path):
    """Return {name: (mortality, return)} for random/noop, averaged over their rows."""
    out = {}
    if not Path(path).exists():
        return out
    rows = list(csv.DictReader(open(path)))
    for name in ("random", "noop"):
        rs = [r for r in rows if r.get("algo") == name]
        if rs:
            m = np.mean([float(r["mortality_rate"]) for r in rs])
            ret = np.mean([float(r.get("reward_mean", r.get("avg_return", "nan"))) for r in rs])
            out[name] = (m, ret)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--baselines", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by = load(args.csv)
    base = baseline_consts(args.baselines) if args.baselines else {}

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    })
    fig, (axm, axr) = plt.subplots(1, 2, figsize=(7.2, 3.3))

    handles, labels = [], []
    for algo in ORDER:
        c = COLORS[algo]
        for ax, metric in ((axm, "mortality"), (axr, "return")):
            s, m, sd = series(by, metric, algo)
            line, = ax.plot(s, m, color=c, lw=1.9, label=PRETTY[algo])
            ax.fill_between(s, m - sd, m + sd, color=c, alpha=0.13, linewidth=0)
        handles.append(line); labels.append(PRETTY[algo])

    # reference baselines as flat dashed lines
    refstyle = {"random": ("#444444", (0, (5, 4))), "noop": ("#999999", (0, (1, 2)))}
    for name, (mort, ret) in base.items():
        col, dash = refstyle.get(name, ("black", "--"))
        axm.axhline(mort, color=col, linestyle=dash, lw=1.4)
        h = axm.axhline(np.nan, color=col, linestyle=dash, lw=1.4, label=name)
        handles.append(h); labels.append(name)

    axm.set_title("Mortality rate"); axm.set_xlabel("environment steps"); axm.set_ylabel("mortality rate")
    axr.set_title("Average return"); axr.set_xlabel("environment steps"); axr.set_ylabel("average return")
    for ax in (axm, axr):
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.margins(x=0.02)
    axm.set_ylim(0, 1.02)

    # single shared legend BELOW both panels -> never overlaps the data
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.4, handlelength=1.8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
