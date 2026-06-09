"""Paper-grade training-dynamics figures for the final report.

Two matching 2-panel figures (mortality + average return), each with a single shared legend
placed BELOW the panels (never overlapping the data), enlarged fonts sized for the final print
width, thick lines, and std bands. The x-axis is rescaled to units of 100k steps to avoid tick
overlap.

  --mode online  : reads Liane's eval_timeseries_reward0.csv (+ baselines) -> env-steps
  --mode offline : reads the aggregated learning_curves.parquet (iql/iql_kl_f) -> gradient steps
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

ONLINE_PRETTY = {
    "sac_kl_f": "SAC-KL-F", "sac": "SAC", "sac_kl_ppo": "SAC-KL-PPO", "ppo": "PPO",
    "factppo": "FactPPO", "dqn": "DQN", "ddqn": "Double DQN", "heuristic": "Heuristic",
}
ONLINE_ORDER = ["sac_kl_f", "sac", "sac_kl_ppo", "ppo", "factppo", "dqn", "ddqn", "heuristic"]
ONLINE_COLORS = {
    "sac_kl_f": "#1f77b4", "sac": "#2ca02c", "sac_kl_ppo": "#9467bd", "ppo": "#ff7f0e",
    "factppo": "#d62728", "dqn": "#8c564b", "ddqn": "#17becf", "heuristic": "#7f7f7f",
}
OFFLINE_PRETTY = {"iql": "IQL", "iql_kl_f": "IQL-KL-F"}
OFFLINE_ORDER = ["iql", "iql_kl_f"]
OFFLINE_COLORS = {"iql": "#8c564b", "iql_kl_f": "#e377c2"}

SCALE = 1e5  # x-axis in units of 100k steps


def _render(curves, refs, xlabel, out, ncol):
    """curves: list of dicts {label,color,x,mort_m,mort_s,ret_m,ret_s}.
    refs: list of (label, color, dash, mort_value)."""
    plt.rcParams.update({
        "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
    })
    fig, (axm, axr) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    handles, labels = [], []
    for c in curves:
        line, = axm.plot(c["x"], c["mort_m"], color=c["color"], lw=1.9)
        axm.fill_between(c["x"], c["mort_m"] - c["mort_s"], c["mort_m"] + c["mort_s"],
                         color=c["color"], alpha=0.13, linewidth=0)
        axr.plot(c["x"], c["ret_m"], color=c["color"], lw=1.9)
        axr.fill_between(c["x"], c["ret_m"] - c["ret_s"], c["ret_m"] + c["ret_s"],
                         color=c["color"], alpha=0.13, linewidth=0)
        handles.append(line); labels.append(c["label"])
    for name, col, dash, mort in refs:
        axm.axhline(mort, color=col, linestyle=dash, lw=1.4)
        h = axm.axhline(np.nan, color=col, linestyle=dash, lw=1.4)
        handles.append(h); labels.append(name)
    axm.set_xlabel(xlabel); axm.set_ylabel("mortality rate")
    axr.set_xlabel(xlabel); axr.set_ylabel("average return")
    for ax in (axm, axr):
        ax.grid(alpha=0.25, linewidth=0.5); ax.margins(x=0.02)
    axm.set_ylim(0, 1.02)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.4, handlelength=1.8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def online(csv_path, baselines, out):
    rows = list(csv.DictReader(open(csv_path)))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        st, mo, re_ = _f(r["step"]), _f(r["mortality_rate"]), _f(r["avg_return"])
        if st is None or r.get("eval_idx", "") == "":
            continue
        key = (r["algo"], int(float(r["eval_idx"])))
        by["step"][key].append(st)
        if mo is not None: by["mortality"][key].append(mo)
        if re_ is not None: by["return"][key].append(re_)

    def series(metric, algo):
        keys = sorted((k for k in by[metric] if k[0] == algo and by[metric][k]), key=lambda k: k[1])
        return (np.array([np.mean(by["step"][k]) for k in keys]) / SCALE,
                np.array([np.mean(by[metric][k]) for k in keys]),
                np.array([np.std(by[metric][k]) for k in keys]))

    curves = []
    for algo in ONLINE_ORDER:
        x, mm, ms = series("mortality", algo)
        _, rm, rs = series("return", algo)
        curves.append(dict(label=ONLINE_PRETTY[algo], color=ONLINE_COLORS[algo],
                           x=x, mort_m=mm, mort_s=ms, ret_m=rm, ret_s=rs))
    refs = []
    if baselines and Path(baselines).exists():
        brows = list(csv.DictReader(open(baselines)))
        style = {"random": ("#444444", (0, (5, 4))), "noop": ("#999999", (0, (1, 2)))}
        for name in ("random", "noop"):
            rs = [r for r in brows if r.get("algo") == name]
            if rs:
                refs.append((name, *style[name], float(np.mean([float(r["mortality_rate"]) for r in rs]))))
    _render(curves, refs, r"environment steps ($\times 100k$)", out, ncol=5)


def offline(parquet_path, out):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    curves = []
    for algo in OFFLINE_ORDER:
        sub = df[df.algo == algo]
        g = sub.groupby("step")
        x = np.array(sorted(sub.step.unique())) / SCALE
        mm = g["mortality_rate"].mean().reindex(sorted(sub.step.unique())).values
        ms = g["mortality_rate"].std().reindex(sorted(sub.step.unique())).values
        rm = g["reward_mean"].mean().reindex(sorted(sub.step.unique())).values
        rs = g["reward_mean"].std().reindex(sorted(sub.step.unique())).values
        curves.append(dict(label=OFFLINE_PRETTY[algo], color=OFFLINE_COLORS[algo],
                           x=x, mort_m=mm, mort_s=np.nan_to_num(ms),
                           ret_m=rm, ret_s=np.nan_to_num(rs)))
    refs = []
    for name, col, dash in (("random", "#444444", (0, (5, 4))), ("noop", "#999999", (0, (1, 2)))):
        ref = df[df.algo == name]
        if len(ref):
            refs.append((name, col, dash, float(ref["mortality_rate"].mean())))
    _render(curves, refs, r"gradient steps ($\times 100k$)", out, ncol=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["online", "offline"], required=True)
    ap.add_argument("--csv", default="")
    ap.add_argument("--baselines", default="")
    ap.add_argument("--parquet", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "online":
        online(args.csv, args.baselines, args.out)
    else:
        offline(args.parquet, args.out)


if __name__ == "__main__":
    main()
