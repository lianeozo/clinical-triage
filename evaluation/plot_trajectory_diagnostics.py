# evaluation/plot_trajectory_diagnostics.py

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_SPECS = {
    "avg_abnormal_vitals": {
        "title": "Average Abnormal Vitals vs env-steps",
        "ylabel": "Avg Abnormal Vitals",
        "filename": "avg_abnormal_vitals.png",
    },
    "avg_final_abnormal_vitals": {
        "title": "Average Final Abnormal Vitals vs env-steps",
        "ylabel": "Avg Final Abnormal Vitals",
        "filename": "avg_final_abnormal_vitals.png",
    },
    "high_abnormal_icu_rate": {
        "title": "ICU Rate When Abnormal Vitals >= 2 vs env-steps",
        "ylabel": "High-Abnormal ICU Rate",
        "filename": "high_abnormal_icu_rate.png",
    },
    "low_abnormal_icu_rate": {
        "title": "ICU Rate When Abnormal Vitals == 0 vs env-steps",
        "ylabel": "Low-Abnormal ICU Rate",
        "filename": "low_abnormal_icu_rate.png",
    },
    "death_avg_length": {
        "title": "Average Length of Death Episodes vs env-steps",
        "ylabel": "Death Avg Length",
        "filename": "death_avg_length.png",
    },
    "discharge_avg_length": {
        "title": "Average Length of Discharge Episodes vs env-steps",
        "ylabel": "Discharge Avg Length",
        "filename": "discharge_avg_length.png",
    },
    "death_avg_final_abnormal": {
        "title": "Final Abnormal Vitals in Death Episodes vs env-steps",
        "ylabel": "Death Avg Final Abnormal",
        "filename": "death_avg_final_abnormal.png",
    },
    "discharge_avg_final_abnormal": {
        "title": "Final Abnormal Vitals in Discharge Episodes vs env-steps",
        "ylabel": "Discharge Avg Final Abnormal",
        "filename": "discharge_avg_final_abnormal.png",
    },
    "death_icu_rate": {
        "title": "ICU Dwell Fraction in Death Episodes vs env-steps",
        "ylabel": "Death ICU Rate",
        "filename": "death_icu_rate.png",
    },
    "discharge_icu_rate": {
        "title": "ICU Dwell Fraction in Discharge Episodes vs env-steps",
        "ylabel": "Discharge ICU Rate",
        "filename": "discharge_icu_rate.png",
    },
}


def plot_metric(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    spec = METRIC_SPECS[metric]

    plt.figure(figsize=(11, 6))

    for algo, group in df.groupby("algo"):
        summary = (
            group.groupby("eval_idx")
            .agg(
                step_mean=("step", "mean"),
                mean=(metric, "mean"),
                std=(metric, "std"),
                count=(metric, "count"),
            )
            .reset_index()
            .sort_values("eval_idx")
        )

        summary = summary.dropna(subset=["mean"])
        if summary.empty:
            continue

        x = summary["step_mean"].to_numpy()
        mean = summary["mean"].to_numpy()
        std = summary["std"].fillna(0.0).to_numpy()

        n_seeds = group["seed"].nunique()
        label = f"{algo.upper()} (mean ± std, n={n_seeds} seeds)"

        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.18)

    plt.title(spec["title"])
    plt.xlabel("env steps")
    plt.ylabel(spec["ylabel"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / spec["filename"]
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="outputs/eval_reward1/trajectory_diagnostic_timeseries.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/eval_reward1/trajectory_figures",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRIC_SPECS:
        if metric in df.columns:
            plot_metric(df, metric, out_dir)
        else:
            print(f"[WARN] Missing metric column: {metric}")


if __name__ == "__main__":
    main()