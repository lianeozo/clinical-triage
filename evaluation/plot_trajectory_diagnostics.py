# evaluation/plot_trajectory_diagnostics.py

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REWARD_VERSION = "reward3"  # change this to reward0 / reward1 / reward2 / reward3


METRIC_SPECS = {
    "avg_abnormal_vitals": {
        "title": f"Average Abnormal Vitals vs env-steps ({REWARD_VERSION})",
        "ylabel": "Avg Abnormal Vitals",
        "filename": f"avg_abnormal_vitals_{REWARD_VERSION}.png",
    },
    "avg_final_abnormal_vitals": {
        "title": f"Average Final Abnormal Vitals vs env-steps ({REWARD_VERSION})",
        "ylabel": "Avg Final Abnormal Vitals",
        "filename": f"avg_final_abnormal_vitals_{REWARD_VERSION}.png",
    },
    "high_abnormal_icu_rate": {
        "title": f"ICU Rate When Abnormal Vitals >= 2 vs env-steps ({REWARD_VERSION})",
        "ylabel": "High-Abnormal ICU Rate",
        "filename": f"high_abnormal_icu_rate_{REWARD_VERSION}.png",
    },
    "low_abnormal_icu_rate": {
        "title": f"ICU Rate When Abnormal Vitals == 0 vs env-steps ({REWARD_VERSION})",
        "ylabel": "Low-Abnormal ICU Rate",
        "filename": f"low_abnormal_icu_rate_{REWARD_VERSION}.png",
    },
    "death_avg_length": {
        "title": f"Average Length of Death Episodes vs env-steps ({REWARD_VERSION})",
        "ylabel": "Death Avg Length",
        "filename": f"death_avg_length_{REWARD_VERSION}.png",
    },
    "discharge_avg_length": {
        "title": f"Average Length of Discharge Episodes vs env-steps ({REWARD_VERSION})",
        "ylabel": "Discharge Avg Length",
        "filename": f"discharge_avg_length_{REWARD_VERSION}.png",
    },
    "death_avg_final_abnormal": {
        "title": f"Final Abnormal Vitals in Death Episodes vs env-steps ({REWARD_VERSION})",
        "ylabel": "Death Avg Final Abnormal",
        "filename": f"death_avg_final_abnormal_{REWARD_VERSION}.png",
    },
    "discharge_avg_final_abnormal": {
        "title": f"Final Abnormal Vitals in Discharge Episodes vs env-steps ({REWARD_VERSION})",
        "ylabel": "Discharge Avg Final Abnormal",
        "filename": f"discharge_avg_final_abnormal_{REWARD_VERSION}.png",
    },
    "death_icu_rate": {
        "title": f"ICU Dwell Fraction in Death Episodes vs env-steps ({REWARD_VERSION})",
        "ylabel": "Death ICU Rate",
        "filename": f"death_icu_rate_{REWARD_VERSION}.png",
    },
    "discharge_icu_rate": {
        "title": f"ICU Dwell Fraction in Discharge Episodes vs env-steps ({REWARD_VERSION})",
        "ylabel": "Discharge ICU Rate",
        "filename": f"discharge_icu_rate_{REWARD_VERSION}.png",
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
        default=f"outputs/eval_{REWARD_VERSION}/trajectory_diagnostic_timeseries_{REWARD_VERSION}.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=f"outputs/eval_{REWARD_VERSION}/trajectory_figures",
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