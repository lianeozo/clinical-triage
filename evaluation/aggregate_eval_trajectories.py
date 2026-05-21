# evaluation/aggregate_eval_trajectory_diagnostics.py

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


RUNS = [
    {
        "run_name": "dqn_reward1",
        "algo": "dqn",
        "reward_version": "reward1",
        "path": "sample_runs/reward1/standard-dqn-reward1",
    },
    {
        "run_name": "ppo_reward1",
        "algo": "ppo",
        "reward_version": "reward1",
        "path": "sample_runs/reward1/standard-ppo-reward1",
    },
    {
        "run_name": "sac_reward1",
        "algo": "sac",
        "reward_version": "reward1",
        "path": "sample_runs/reward1/standard-sac-reward1",
    },
    {
        "run_name": "sac_kl_f_reward1",
        "algo": "sac_kl_f",
        "reward_version": "reward1",
        "path": "sample_runs/reward1/standard-sac_kl_f-reward1",
    },
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_seed(seed_dir: Path) -> int | None:
    if seed_dir.name.startswith("seed_"):
        try:
            return int(seed_dir.name.split("_")[-1])
        except ValueError:
            return None
    return None


def infer_step(path: Path) -> int | None:
    m = re.match(r"step_(\d+)\.jsonl$", path.name)
    if not m:
        return None
    return int(m.group(1))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def soc_icu_rate_for_episodes(episodes: list[dict[str, Any]]) -> float | None:
    socs = []
    for ep in episodes:
        ep_socs = ep.get("socs", [])
        if isinstance(ep_socs, list):
            socs.extend(ep_socs)

    if not socs:
        return None
    return socs.count(3) / len(socs)


def summarize_file(
    traj_path: Path,
    run: dict[str, str],
    seed: int | None,
    eval_idx: int,
) -> dict[str, Any]:
    episodes = read_jsonl(traj_path)
    step = infer_step(traj_path)

    all_abn = []
    final_abn = []

    high_abnormal_total = 0
    high_abnormal_icu = 0
    low_abnormal_total = 0
    low_abnormal_icu = 0

    death_eps = []
    discharge_eps = []

    for ep in episodes:
        abn = ep.get("num_abnormal_vitals", [])
        socs = ep.get("socs", [])

        if isinstance(abn, list):
            all_abn.extend(abn)
            if abn:
                final_abn.append(float(abn[-1]))

        if isinstance(abn, list) and isinstance(socs, list):
            for a, s in zip(abn, socs):
                if a >= 2:
                    high_abnormal_total += 1
                    if s == 3:
                        high_abnormal_icu += 1
                if a == 0:
                    low_abnormal_total += 1
                    if s == 3:
                        low_abnormal_icu += 1

        reason = ep.get("terminal_reason")
        if reason == "death":
            death_eps.append(ep)
        elif reason == "discharge":
            discharge_eps.append(ep)

    def ep_lengths(eps: list[dict[str, Any]]) -> list[float]:
        vals = []
        for ep in eps:
            x = ep.get("length")
            if isinstance(x, (int, float)):
                vals.append(float(x))
        return vals

    def ep_final_abn(eps: list[dict[str, Any]]) -> list[float]:
        vals = []
        for ep in eps:
            abn = ep.get("num_abnormal_vitals", [])
            if isinstance(abn, list) and abn:
                vals.append(float(abn[-1]))
        return vals

    return {
        "run_name": run["run_name"],
        "algo": run["algo"],
        "reward_version": run["reward_version"],
        "seed": seed,
        "eval_idx": eval_idx,
        "step": step,
        "source_file": str(traj_path),

        # new trajectory-level diagnostics only
        "avg_abnormal_vitals": mean_or_none([float(x) for x in all_abn]),
        "avg_final_abnormal_vitals": mean_or_none(final_abn),

        "high_abnormal_icu_rate": (
            high_abnormal_icu / high_abnormal_total
            if high_abnormal_total > 0 else None
        ),
        "low_abnormal_icu_rate": (
            low_abnormal_icu / low_abnormal_total
            if low_abnormal_total > 0 else None
        ),

        "death_avg_length": mean_or_none(ep_lengths(death_eps)),
        "discharge_avg_length": mean_or_none(ep_lengths(discharge_eps)),

        "death_avg_final_abnormal": mean_or_none(ep_final_abn(death_eps)),
        "discharge_avg_final_abnormal": mean_or_none(ep_final_abn(discharge_eps)),

        "death_icu_rate": soc_icu_rate_for_episodes(death_eps),
        "discharge_icu_rate": soc_icu_rate_for_episodes(discharge_eps),

        # optional counts for debugging / interpretation
        "n_episodes": len(episodes),
        "n_death_episodes": len(death_eps),
        "n_discharge_episodes": len(discharge_eps),
    }


def load_diagnostics() -> pd.DataFrame:
    rows = []

    for run in RUNS:
        run_path = Path(run["path"])
        if not run_path.exists():
            print(f"[WARN] Missing run path: {run_path}")
            continue

        seed_dirs = sorted([p for p in run_path.glob("seed_*") if p.is_dir()])
        if not seed_dirs:
            print(f"[WARN] No seed_* dirs under {run_path}")
            continue

        for seed_dir in seed_dirs:
            seed = infer_seed(seed_dir)
            traj_dir = seed_dir / "eval_trajectories"
            if not traj_dir.exists():
                print(f"[WARN] Missing eval_trajectories dir: {traj_dir}")
                continue

            traj_files = sorted(
                traj_dir.glob("step_*.jsonl"),
                key=lambda p: infer_step(p) if infer_step(p) is not None else -1,
            )

            for eval_idx, traj_path in enumerate(traj_files):
                rows.append(
                    summarize_file(
                        traj_path=traj_path,
                        run=run,
                        seed=seed,
                        eval_idx=eval_idx,
                    )
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No trajectory diagnostics loaded. Check paths.")
    return df


def make_final_by_seed(df: pd.DataFrame) -> pd.DataFrame:
    final_rows = []
    for _, group in df.groupby(["run_name", "algo", "reward_version", "seed"], dropna=False):
        group = group.sort_values(["eval_idx", "step"], na_position="last")
        final_rows.append(group.iloc[-1].to_dict())
    return pd.DataFrame(final_rows)


def make_final_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "avg_abnormal_vitals",
        "avg_final_abnormal_vitals",
        "high_abnormal_icu_rate",
        "low_abnormal_icu_rate",
        "death_avg_length",
        "discharge_avg_length",
        "death_avg_final_abnormal",
        "discharge_avg_final_abnormal",
        "death_icu_rate",
        "discharge_icu_rate",
    ]

    summary_rows = []
    for (algo, reward_version), group in final_df.groupby(["algo", "reward_version"], dropna=False):
        row = {
            "algo": algo,
            "reward_version": reward_version,
            "n_seeds": group["seed"].nunique(dropna=True),
        }
        if row["n_seeds"] == 0:
            row["n_seeds"] = len(group)

        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/eval_reward1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_diagnostics()
    df = df.sort_values(["reward_version", "algo", "seed", "eval_idx", "step"])

    timeseries_out = out_dir / "trajectory_diagnostic_timeseries.csv"
    df.to_csv(timeseries_out, index=False)

    final_by_seed = make_final_by_seed(df)
    final_by_seed_out = out_dir / "trajectory_diagnostic_final_by_seed.csv"
    final_by_seed.to_csv(final_by_seed_out, index=False)

    final_summary = make_final_summary(final_by_seed)
    final_summary_out = out_dir / "trajectory_diagnostic_final_summary.csv"
    final_summary.to_csv(final_summary_out, index=False)

    print(f"Wrote: {timeseries_out}")
    print(f"Wrote: {final_by_seed_out}")
    print(f"Wrote: {final_summary_out}")

    cols = [
        "algo",
        "reward_version",
        "n_seeds",
        "avg_abnormal_vitals_mean",
        "avg_final_abnormal_vitals_mean",
        "high_abnormal_icu_rate_mean",
        "low_abnormal_icu_rate_mean",
        "death_avg_length_mean",
        "discharge_avg_length_mean",
        "death_avg_final_abnormal_mean",
        "discharge_avg_final_abnormal_mean",
        "death_icu_rate_mean",
        "discharge_icu_rate_mean",
    ]
    print("\nTrajectory diagnostic final summary:")
    print(final_summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()