#!/usr/bin/env python3
"""Extract IQL / IQL-KL-F reward-variant points for the CS224R poster.

For each (algo in {iql, iql_kl_f}, variant in {reward0..reward4}, seed in 0..4):
  * Pool across the fragmented timestamped run dirs under
    ``results/phase1_ppo_dqn/_modal_pull/2026-06-02T*-standard-{algo}-{variant}/seed_{seed}/``.
    A seed may appear in multiple timestamp dirs (a now()-per-container run-name
    artifact); pick the dir whose ``eval_checkpoints.jsonl`` reaches the highest
    step. Only keep seeds that reach step >= 200000.
  * Mortality: from ``eval_checkpoints.jsonl``, the record with step == 200000
    (the common comparison checkpoint). If no exact 200000 record exists, fall
    back to the record with the smallest step >= 200000. Uses ``mortality_rate``.
  * low_abnormal_icu_rate: from ``eval_trajectories/step_200000.jsonl`` using
    Yun's exact formula (see evaluation/aggregate_eval_trajectories.py L108-111):
        for each (a, s) in zip(num_abnormal_vitals, socs):
            if a == 0: total += 1; if s == 3: icu += 1
        rate = icu / total   (None if total == 0)

Aggregates over seeds -> mean/std per (algo, variant). Writes a tidy CSV and
prints the full table.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODAL_PULL = REPO / "results" / "phase1_ppo_dqn" / "_modal_pull"
OUT_CSV = REPO / "results" / "local_artifacts" / "iql_variant_points.csv"

ALGOS = ["iql", "iql_kl_f"]
VARIANTS = ["reward0", "reward1", "reward2", "reward3", "reward4"]
SEEDS = [0, 1, 2, 3, 4]
COMPARE_STEP = 200000


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _candidate_dirs(algo: str, variant: str, seed: int) -> list[Path]:
    """All seed dirs for (algo, variant, seed) that have an eval_checkpoints.jsonl.

    The glob ``*-standard-{algo}-{variant}`` would match iql_kl_f when algo=iql
    only if not anchored; we anchor on the exact suffix so iql does not absorb
    iql_kl_f runs (they share the ``iql`` prefix).
    """
    pat = str(MODAL_PULL / f"2026-06-02T*-standard-{algo}-{variant}" / f"seed_{seed}")
    dirs = []
    for d in glob.glob(pat):
        dp = Path(d)
        # Exact-suffix guard against algo prefix collisions (iql vs iql_kl_f).
        run_name = dp.parent.name
        if not run_name.endswith(f"-standard-{algo}-{variant}"):
            continue
        if (dp / "eval_checkpoints.jsonl").is_file():
            dirs.append(dp)
    return dirs


def _max_step(ck_path: Path) -> int:
    steps = [r.get("step", -1) for r in _read_jsonl(ck_path)]
    return max(steps) if steps else -1


def _pick_dir(algo: str, variant: str, seed: int) -> Path | None:
    """Pick the candidate dir whose eval_checkpoints reaches the highest step,
    among those reaching step >= 200000. Returns None if none qualify."""
    best = None
    best_max = -1
    for dp in _candidate_dirs(algo, variant, seed):
        m = _max_step(dp / "eval_checkpoints.jsonl")
        if m >= COMPARE_STEP and m > best_max:
            best_max = m
            best = dp
    return best


def _mortality(ck_path: Path) -> tuple[float, int]:
    """Return (mortality_rate, step_used). Prefers step==200000, else smallest
    step >= 200000."""
    recs = _read_jsonl(ck_path)
    exact = [r for r in recs if r.get("step") == COMPARE_STEP]
    if exact:
        return float(exact[0]["mortality_rate"]), COMPARE_STEP
    ge = sorted(
        (r for r in recs if r.get("step", -1) >= COMPARE_STEP),
        key=lambda r: r["step"],
    )
    if not ge:
        raise ValueError(f"no checkpoint >= {COMPARE_STEP} in {ck_path}")
    return float(ge[0]["mortality_rate"]), int(ge[0]["step"])


def _low_abn_icu_rate(traj_path: Path) -> float | None:
    """Yun's exact formula (aggregate_eval_trajectories.py L108-111)."""
    episodes = _read_jsonl(traj_path)
    low_abnormal_total = 0
    low_abnormal_icu = 0
    for ep in episodes:
        abn = ep.get("num_abnormal_vitals", [])
        socs = ep.get("socs", [])
        if isinstance(abn, list) and isinstance(socs, list):
            for a, s in zip(abn, socs):
                if a == 0:
                    low_abnormal_total += 1
                    if s == 3:
                        low_abnormal_icu += 1
    if low_abnormal_total == 0:
        return None
    return low_abnormal_icu / low_abnormal_total


def main() -> None:
    rows = []
    fallbacks = []  # (algo, variant, seed, step_used)

    for algo in ALGOS:
        for variant in VARIANTS:
            morts = []
            lowicus = []
            n_seeds = 0
            for seed in SEEDS:
                dp = _pick_dir(algo, variant, seed)
                if dp is None:
                    continue
                ck = dp / "eval_checkpoints.jsonl"
                traj = dp / "eval_trajectories" / f"step_{COMPARE_STEP}.jsonl"
                mort, step_used = _mortality(ck)
                if step_used != COMPARE_STEP:
                    fallbacks.append((algo, variant, seed, step_used, dp.parent.name))
                morts.append(mort)
                if traj.is_file():
                    li = _low_abn_icu_rate(traj)
                    if li is not None:
                        lowicus.append(li)
                n_seeds += 1

            def _mean(xs):
                return statistics.fmean(xs) if xs else float("nan")

            def _std(xs):
                return statistics.stdev(xs) if len(xs) > 1 else 0.0

            rows.append(
                {
                    "algo": algo,
                    "reward": variant,
                    "n_seeds": n_seeds,
                    "mortality_mean": _mean(morts),
                    "mortality_std": _std(morts),
                    "low_abn_icu_mean": _mean(lowicus),
                    "low_abn_icu_std": _std(lowicus),
                }
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "algo",
        "reward",
        "n_seeds",
        "mortality_mean",
        "mortality_std",
        "low_abn_icu_mean",
        "low_abn_icu_std",
    ]
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Pretty print.
    header = f"{'algo':<10} {'reward':<8} {'n':>2}  {'mort_mean':>9} {'mort_std':>9}  {'lowICU_mean':>11} {'lowICU_std':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['algo']:<10} {r['reward']:<8} {r['n_seeds']:>2}  "
            f"{r['mortality_mean']:>9.4f} {r['mortality_std']:>9.4f}  "
            f"{r['low_abn_icu_mean']:>11.4f} {r['low_abn_icu_std']:>10.4f}"
        )
    print(f"\nWrote {OUT_CSV}")
    if fallbacks:
        print("\nNon-200000 mortality fallbacks (algo, variant, seed, step_used, run_dir):")
        for fb in fallbacks:
            print(f"  {fb}")
    else:
        print("\nAll mortality records used exact step==200000.")


if __name__ == "__main__":
    main()
