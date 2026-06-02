#!/usr/bin/env python3
"""Build a Delta-summary LaTeX table from the milestone CSVs.

For each reward variant (reward1..reward4), compute the change relative to the
reward0 baseline in two metrics, per algorithm:

    Delta-mort   = mortality_rate_mean(algo, rew)        - mortality_rate_mean(algo, reward0)
    Delta-lowICU = low_abnormal_icu_rate_mean(algo, rew) - low_abnormal_icu_rate_mean(algo, reward0)

Values in the CSVs are fractions in [0, 1]; Deltas are rendered in percentage
points (multiplied by 100) with one decimal and an explicit sign.

Emits a standalone booktabs ``tabular`` (no preamble) suitable for ``\\input``.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# Algorithm order + pretty names for the table rows.
ALGOS = ["dqn", "ppo", "sac", "sac_kl_f", "sac_kl_ppo"]
ALGO_PRETTY = {
    "dqn": "DQN",
    "ppo": "PPO",
    "sac": "SAC",
    "sac_kl_f": "SAC-KL-F",
    "sac_kl_ppo": "SAC-KL-PPO",
}

# Reward variants used as columns (reward0 is the baseline, omitted).
REWARD_COLS = ["reward1", "reward2", "reward3", "reward4"]
BASELINE = "reward0"
ALL_REWARDS = [BASELINE] + REWARD_COLS

# Short two/three-word descriptors for the header second line.
REWARD_DESC = {
    "reward1": r"$\downarrow$terminal",
    "reward2": r"$\uparrow$treatment",
    "reward3": r"ICU penalty",
    "reward4": r"$\uparrow$SOC cost",
}

EVAL_PREFIX = "eval_final_summary_"
TRAJ_PREFIX = "trajectory_diagnostic_final_summary_"

# IQL offline algos appended as extra rows, sourced from the Step-1 CSV
# (analysis/iql_variant_points.py) rather than the milestone CSVs.
IQL_ALGOS = ["iql", "iql_kl_f"]
IQL_ALGO_PRETTY = {"iql": "IQL", "iql_kl_f": "IQL-KL-F"}
IQL_CSV_DEFAULT = "results/local_artifacts/iql_variant_points.csv"


def _load_metric(csv_path: Path, value_col: str) -> dict[str, float]:
    """Return {algo: value} for ``value_col`` from a summary CSV."""
    out: dict[str, float] = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if value_col not in (reader.fieldnames or []):
            raise KeyError(f"{value_col!r} not in {csv_path} (cols={reader.fieldnames})")
        for row in reader:
            algo = row.get("algo", "").strip()
            if not algo:
                continue
            raw = row.get(value_col, "")
            try:
                out[algo] = float(raw)
            except (TypeError, ValueError):
                # leave missing -> handled as em-dash downstream
                continue
    return out


def load_all(csv_dir: Path) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Load mortality and low-ICU metrics keyed by reward variant then algo."""
    mort: dict[str, dict[str, float]] = {}
    lowicu: dict[str, dict[str, float]] = {}
    for rew in ALL_REWARDS:
        eval_path = csv_dir / f"{EVAL_PREFIX}{rew}.csv"
        traj_path = csv_dir / f"{TRAJ_PREFIX}{rew}.csv"
        mort[rew] = _load_metric(eval_path, "mortality_rate_mean")
        lowicu[rew] = _load_metric(traj_path, "low_abnormal_icu_rate_mean")
    return mort, lowicu


def load_iql(iql_csv: Path) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Load IQL mortality/low-ICU keyed by reward variant then algo.

    Mirrors the milestone structure: returns (mort, lowicu) where each is
    {reward: {algo: value}}, with values as fractions in [0, 1].
    """
    mort: dict[str, dict[str, float]] = {rew: {} for rew in ALL_REWARDS}
    lowicu: dict[str, dict[str, float]] = {rew: {} for rew in ALL_REWARDS}
    with iql_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            algo = row.get("algo", "").strip()
            rew = row.get("reward", "").strip()
            if algo not in IQL_ALGOS or rew not in ALL_REWARDS:
                continue
            try:
                mort[rew][algo] = float(row["mortality_mean"])
            except (TypeError, ValueError, KeyError):
                pass
            try:
                lowicu[rew][algo] = float(row["low_abn_icu_mean"])
            except (TypeError, ValueError, KeyError):
                pass
    return mort, lowicu


def _fmt_pp(delta: float) -> str:
    """Signed percentage-point string with one decimal; real minus via math mode."""
    pp = delta * 100.0
    # Round before sign decision to avoid "-0.0".
    rounded = round(pp, 1)
    if rounded == 0.0:
        rounded = 0.0  # normalize -0.0
    if rounded < 0:
        return f"$-${abs(rounded):.1f}"
    return f"$+${rounded:.1f}"


def _cell(mort_by_algo, lowicu_by_algo, base_mort, base_lowicu, algo) -> str:
    """Render one 'Delta-mort / Delta-lowICU' cell, or em-dash if data missing."""
    m = mort_by_algo.get(algo)
    li = lowicu_by_algo.get(algo)
    bm = base_mort.get(algo)
    bli = base_lowicu.get(algo)
    if m is None or bm is None:
        mort_str = "---"
    else:
        mort_str = _fmt_pp(m - bm)
    if li is None or bli is None:
        icu_str = "---"
    else:
        icu_str = _fmt_pp(li - bli)
    return f"{mort_str} / {icu_str}"


def build_table(mort, lowicu, iql_mort=None, iql_lowicu=None) -> str:
    base_mort = mort[BASELINE]
    base_lowicu = lowicu[BASELINE]
    iql_base_mort = iql_mort[BASELINE] if iql_mort else {}
    iql_base_lowicu = iql_lowicu[BASELINE] if iql_lowicu else {}

    lines = []
    lines.append(
        "% Delta-summary table: each cell is "
        "Delta-mortality / Delta-low-abnormal-ICU in percentage points,"
    )
    lines.append("% relative to the reward0 baseline. Generated by analysis/delta_table.py.")
    lines.append(r"\begin{tabular}{l cccc}")
    lines.append(r"\toprule")

    # Header: two-line column titles.
    top = [r"\textbf{Algorithm}"] + [
        rf"\textbf{{Reward{rew[-1]}}}" for rew in REWARD_COLS
    ]
    lines.append(" & ".join(top) + r" \\")
    sub = [""] + [
        rf"{{\footnotesize {REWARD_DESC[rew]}}}" for rew in REWARD_COLS
    ]
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")

    for algo in ALGOS:
        cells = [ALGO_PRETTY[algo]]
        for rew in REWARD_COLS:
            cells.append(
                _cell(mort[rew], lowicu[rew], base_mort, base_lowicu, algo)
            )
        lines.append(" & ".join(cells) + r" \\")

    # Append IQL offline rows (delta vs each IQL algo's own reward0 baseline).
    if iql_mort and iql_lowicu:
        lines.append(r"\midrule")
        for algo in IQL_ALGOS:
            cells = [IQL_ALGO_PRETTY[algo]]
            for rew in REWARD_COLS:
                cells.append(
                    _cell(iql_mort[rew], iql_lowicu[rew], iql_base_mort, iql_base_lowicu, algo)
                )
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument(
        "--iql-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / IQL_CSV_DEFAULT,
        help="Step-1 IQL variant-points CSV (set to '' to omit IQL rows).",
    )
    args = ap.parse_args()

    csv_dir = args.csv_dir
    out_path = Path(os.path.expanduser(args.out))

    mort, lowicu = load_all(csv_dir)
    iql_mort = iql_lowicu = None
    if args.iql_csv and str(args.iql_csv) and Path(args.iql_csv).is_file():
        iql_mort, iql_lowicu = load_iql(Path(args.iql_csv))
    table = build_table(mort, lowicu, iql_mort, iql_lowicu)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"Wrote {out_path}")
    print(table)


if __name__ == "__main__":
    main()
