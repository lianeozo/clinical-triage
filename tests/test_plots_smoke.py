"""Smoke test: plot scripts produce non-empty PNGs given the aggregated tables."""
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.aggregate import aggregate_run_dirs


@pytest.fixture
def aggregated_from_local_smoke(tmp_path):
    """Aggregate the existing local SMOKE run dirs into parquet files in tmp_path."""
    results_root = Path("results/phase1_ppo_dqn")
    lc, ad = aggregate_run_dirs(results_root, run_pattern="*-smoke-*")
    if lc.empty or ad.empty:
        pytest.skip("no local smoke run dirs found; run smoke first")
    out_dir = tmp_path / "aggregated"
    out_dir.mkdir()
    lc.to_parquet(out_dir / "learning_curves.parquet", index=False)
    ad.to_parquet(out_dir / "action_distributions.parquet", index=False)
    return out_dir


def test_plot_learning_curves_produces_four_pngs(tmp_path, aggregated_from_local_smoke):
    fig_dir = tmp_path / "figures"
    # Use sys.executable so the subprocess inherits pytest's python (already in the
    # clinical-triage env with PYTHONNOUSERSITE / PYTHONPATH set correctly by the caller).
    subprocess.run([
        sys.executable, "-m", "analysis.plot_learning_curves",
        "--aggregated", str(aggregated_from_local_smoke / "learning_curves.parquet"),
        "--out-dir", str(fig_dir),
    ], check=True)
    for name in ("learning_reward.png", "learning_mortality.png",
                 "learning_discharge.png", "learning_ep_length.png"):
        path = fig_dir / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0, f"empty {name}"
