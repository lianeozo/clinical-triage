"""Modal wrapper for Part 4 MBPO-MCTS variants.

Reuses the same image and `phase1-results` volume as Parts 1-3. The offline dataset
must already be on the volume at /results/offline_dataset.npz before running.

Operator commands:
    modal run phase4_mbpo_mcts/modal_app.py --preset smoke
    modal run phase4_mbpo_mcts/modal_app.py --preset standard
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands([
        "git clone https://github.com/lianeozo/clinical-triage.git /app",
    ])
)

volume = modal.Volume.from_name("phase1-results", create_if_missing=True)
app = modal.App("clinical-triage-phase4", image=image)


@app.function(gpu="A100", volumes={"/results": volume}, timeout=60 * 60 * 3,
              max_containers=10)
def run_one(condition: str, seed: int, preset: str = "standard",
            branch: str = "saimai",
            dataset_path: str = "/results/offline_dataset.npz") -> str:
    """One Modal container = one (condition, seed) MBPO-MCTS training run."""
    import subprocess
    subprocess.run(["git", "fetch", "origin"], cwd="/app", check=True)
    subprocess.run(["git", "checkout", "--detach", f"origin/{branch}"],
                   cwd="/app", check=True)
    args = ["python", "-m", "phase4_mbpo_mcts.train",
            "--condition", condition, "--seed", str(seed),
            "--preset", preset,
            "--dataset-path", dataset_path,
            "--out-root", "/results"]
    subprocess.run(args, cwd="/app", check=True)
    volume.commit()
    return f"{condition}/seed_{seed} ({preset}) done"


@app.local_entrypoint()
def main(preset: str = "standard",
         conditions: str = "main,no_mcts",
         branch: str = "saimai",
         dataset_path: str = "/results/offline_dataset.npz") -> None:
    from phase4_mbpo_mcts.presets import PRESETS
    cond_list = [c.strip() for c in conditions.split(",") if c.strip()]
    if not cond_list:
        raise SystemExit("no conditions specified")
    seeds = PRESETS[preset]["seeds"]
    jobs = [(c, s, preset, branch, dataset_path) for c in cond_list for s in seeds]
    print(f"launching {len(jobs)} containers (max_containers=10): "
          f"{[(c, s) for c, s, _, _, _ in jobs]}")
    results = list(run_one.starmap(jobs))
    for r in results:
        print(f"  {r}")
    print(f"all {len(jobs)} containers done; volume committed")
