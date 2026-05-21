"""Modal wrapper for Part 3 IQL variants.

Reuses the same image and `phase1-results` volume as Parts 1+2. The offline dataset
must already be on the volume at /results/offline_dataset.npz before running.

Operator commands:
    modal run phase3_iql/modal_app.py --preset smoke
    modal run phase3_iql/modal_app.py --preset standard
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
app = modal.App("clinical-triage-phase3", image=image)


@app.function(gpu="A100", volumes={"/results": volume}, timeout=60 * 60 * 3,
              max_containers=10)
def run_one(
    algo: str,
    seed: int,
    preset: str = "standard",
    branch: str = "saimai",
    dataset_path: str = "/results/offline_dataset.npz",
    tag: str = "reward0",
) -> str:
    """One Modal container = one (algo, seed) offline-IQL training run."""
    import subprocess

    subprocess.run(["git", "fetch", "origin"], cwd="/app", check=True)
    subprocess.run(["git", "checkout", "--detach", f"origin/{branch}"],
                   cwd="/app", check=True)

    args = [
        "python", "-m", "phase3_iql.train",
        "--algo", algo,
        "--seed", str(seed),
        "--preset", preset,
        "--out-root", "/results",
        "--tag", tag,
    ]

    if algo in ("iql", "iql_kl_f"):
        args += ["--dataset-path", dataset_path]

    subprocess.run(args, cwd="/app", check=True)
    volume.commit()
    return f"{algo}/seed_{seed} ({preset}, {tag}) done"


@app.local_entrypoint()
def main(
    preset: str = "standard",
    algos: str = "iql,iql_kl_f",
    branch: str = "saimai",
    dataset_path: str = "/results/offline_dataset.npz",
    tag: str = "reward0",
) -> None:
    from phase3_iql.presets import PRESETS

    algo_list = [a.strip() for a in algos.split(",") if a.strip()]
    if not algo_list:
        raise SystemExit("no algos specified")

    seeds = PRESETS[preset]["seeds"]
    jobs = [(a, s, preset, branch, dataset_path, tag) for a in algo_list for s in seeds]

    print(f"launching {len(jobs)} containers (max_containers=10): "
          f"{[(a, s, tag) for a, s, _, _, _, tag in jobs]}")

    results = list(run_one.starmap(jobs))

    for r in results:
        print(f"  {r}")

    print(f"all {len(jobs)} containers done; volume committed")
