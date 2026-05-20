"""Modal wrapper: one container per (algo, seed). Image bakes the initial clone +
deps; each container fetches the latest saimai commit at runtime.

Operator commands:
    modal run phase1_ppo_dqn/modal_app.py --preset smoke
    modal run phase1_ppo_dqn/modal_app.py --preset standard

After completion:
    modal volume get phase1-results / results/phase1_ppo_dqn/_modal_pull/
"""
from __future__ import annotations

import modal

# Image: Python 3.10 + git + project requirements + a fresh clone of the repo.
# Image rebuilds only when requirements.txt or the clone-line itself change.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands([
        "git clone https://github.com/lianeozo/clinical-triage.git /app",
    ])
)

# Persistent volume for results — survives container teardown, can be downloaded later.
volume = modal.Volume.from_name("phase1-results", create_if_missing=True)

app = modal.App("clinical-triage-phase1", image=image)


@app.function(gpu="T4", volumes={"/results": volume}, timeout=60 * 60)
def run_one(algo: str, seed: int, preset: str = "standard", branch: str = "saimai") -> str:
    """One Modal container = one (algo, seed) training run."""
    import subprocess
    # Fetch latest, detach HEAD directly at origin/<branch>. No local branch maintained.
    subprocess.run(["git", "fetch", "origin"], cwd="/app", check=True)
    subprocess.run(["git", "checkout", "--detach", f"origin/{branch}"], cwd="/app", check=True)
    subprocess.run([
        "python", "-m", "phase1_ppo_dqn.train",
        "--algo", algo, "--seed", str(seed),
        "--preset", preset, "--out-root", "/results",
    ], cwd="/app", check=True)
    volume.commit()
    return f"{algo}/seed_{seed} ({preset}) done"


@app.local_entrypoint()
def main(preset: str = "standard", algos: str = "dqn,ppo", branch: str = "saimai") -> None:
    """Fan out (#algos × #seeds) containers in parallel."""
    from phase1_ppo_dqn.presets import PRESETS
    algo_list = [a.strip() for a in algos.split(",") if a.strip()]
    if not algo_list:
        raise SystemExit("no algos specified")
    seeds = PRESETS[preset]["seeds"]
    jobs = [(a, s, preset, branch) for a in algo_list for s in seeds]
    print(f"launching {len(jobs)} containers: {[(a, s) for a, s, _, _ in jobs]}")
    results = list(run_one.starmap(jobs))
    for r in results:
        print(f"  {r}")
    print(f"all {len(jobs)} containers done; volume committed")
