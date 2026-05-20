"""Modal wrapper for Phase 2 SAC variants.

Reuses the same image and `phase1-results` volume as Phase 1.
concurrency_limit=10 respects the operator's 10-GPU account cap.

Operator commands:
    modal run phase2_sac/modal_app.py --preset smoke --ppo-run-path /results/<phase1-ppo-run-name>
    modal run phase2_sac/modal_app.py --preset standard --ppo-run-path /results/2026-05-20T01-26-standard-ppo

After completion:
    modal volume get phase1-results / results/phase1_ppo_dqn/_modal_pull/
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
app = modal.App("clinical-triage-phase2", image=image)


@app.function(gpu="T4", volumes={"/results": volume}, timeout=60 * 60 * 3,
              max_containers=10)
def run_one(algo: str, seed: int, preset: str = "standard",
            branch: str = "saimai", ppo_run_path: str = "") -> str:
    """One Modal container = one (algo, seed) training run."""
    import subprocess
    subprocess.run(["git", "fetch", "origin"], cwd="/app", check=True)
    subprocess.run(["git", "checkout", "--detach", f"origin/{branch}"],
                   cwd="/app", check=True)
    args = ["python", "-m", "phase2_sac.train",
            "--algo", algo, "--seed", str(seed),
            "--preset", preset, "--out-root", "/results"]
    if algo == "sac_kl_ppo":
        if not ppo_run_path:
            raise ValueError("sac_kl_ppo requires --ppo-run-path")
        args += ["--ppo-run-dir", ppo_run_path]
    subprocess.run(args, cwd="/app", check=True)
    volume.commit()
    return f"{algo}/seed_{seed} ({preset}) done"


@app.local_entrypoint()
def main(preset: str = "standard",
         algos: str = "sac,sac_kl_f,sac_kl_ppo",
         branch: str = "saimai",
         ppo_run_path: str = "/results/2026-05-20T01-26-standard-ppo") -> None:
    """Fan out (#algos x #seeds) containers; Modal serializes the excess past concurrency_limit=10."""
    from phase2_sac.presets import PRESETS
    algo_list = [a.strip() for a in algos.split(",") if a.strip()]
    if not algo_list:
        raise SystemExit("no algos specified")
    seeds = PRESETS[preset]["seeds"]
    jobs = [(a, s, preset, branch, ppo_run_path) for a in algo_list for s in seeds]
    print(f"launching {len(jobs)} containers (concurrency_limit=10): "
          f"{[(a, s) for a, s, _, _, _ in jobs]}")
    results = list(run_one.starmap(jobs))
    for r in results:
        print(f"  {r}")
    print(f"all {len(jobs)} containers done; volume committed")
