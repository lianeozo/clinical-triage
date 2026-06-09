"""Rebuild offline IQL datasets, one per reward variant, from eval-trajectory logs.

Why not recompute on offline_dataset.npz? That npz stores MASKED observations but
rewards were computed on the TRUE simulator state, so rewards can't be re-derived
from it. The trajectory JSONLs, however, log the TRUE per-step `num_abnormal_vitals`
(post-step) and `socs` (post-step) alongside `states` (pre-step, masked),
`executed_actions`, `rewards_raw`, and `terminal_reason` — enough to recompute any
variant's reward exactly.

Alignment (from triage_rl/evaluator.py): at step t,
  states[t]               = s_t                (pre-step, masked observation)
  executed_actions[t]     = a_t                (post-clamp)
  rewards_raw[t]          = r_t for s_t -> s_{t+1}
  socs[t]                 = soc(s_{t+1})        (post-step)
  num_abnormal_vitals[t]  = true #abnormal(s_{t+1})  (post-step)
So the transition t reward uses prev-count = nav[t-1], next-count = nav[t],
prev-soc = socs[t-1], next-soc = socs[t]. The initial state's true count is NOT
logged, so transition t=0 is dropped (~1.9% of steps).

IQL input features stay the masked observations (obs=states[t], next_obs=states[t+1]),
exactly as the original dataset; only the reward labels change per variant.
Random/noop supplement is omitted (variant-consistent, trajectory-sourced only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sepsisSimDiabetes.reward_variants import REWARD_VARIANTS, compute_reward_from_counts

_TERMINAL_REASONS = {"discharge", "death", "timeout"}

# The five online behavior policies that constitute the offline dataset (matches the
# original Phase 3 composition). Excludes stale pre-rename duplicates (qac*), offline
# algos (iql*), model-based (mbpo*), and references (random/noop).
_BEHAVIOR_ALGOS = {"dqn", "ppo", "sac", "sac_kl_f", "sac_kl_ppo"}


def _episode_rows(rec: dict):
    """Yield per-transition dicts for t=1..T-1 (drop t=0; initial true count unknown).

    Each row carries obs/action/next_obs/terminated (variant-independent) plus the
    counts/socs/terminal_reason needed to compute any variant's reward, and the
    stored rewards_raw[t] for the reward0 verification gate.
    """
    states = rec["states"]
    execu = rec["executed_actions"]
    socs = rec["socs"]
    nav = rec["num_abnormal_vitals"]
    rr = rec["rewards_raw"]
    T = rec["length"]
    terminal_reason = rec.get("terminal_reason", "none")

    for t in range(1, T):
        obs = np.asarray(states[t], dtype=np.float32)
        next_obs = np.asarray(states[t + 1], dtype=np.float32) if t < T - 1 else obs
        is_last = (t == T - 1)
        terminated = is_last and (terminal_reason in _TERMINAL_REASONS)
        yield {
            "obs": obs,
            "action": int(execu[t]),
            "next_obs": next_obs,
            "terminated": bool(terminated),
            "n_prev": int(nav[t - 1]),
            "n_next": int(nav[t]),
            "prev_soc": int(socs[t - 1]),
            "next_soc": int(socs[t]),
            "term_reason": terminal_reason if is_last else None,
            "stored_reward": float(rr[t]),
        }


def build(modal_pull_root: Path, run_pattern: str = "*-standard-*"):
    rows = []
    for run_dir in sorted(Path(modal_pull_root).glob(run_pattern)):
        if not run_dir.is_dir():
            continue
        for seed_dir in sorted(run_dir.iterdir()):
            traj_dir = seed_dir / "eval_trajectories"
            if not traj_dir.is_dir():
                continue
            for traj_file in sorted(traj_dir.glob("step_*.jsonl")):
                for line in open(traj_file):
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    algo = rec.get("algo", "unknown")
                    if algo not in _BEHAVIOR_ALGOS:
                        continue
                    for row in _episode_rows(rec):
                        row["source_algo"] = algo
                        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modal-pull-root", default="results/phase1_ppo_dqn/_modal_pull")
    ap.add_argument("--run-pattern", default="*-standard-*")
    ap.add_argument("--out-dir", default="results/phase1_ppo_dqn/aggregated")
    args = ap.parse_args()

    rows = build(args.modal_pull_root, args.run_pattern)
    n = len(rows)
    if n == 0:
        raise SystemExit("no transitions built; check modal-pull-root / run-pattern")
    print(f"[build] {n} transitions from trajectory logs")

    obs = np.stack([r["obs"] for r in rows]).astype(np.float32)
    action = np.array([r["action"] for r in rows], dtype=np.int64)
    next_obs = np.stack([r["next_obs"] for r in rows]).astype(np.float32)
    terminated = np.array([r["terminated"] for r in rows], dtype=np.bool_)
    source_algo = np.array([r["source_algo"] for r in rows], dtype="<U20")

    # GATE: reward0 reconstruction must match stored rewards_raw.
    r0 = np.array([
        compute_reward_from_counts(r["n_prev"], r["n_next"], r["prev_soc"], r["next_soc"],
                                   r["action"], "reward0", r["term_reason"])
        for r in rows
    ], dtype=np.float32)
    stored = np.array([r["stored_reward"] for r in rows], dtype=np.float32)
    n_mismatch = int(np.sum(~np.isclose(r0, stored, atol=1e-3)))
    frac = n_mismatch / n
    print(f"[gate] reward0 vs stored rewards_raw: {n_mismatch}/{n} mismatches ({frac:.4%})")
    if frac > 0.001:
        idx = np.where(~np.isclose(r0, stored, atol=1e-3))[0][:8]
        for i in idx:
            r = rows[i]
            print(f"  mismatch: stored={stored[i]} recon={r0[i]} "
                  f"n_prev={r['n_prev']} n_next={r['n_next']} "
                  f"soc {r['prev_soc']}->{r['next_soc']} act={r['action']} "
                  f"term={r['term_reason']} algo={r['source_algo']}")
        raise SystemExit("reward0 reconstruction failed; alignment wrong — do not proceed")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for version in REWARD_VARIANTS:
        if version == "reward0":
            reward = r0
        else:
            reward = np.array([
                compute_reward_from_counts(r["n_prev"], r["n_next"], r["prev_soc"],
                                           r["next_soc"], r["action"], version, r["term_reason"])
                for r in rows
            ], dtype=np.float32)
        out_path = out_dir / f"offline_dataset_{version}.npz"
        np.savez_compressed(out_path, obs=obs, action=action, reward=reward,
                            next_obs=next_obs, terminated=terminated, source_algo=source_algo)
        print(f"  wrote {out_path}  reward mean={reward.mean():.1f}  "
              f"min={reward.min():.0f} max={reward.max():.0f}")


if __name__ == "__main__":
    main()
