# phase3_iql/recompute_dataset_rewards.py
"""Write offline_dataset_reward{0..4}.npz by recomputing the reward column from
stored (obs, action, next_obs) under each variant. Verifies reward0 reproduces
the stored reward column (gate: the dataset was generated under the original reward).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sepsisSimDiabetes.reward_variants import compute_reward, REWARD_VARIANTS


def recompute(obs, action, next_obs, version):
    out = np.empty(len(action), dtype=np.float32)
    for i in range(len(action)):
        out[i] = compute_reward(obs[i], int(action[i]), next_obs[i], version)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/phase1_ppo_dqn/aggregated/offline_dataset.npz")
    ap.add_argument("--out-dir", default="results/phase1_ppo_dqn/aggregated")
    args = ap.parse_args()

    d = np.load(args.src)
    obs, action, next_obs = d["obs"], d["action"], d["next_obs"]
    stored = d["reward"]

    # GATE: reward0 recompute must match the stored reward column.
    r0 = recompute(obs, action, next_obs, "reward0")
    n_mismatch = int(np.sum(~np.isclose(r0, stored, atol=1e-3)))
    frac = n_mismatch / len(stored)
    print(f"[gate] reward0 vs stored: {n_mismatch}/{len(stored)} mismatches ({frac:.4%})")
    if frac > 0.001:
        # show a few examples to debug
        idx = np.where(~np.isclose(r0, stored, atol=1e-3))[0][:5]
        for i in idx:
            print(f"  row {i}: stored={stored[i]} recomputed={r0[i]} "
                  f"obs={obs[i]} act={action[i]} next={next_obs[i]}")
        raise SystemExit("reward0 recompute does not match stored rewards; investigate before proceeding")

    out_dir = Path(args.out_dir)
    for version in REWARD_VARIANTS:
        r = r0 if version == "reward0" else recompute(obs, action, next_obs, version)
        out_path = out_dir / f"offline_dataset_{version}.npz"
        np.savez_compressed(out_path, obs=obs, action=action, reward=r,
                            next_obs=next_obs, terminated=d["terminated"],
                            source_algo=d["source_algo"])
        print(f"wrote {out_path}  reward mean={r.mean():.1f}")


if __name__ == "__main__":
    main()
