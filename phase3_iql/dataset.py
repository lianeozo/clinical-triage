"""Assemble an offline dataset from Parts 1+2 eval trajectory JSONLs + a Random-policy supplement.

Output: dict of numpy arrays (and saved as .npz) with keys:
  obs (N, 8) f32, action (N,) i64, reward (N,) f32 RAW, next_obs (N, 8) f32,
  terminated (N,) bool, source_algo (N,) str.

Logic per episode:
  obs_t      = states[t]
  action_t   = executed_actions[t]   # post-F4a-clamp
  reward_t   = rewards_raw[t]        # raw; scaled inside agent
  next_obs_t = states[t+1] if t < T-1 else obs_t
  terminated_t = (t == T - 1) and (terminal_reason in {"discharge", "death", "timeout"})

Timeout-ending episodes treated as terminated=True too -- standard offline-RL handling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.agents.random import RandomAgent
from triage_rl.env import Env

_TERMINAL_REASONS = {"discharge", "death", "timeout"}


def _trajectory_to_transitions(rec: dict) -> list[dict]:
    """Convert one episode trajectory record into a list of transition dicts."""
    states = [np.asarray(s, dtype=np.float32) for s in rec["states"]]
    executed = rec["executed_actions"]
    rewards = rec["rewards_raw"]
    T = rec["length"]
    terminal_reason = rec.get("terminal_reason", "none")
    assert len(states) >= T, f"trajectory has {len(states)} states but length={T}"

    rows = []
    for t in range(T):
        obs_t = states[t]
        next_obs_t = states[t + 1] if t < T - 1 else obs_t
        terminated_t = (t == T - 1) and (terminal_reason in _TERMINAL_REASONS)
        rows.append({
            "obs":        obs_t,
            "action":     int(executed[t]),
            "reward":     float(rewards[t]),
            "next_obs":   next_obs_t,
            "terminated": bool(terminated_t),
        })
    return rows


def _generate_random_transitions(n: int, seed: int = 0) -> dict:
    """Run RandomAgent in Env locally to produce n transitions for state-coverage supplement."""
    rng_env = np.random.default_rng(seed)
    agent = RandomAgent(seed=seed)
    env = Env(p_diabetes=0.2, max_steps=100)

    obs_list, action_list, reward_list, next_obs_list, term_list = [], [], [], [], []
    collected = 0
    while collected < n:
        obs, _ = env.reset(seed=int(rng_env.integers(0, 2**31 - 1)))
        done = False
        while not done and collected < n:
            a = agent.act(obs, eval_mode=False)
            next_obs, raw_reward, terminated, truncated, info = env.step(a)
            obs_list.append(obs.astype(np.float32))
            action_list.append(int(info["executed_action"]))
            reward_list.append(float(raw_reward))
            next_obs_list.append(next_obs.astype(np.float32))
            term_list.append(bool(terminated or truncated))
            collected += 1
            done = terminated or truncated
            obs = next_obs

    return {
        "obs":         np.stack(obs_list)[:n],
        "action":      np.array(action_list[:n], dtype=np.int64),
        "reward":      np.array(reward_list[:n], dtype=np.float32),
        "next_obs":    np.stack(next_obs_list)[:n],
        "terminated":  np.array(term_list[:n], dtype=np.bool_),
        "source_algo": np.array(["random"] * n, dtype="<U20"),
    }


def assemble_offline_dataset(modal_pull_root: Path, run_pattern: str = "*-standard-*",
                              n_random_transitions: int = 25_000,
                              random_seed: int = 0) -> dict:
    """Scan eval trajectories under modal_pull_root and build the offline dataset."""
    modal_pull_root = Path(modal_pull_root)

    obs_chunks, action_chunks, reward_chunks = [], [], []
    next_obs_chunks, term_chunks, source_chunks = [], [], []

    for run_dir in sorted(modal_pull_root.glob(run_pattern)):
        if not run_dir.is_dir():
            continue
        for seed_dir in sorted(run_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            traj_dir = seed_dir / "eval_trajectories"
            if not traj_dir.exists():
                continue
            for traj_file in sorted(traj_dir.glob("step_*.jsonl")):
                with open(traj_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        algo = rec.get("algo", "unknown")
                        if algo in ("random", "noop"):
                            continue  # references; supplement separately
                        rows = _trajectory_to_transitions(rec)
                        for row in rows:
                            obs_chunks.append(row["obs"])
                            action_chunks.append(row["action"])
                            reward_chunks.append(row["reward"])
                            next_obs_chunks.append(row["next_obs"])
                            term_chunks.append(row["terminated"])
                            source_chunks.append(algo)

    rand = _generate_random_transitions(n_random_transitions, seed=random_seed)

    obs_arr = np.stack(obs_chunks) if obs_chunks else np.zeros((0, State.NUM_STATE_VARS), dtype=np.float32)
    action_arr = np.array(action_chunks, dtype=np.int64) if action_chunks else np.zeros(0, dtype=np.int64)
    reward_arr = np.array(reward_chunks, dtype=np.float32) if reward_chunks else np.zeros(0, dtype=np.float32)
    next_obs_arr = np.stack(next_obs_chunks) if next_obs_chunks else np.zeros((0, State.NUM_STATE_VARS), dtype=np.float32)
    term_arr = np.array(term_chunks, dtype=np.bool_) if term_chunks else np.zeros(0, dtype=np.bool_)
    source_arr = np.array(source_chunks, dtype="<U20") if source_chunks else np.zeros(0, dtype="<U20")

    return {
        "obs":         np.concatenate([obs_arr, rand["obs"]], axis=0),
        "action":      np.concatenate([action_arr, rand["action"]], axis=0),
        "reward":      np.concatenate([reward_arr, rand["reward"]], axis=0),
        "next_obs":    np.concatenate([next_obs_arr, rand["next_obs"]], axis=0),
        "terminated":  np.concatenate([term_arr, rand["terminated"]], axis=0),
        "source_algo": np.concatenate([source_arr, rand["source_algo"]], axis=0),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--modal-pull-root", type=Path,
                   default=Path("results/phase1_ppo_dqn/_modal_pull"))
    p.add_argument("--run-pattern", default="*-standard-*")
    p.add_argument("--n-random-transitions", type=int, default=25_000)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--out-file", type=Path,
                   default=Path("results/phase1_ppo_dqn/aggregated/offline_dataset.npz"))
    args = p.parse_args()

    data = assemble_offline_dataset(args.modal_pull_root, args.run_pattern,
                                    args.n_random_transitions, args.random_seed)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_file, **data)

    n = len(data['action'])
    print(f"offline_dataset: {n} transitions -> {args.out_file}")
    src = data['source_algo']
    for algo in sorted(set(src.tolist())):
        c = (src == algo).sum()
        print(f"  {algo}: {c} transitions")


if __name__ == "__main__":
    main()
