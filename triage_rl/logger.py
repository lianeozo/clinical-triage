"""JSONL logger. One file per instrumentation bucket per seed.

All writes flush immediately so partially-completed runs leave readable logs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Logger:
    EPISODE_FILE = "train_episodes.jsonl"
    CHECKPOINT_FILE = "eval_checkpoints.jsonl"
    INTERNALS_FILE = "training_internals.jsonl"
    TRAJECTORY_DIR = "eval_trajectories"

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / self.TRAJECTORY_DIR).mkdir(parents=True, exist_ok=True)
        self._files: dict[str, Any] = {}

    def _open(self, name: str):
        if name not in self._files:
            self._files[name] = open(self.out_dir / name, "a", buffering=1)
        return self._files[name]

    def _write(self, name: str, record: dict) -> None:
        record = {**record, "wall_time": time.time()}
        f = self._open(name)
        f.write(json.dumps(record) + "\n")
        f.flush()

    def log_episode(self, step: int, return_: float, length: int, terminal_reason: str) -> None:
        self._write(self.EPISODE_FILE, {
            "step": int(step),
            "return": float(return_),
            "length": int(length),
            "terminal_reason": str(terminal_reason),
        })

    def log_checkpoint(self, step: int, aggregates: dict) -> None:
        record = {"step": int(step), **aggregates}
        self._write(self.CHECKPOINT_FILE, record)

    def log_internals(self, step: int, metrics: dict) -> None:
        record = {"step": int(step), **metrics}
        self._write(self.INTERNALS_FILE, record)

    def log_eval_trajectory(self, step: int, episode_idx: int, trajectory: dict) -> None:
        path = self.out_dir / self.TRAJECTORY_DIR / f"step_{step}.jsonl"
        with open(path, "a", buffering=1) as f:
            f.write(json.dumps({"episode_idx": int(episode_idx), **trajectory}) + "\n")

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
