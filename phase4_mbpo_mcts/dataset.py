"""Phase 4 seed-dataset assembly and trajectory buffer.

Spec: ``docs/superpowers/specs/2026-05-20-part4-mbpo-mcts-design.md`` §4.

Two responsibilities:

1. ``assemble_seed_100k`` — slice 100K transitions (25K Random + 75K SAC-KL-F)
   out of the Phase 3 mixed offline dataset to seed Phase 4's buffer and the
   transition-model pretraining.
2. ``TrajectoryBuffer`` — minimal episode-aware replay buffer that backs both
   the recurrent-model trainer (``sample_segments``) and the AlphaZero π/V
   trainer (``sample_steps``).

Per the spec, the seed-mix order is irrelevant (the buffer shuffles on
sample), but the concatenated 100K is shuffled once with a seeded RNG for
reproducibility regardless of how it's consumed downstream.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


# ---- seed assembly -------------------------------------------------------


def assemble_seed_100k(src_path: str, seed: int) -> dict:
    """Build a 100K seed dataset by mixing Random and SAC-KL-F transitions.

    Args:
        src_path: Path to the Phase 3 ``.npz`` with keys ``obs``, ``action``,
            ``reward``, ``next_obs``, ``terminated``, ``source_algo``.
        seed: Seed for the SAC-KL-F subsampling and the final shuffle.

    Returns:
        Dict with the same keys as the source file. Length is
        25_000 (or all available) + 75_000 = up to 100_000.
    """
    with np.load(src_path, allow_pickle=True) as src:
        obs = src["obs"]
        action = src["action"]
        reward = src["reward"]
        next_obs = src["next_obs"]
        terminated = src["terminated"]
        source_algo = src["source_algo"]

    rng = np.random.default_rng(seed)

    random_idx = np.flatnonzero(source_algo == "random")
    sac_idx = np.flatnonzero(source_algo == "sac_kl_f")

    # Random: take up to the full 25K subset.
    n_random_take = min(25_000, len(random_idx))
    random_sel = random_idx[:n_random_take]

    # SAC-KL-F: uniform subsample of 75K (or all if fewer exist).
    n_sac_take = min(75_000, len(sac_idx))
    sac_sel = rng.choice(sac_idx, size=n_sac_take, replace=False)

    sel = np.concatenate([random_sel, sac_sel])
    # Deterministic shuffle of the concatenated mix.
    rng.shuffle(sel)

    return {
        "obs": obs[sel],
        "action": action[sel],
        "reward": reward[sel],
        "next_obs": next_obs[sel],
        "terminated": terminated[sel],
        "source_algo": source_algo[sel],
    }


# ---- trajectory buffer ---------------------------------------------------


# Each episode is a list of per-step tuples:
#   (obs: np.ndarray[V], action: int, reward: float,
#    next_obs: np.ndarray[V], done: bool)
Transition = tuple[np.ndarray, int, float, np.ndarray, bool]
Episode = list[Transition]


class TrajectoryBuffer:
    """Simple episode-keyed replay buffer.

    Stores episodes as Python lists of transitions and maintains a flat
    index of valid ``(episode_idx, start_idx)`` pairs for fast segment
    sampling. ``segment_len`` is bound at sample-time, so the valid-start
    index is recomputed lazily per ``sample_segments`` call (cheap: O(E)).
    """

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self._episodes: list[Episode] = []
        self._n_transitions: int = 0

    # ---- mutation -------------------------------------------------------

    def append_episode(self, episode: Episode) -> None:
        if len(episode) == 0:
            return
        # Defensive copy so caller mutations don't corrupt the buffer.
        ep_copy: Episode = [
            (
                np.asarray(o, dtype=np.int64).reshape(self.obs_dim).copy(),
                int(a),
                float(r),
                np.asarray(no, dtype=np.int64).reshape(self.obs_dim).copy(),
                bool(d),
            )
            for (o, a, r, no, d) in episode
        ]
        self._episodes.append(ep_copy)
        self._n_transitions += len(ep_copy)

    def bulk_load_from_dict(
        self,
        data: dict,
        episode_boundaries: list[tuple[int, int]] | None = None,
    ) -> None:
        """Append many episodes at once from a dict-of-arrays.

        Args:
            data: Dict with at minimum ``obs``, ``action``, ``reward``,
                ``next_obs``, ``terminated`` arrays of the same length N.
            episode_boundaries: Optional explicit list of ``(start, end)``
                exclusive-end half-open intervals into the arrays. If None,
                boundaries are inferred from ``data["terminated"]`` (each
                ``True`` flag marks the last step of an episode; any trailing
                steps after the final ``True`` are still grouped as one
                episode).
        """
        obs = np.asarray(data["obs"])
        action = np.asarray(data["action"])
        reward = np.asarray(data["reward"])
        next_obs = np.asarray(data["next_obs"])
        terminated = np.asarray(data["terminated"]).astype(bool)
        n = len(obs)

        if episode_boundaries is None:
            episode_boundaries = []
            start = 0
            for i in range(n):
                if terminated[i]:
                    episode_boundaries.append((start, i + 1))
                    start = i + 1
            if start < n:  # trailing un-terminated steps -> one final episode
                episode_boundaries.append((start, n))

        for (s, e) in episode_boundaries:
            ep: Episode = [
                (obs[t], int(action[t]), float(reward[t]),
                 next_obs[t], bool(terminated[t]))
                for t in range(s, e)
            ]
            self.append_episode(ep)

    # ---- introspection --------------------------------------------------

    @property
    def n_transitions(self) -> int:
        return self._n_transitions

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    # ---- sampling -------------------------------------------------------

    def _valid_starts(self, segment_len: int) -> list[tuple[int, int]]:
        """Return all ``(ep_idx, start_idx)`` pairs s.t. ep can yield a
        segment of length ``segment_len`` AND has a step at index
        ``start + segment_len`` to serve as the prediction target."""
        out: list[tuple[int, int]] = []
        for ei, ep in enumerate(self._episodes):
            # Need start..start+segment_len-1 as history (segment_len steps)
            # AND step at index start+segment_len to read next_obs from.
            max_start = len(ep) - segment_len  # inclusive
            if max_start <= 0:
                # max_start == 0 is fine in principle (start=0 -> need step
                # at index segment_len, i.e., len(ep) > segment_len). Convert
                # to strict-positive: require len(ep) > segment_len.
                continue
            for s in range(max_start):
                out.append((ei, s))
        return out

    def sample_segments(
        self,
        n: int,
        segment_len: int,
        rng: np.random.Generator,
    ) -> dict:
        """Sample ``n`` independent random segments of length ``segment_len``.

        Returns a dict with:
            obs_hist: int64 array of shape ``[n, segment_len, obs_dim]``
            act_hist: int64 array of shape ``[n, segment_len]``
            next_obs: int64 array of shape ``[n, obs_dim]`` — the obs at step
                ``start + segment_len`` (the model's prediction target).
        """
        valid = self._valid_starts(segment_len)
        if len(valid) == 0:
            raise ValueError(
                f"no episodes long enough to yield a segment of length "
                f"{segment_len} (need >={segment_len + 1} steps)")

        choices = rng.integers(0, len(valid), size=n)
        obs_hist = np.empty((n, segment_len, self.obs_dim), dtype=np.int64)
        act_hist = np.empty((n, segment_len), dtype=np.int64)
        next_obs = np.empty((n, self.obs_dim), dtype=np.int64)
        for i, ci in enumerate(choices):
            ei, s = valid[int(ci)]
            ep = self._episodes[ei]
            for t in range(segment_len):
                o, a, _r, _no, _d = ep[s + t]
                obs_hist[i, t] = o
                act_hist[i, t] = a
            # Target = obs at the step immediately after the segment.
            target = ep[s + segment_len][0]
            next_obs[i] = target
        return {
            "obs_hist": obs_hist,
            "act_hist": act_hist,
            "next_obs": next_obs,
        }

    def sample_steps(self, n: int, rng: np.random.Generator) -> dict:
        """Sample ``n`` random transitions uniformly across the buffer.

        Returns a dict with ``obs``, ``action``, ``reward``, ``next_obs``,
        ``done``. Phase 4 T7 will extend the consumer side to add ``history``,
        ``visit_dist``, and ``return_to_go``; this minimal sampler covers T6.
        """
        if self._n_transitions == 0:
            raise ValueError("buffer is empty")

        # Build a flat (ep_idx, step_idx) view; cheap to recompute since
        # this method is called once per π/V minibatch and episode count is
        # bounded by the outer loop's collect budget.
        flat: list[tuple[int, int]] = []
        for ei, ep in enumerate(self._episodes):
            for si in range(len(ep)):
                flat.append((ei, si))

        choices = rng.integers(0, len(flat), size=n)
        obs = np.empty((n, self.obs_dim), dtype=np.int64)
        action = np.empty((n,), dtype=np.int64)
        reward = np.empty((n,), dtype=np.float32)
        next_obs = np.empty((n, self.obs_dim), dtype=np.int64)
        done = np.empty((n,), dtype=bool)
        for i, ci in enumerate(choices):
            ei, si = flat[int(ci)]
            o, a, r, no, d = self._episodes[ei][si]
            obs[i] = o
            action[i] = a
            reward[i] = r
            next_obs[i] = no
            done[i] = d
        return {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
        }
