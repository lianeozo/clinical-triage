"""MBPO agent orchestrator.

Ties together the three Phase 4 components behind a single ``act`` interface:

* ``EnsembleModel`` (T2) — K recurrent transition members.
* ``PiVNet`` (T3) — separate-encoder policy/value heads.
* ``run_mcts`` (T4) — AlphaZero-style planner over the ensemble.

The agent exposes:

    MBPOAgent.act(obs, history, step_in_episode=0) -> (action_idx, info)
    MBPOAgent.save(path) / .load(path)              -> serialise both nets

Two callables provided by the caller wire in the env's known dynamics:

    env_reward_fn(obs_now, action) -> float    # per-step reward
    env_done_fn(next_obs)          -> bool     # episode termination flag

The ``no_mcts`` flag is the ablation switch from spec §7: when True, the
planner is skipped entirely and the action is sampled directly from the
policy head (still under the same temperature schedule).

Spec: docs/superpowers/specs/2026-05-20-part4-mbpo-mcts-design.md §3c.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from triage_rl.config import EnsembleModelConfig, MCTSConfig, PiVNetConfig
from phase4_mbpo_mcts.agents.ensemble_model import EnsembleModel
from phase4_mbpo_mcts.agents.mcts import run_mcts
from phase4_mbpo_mcts.agents.pi_v_net import PiVNet


# ----------------------------------------------------------------------------
# History encoding helpers
# ----------------------------------------------------------------------------

def _history_to_tensors(
    history: list[tuple[np.ndarray, int]],
    current_obs: np.ndarray,
    n_obs_vars: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a history (plus current obs) as ``(obs_hist, act_hist)`` tensors.

    The GRU consumes a sequence of ``(o_t, a_t)`` pairs. To feed the encoder
    with the *current* observation as the final timestep, we append the
    current obs paired with a dummy action 0. When the history is empty this
    yields a length-1 sequence ``[(current_obs, 0)]``.

    Args:
        history: Past ``(obs, action)`` pairs in the current episode. May be
            empty.
        current_obs: ``(V,)`` int array — the most recent observation, not
            yet in ``history``.
        n_obs_vars: ``len(obs_dims)`` — for the obs tensor's last dim.

    Returns:
        obs_hist: ``(1, T, V)`` int64.
        act_hist: ``(1, T)`` int64.
    """
    obs_list = [obs for obs, _ in history] + [np.asarray(current_obs)]
    act_list = [a for _, a in history] + [0]  # dummy trailing action

    obs_arr = np.stack([np.asarray(o, dtype=np.int64) for o in obs_list], axis=0)
    act_arr = np.asarray(act_list, dtype=np.int64)

    obs_hist = torch.from_numpy(obs_arr).unsqueeze(0)  # (1, T, V)
    act_hist = torch.from_numpy(act_arr).unsqueeze(0)  # (1, T)

    assert obs_hist.shape == (1, len(obs_list), n_obs_vars)
    assert act_hist.shape == (1, len(obs_list))
    return obs_hist, act_hist


def _root_history_to_tensors(
    history: list[tuple[np.ndarray, int]],
    n_obs_vars: int,
    n_obs_vars_fallback: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a bare history (no separate current obs) for MCTS leaves.

    MCTS passes the running ``history_so_far`` directly to ``pi_v_fn``; that
    list already contains the (obs, action) at the leaf. If the list is empty
    we pad with a single zero (obs, action) pair so the GRU has a timestep
    to consume — this only happens for an empty *root* history before any
    expansion, and the same convention is used at the root expansion.
    """
    if not history:
        obs_arr = np.zeros((1, n_obs_vars), dtype=np.int64)
        act_arr = np.zeros((1,), dtype=np.int64)
    else:
        obs_arr = np.stack(
            [np.asarray(o, dtype=np.int64) for o, _ in history], axis=0
        )
        act_arr = np.asarray([a for _, a in history], dtype=np.int64)

    obs_hist = torch.from_numpy(obs_arr).unsqueeze(0)  # (1, T, V)
    act_hist = torch.from_numpy(act_arr).unsqueeze(0)  # (1, T)
    return obs_hist, act_hist


# ----------------------------------------------------------------------------
# Temperature schedule
# ----------------------------------------------------------------------------

def _sample_with_temperature(
    weights: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """Sample an action index from ``weights`` raised to ``1/T`` and renormalized.

    ``weights`` may be visit counts (MCTS) or raw policy probabilities
    (no_mcts). At T=0 we return the argmax; at T>0 we exponentiate and
    sample from the resulting categorical.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if temperature <= 0.0:
        return int(np.argmax(weights))

    # Guard against all-zero (no visits) weights — fall back to uniform.
    if not np.any(weights > 0):
        return int(rng.integers(0, weights.size))

    powered = np.power(weights, 1.0 / temperature)
    total = powered.sum()
    if total <= 0.0 or not np.isfinite(total):
        return int(np.argmax(weights))
    probs = powered / total
    return int(rng.choice(weights.size, p=probs))


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------

class MBPOAgent:
    """Orchestrates ensemble model, π/V net, and MCTS at decision time.

    The ``act`` interface returns an action index plus an ``info`` dict with
    MCTS diagnostics (or ``None`` placeholders when ``no_mcts`` is True so
    downstream loggers can distinguish the two modes).
    """

    def __init__(
        self,
        obs_dims: list[int],
        n_actions: int,
        model_cfg: EnsembleModelConfig,
        pi_v_cfg: PiVNetConfig,
        mcts_cfg: MCTSConfig,
        env_reward_fn: Callable[[np.ndarray, int], float],
        env_done_fn: Callable[[np.ndarray], bool],
        seed: int,
        device: str = "cpu",
        no_mcts: bool = False,
    ) -> None:
        self.obs_dims = list(obs_dims)
        self.n_obs_vars = len(self.obs_dims)
        self.n_actions = int(n_actions)
        self.model_cfg = model_cfg
        self.pi_v_cfg = pi_v_cfg
        self.mcts_cfg = mcts_cfg
        self.env_reward_fn = env_reward_fn
        self.env_done_fn = env_done_fn
        self.seed = int(seed)
        self.device = device
        self.no_mcts = bool(no_mcts)

        self.model = EnsembleModel(
            obs_dims=self.obs_dims,
            n_actions=self.n_actions,
            config=model_cfg,
            seed=seed,
            device=device,
        )
        self.pi_v = PiVNet(
            obs_dims=self.obs_dims,
            n_actions=self.n_actions,
            config=pi_v_cfg,
            seed=seed + 1,  # decouple from ensemble member seeds
            device=device,
        )

        # One RNG drives Dirichlet noise, ensemble sampling, action sampling.
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- act -

    def act(
        self,
        obs: np.ndarray,
        history: list[tuple[np.ndarray, int]],
        step_in_episode: int = 0,
    ) -> tuple[int, dict]:
        """Choose an action for the given observation.

        Args:
            obs: ``(V,)`` int observation NOT yet in ``history``.
            history: Past ``(obs, action)`` pairs in the current episode.
            step_in_episode: 0-indexed step counter; drives the temperature
                schedule (T_initial until ``temperature_steps``, then
                T_final).

        Returns:
            ``(action_idx, info_dict)``. ``info_dict`` always contains the
            keys ``visit_counts, root_q, mean_depth, value``. In ``no_mcts``
            mode the three MCTS-only fields are ``None``.
        """
        obs = np.asarray(obs)
        T = self._temperature(step_in_episode)

        if self.no_mcts:
            return self._act_no_mcts(obs, history, T)
        return self._act_with_mcts(obs, history, T)

    # ---------------------------------------------- temperature schedule -

    def _temperature(self, step_in_episode: int) -> float:
        if step_in_episode < self.mcts_cfg.temperature_steps:
            return float(self.mcts_cfg.temperature_initial)
        return float(self.mcts_cfg.temperature_final)

    # -------------------------------------------------------- no_mcts mode -

    def _act_no_mcts(
        self,
        obs: np.ndarray,
        history: list[tuple[np.ndarray, int]],
        temperature: float,
    ) -> tuple[int, dict]:
        """Bypass MCTS: forward the π/V net on the same root_history MCTS
        would see and sample from the resulting policy.
        """
        # Match MCTS's pi_v_fn input convention: the root_history is exactly
        # the past (obs, action) list. Use the same fallback for empty.
        obs_hist, act_hist = _root_history_to_tensors(
            history, self.n_obs_vars, self.n_obs_vars
        )
        with torch.no_grad():
            logits, value = self.pi_v.forward(obs_hist, act_hist)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        value_scalar = float(value[0].item())

        action_idx = _sample_with_temperature(probs, temperature, self.rng)
        info = {
            "visit_counts": None,
            "root_q": None,
            "mean_depth": None,
            "value": value_scalar,
        }
        return action_idx, info

    # ----------------------------------------------------------- mcts mode -

    def _act_with_mcts(
        self,
        obs: np.ndarray,
        history: list[tuple[np.ndarray, int]],
        temperature: float,
    ) -> tuple[int, dict]:
        # Closures captured for run_mcts. They convert the planner's
        # numpy/list world into the tensors the torch components expect.

        def model_step_fn(
            hist: list[tuple[np.ndarray, int]],
            action: int,
            member_idx: int,
        ) -> tuple[np.ndarray, float, bool]:
            obs_hist, act_hist = _model_step_inputs(
                hist, action, self.n_obs_vars
            )
            with torch.no_grad():
                logits_list = self.model.predict(
                    obs_hist, act_hist, member_idx=member_idx
                )
            next_obs = _sample_next_obs(logits_list, self.rng)
            # obs_now is the last obs in history, or `obs` if history empty.
            if hist:
                obs_now = hist[-1][0]
            else:
                obs_now = obs
            r = float(self.env_reward_fn(obs_now, action))
            done = bool(self.env_done_fn(next_obs))
            return next_obs, r, done

        def pi_v_fn(
            hist: list[tuple[np.ndarray, int]],
        ) -> tuple[np.ndarray, float]:
            obs_hist, act_hist = _root_history_to_tensors(
                hist, self.n_obs_vars, self.n_obs_vars
            )
            with torch.no_grad():
                logits, value = self.pi_v.forward(obs_hist, act_hist)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float64)
            return probs, float(value[0].item())

        # The MCTS root_history is the past pairs only; the current obs is
        # passed separately as root_obs.
        result = run_mcts(
            root_obs=np.asarray(obs),
            root_history=list(history),
            model_step_fn=model_step_fn,
            pi_v_fn=pi_v_fn,
            config=self.mcts_cfg,
            n_actions=self.n_actions,
            n_ensemble_members=self.model_cfg.n_members,
            rng=self.rng,
        )

        # Build a dense visit-count vector for temperature sampling so that
        # actions with zero visits get zero probability mass.
        counts = np.zeros(self.n_actions, dtype=np.float64)
        for a, n in result.visit_counts.items():
            counts[a] = n

        action_idx = _sample_with_temperature(counts, temperature, self.rng)
        info = {
            "visit_counts": dict(result.visit_counts),
            "root_q": dict(result.root_q),
            "mean_depth": float(result.mean_depth),
            "value": float(result.value),
        }
        return action_idx, info

    # ---------------------------------------------------------- save/load -

    def save(self, path: Path) -> None:
        """Serialise both the ensemble model and the π/V net to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict_all(),
            "pi_v": self.pi_v.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: Path) -> None:
        """Inverse of ``save``. Replaces in-place weights of both nets."""
        path = Path(path)
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict_all(payload["model"])
        self.pi_v.load_state_dict(payload["pi_v"])


# ----------------------------------------------------------------------------
# Free-function helpers used by the MCTS closures
# ----------------------------------------------------------------------------

def _model_step_inputs(
    history: list[tuple[np.ndarray, int]],
    action: int,
    n_obs_vars: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (obs_hist, act_hist) tensors for the ensemble forward pass.

    The model predicts ``o_{t+1}`` from a sequence ending in ``(o_t, a_t)``.
    At a planner leaf the planner passes (history_so_far, action) where
    ``history_so_far`` is the prefix and the action is the one being taken
    from the leaf node's obs (which is the last obs in history_so_far —
    or, when history_so_far is empty, the root obs, but in that empty case
    the planner has already promoted the root obs into history_so_far before
    calling model_step_fn... actually it has NOT for the very first expansion
    from the root; see mcts.py: child_history = node.history_so_far + [(node.obs, a)]).
    So the model needs to see the (root_obs, a) pair as the final timestep.

    However, the MCTS calls ``model_step_fn(node.history_so_far, a, member_idx)``
    where ``node.history_so_far`` does NOT yet include ``node.obs``. We
    reconstruct the GRU input by inferring obs_now from history (or by an
    implicit fallback)... rather than that, we pair the action with a dummy
    obs at the end (the last obs is in history if non-empty; at the root, we
    use the convention that the empty-history root has its obs encoded via
    the dummy-pad path used for pi_v_fn).

    This implementation builds the sequence as
    ``[(o_0, a_0), ..., (o_{t-1}, a_{t-1}), (last_obs, action)]`` where
    ``last_obs`` is the most recent obs in history (or zeros if history is
    empty — matching the empty-history convention used elsewhere).
    """
    if history:
        prev_obs_arr = np.stack(
            [np.asarray(o, dtype=np.int64) for o, _ in history], axis=0
        )
        prev_act_arr = np.asarray([a for _, a in history], dtype=np.int64)
        last_obs = np.asarray(history[-1][0], dtype=np.int64)
    else:
        prev_obs_arr = np.zeros((0, n_obs_vars), dtype=np.int64)
        prev_act_arr = np.zeros((0,), dtype=np.int64)
        last_obs = np.zeros(n_obs_vars, dtype=np.int64)

    obs_arr = np.concatenate(
        [prev_obs_arr, last_obs[np.newaxis, :]], axis=0
    )  # (T, V)
    act_arr = np.concatenate(
        [prev_act_arr, np.asarray([action], dtype=np.int64)], axis=0
    )  # (T,)

    obs_hist = torch.from_numpy(obs_arr).unsqueeze(0)
    act_hist = torch.from_numpy(act_arr).unsqueeze(0)
    return obs_hist, act_hist


def _sample_next_obs(
    logits_list: list[torch.Tensor],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one int per variable from the per-var categorical heads.

    ``logits_list[i]`` has shape ``(1, dim_i)`` (B=1 inside the planner).
    Returns a ``(V,)`` int64 array.
    """
    parts = []
    for logits in logits_list:
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float64)
        # Guard against numerical drift -> renormalize.
        probs = np.clip(probs, 0.0, None)
        s = probs.sum()
        if s <= 0.0 or not np.isfinite(s):
            idx = 0
        else:
            probs = probs / s
            idx = int(rng.choice(probs.size, p=probs))
        parts.append(idx)
    return np.asarray(parts, dtype=np.int64)
