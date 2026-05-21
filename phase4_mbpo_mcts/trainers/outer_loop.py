"""Phase 4 outer-loop trainer.

Glues the five Phase 4 components into the iterative loop described by
spec §5 and §6:

    seed buffer  ─────► pretrain ensemble model
                          │
                          ▼
                    eval i=0  (random π/V + pretrained model)
                          │
       ┌──────────────────┴──────────────────────────────────┐
       │ for i in 1..N_iters:                                │
       │     A. collect M episodes via MBPOAgent.act         │
       │        (records (history, visit_dist, return)       │
       │         tuples for π/V training; appends episode    │
       │         transitions to the buffer for model train)  │
       │     B. retrain ensemble model on full buffer        │
       │     C. train π/V on the collected (h, π_mcts, G)    │
       │        records via AlphaZero policy + value loss    │
       │     D. eval i, log internals                        │
       └──────────────────────────────────────────────────────┘

Spec: ``docs/superpowers/specs/2026-05-20-part4-mbpo-mcts-design.md`` §5–§6.

Known shortcuts (documented as CONCERNs in the implementation report):

* ``env_reward_fn`` returns the step-wise treatment + SOC-change cost only —
  the terminal ±10000 reward from ``MDP.calculateReward`` is NOT computed
  inside the planner (it depends on the next observation's full vital bins
  AND the diabetic hidden state, which the obs doesn't carry). Terminal
  reward propagates into the learner via the V_ψ bootstrap learning from
  real-env returns instead.
* ``env_done_fn`` calls ``State.check_absorbing_state`` on a synthesized
  State built from the obs vector. Masked vitals (-1 in the env's obs)
  are clipped to a valid bin first (see ``_sanitize_obs``).
* Obs ``-1`` values (masked vitals) are clipped to ``0`` everywhere before
  feeding to the ensemble / π/V net or storing in the buffer. The model
  has no "masked" bin.
* ``no_mcts`` mode skips the planner at action-selection time AND at π
  training time. The policy is updated via a REINFORCE-style log-prob loss
  weighted by Monte-Carlo returns ``G_t`` (spec §7 suggests a parallel MCTS
  for visit-count targets, which is more expensive; we defer that to a
  follow-up).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from phase4_mbpo_mcts.agents.mbpo_agent import MBPOAgent
from phase4_mbpo_mcts.dataset import TrajectoryBuffer, assemble_seed_100k
from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.agents.base import Agent
from triage_rl.config import (
    EnsembleModelConfig,
    MCTSConfig,
    OuterLoopConfig,
    PiVNetConfig,
)
from triage_rl.env import Env
from triage_rl.logger import Logger
from triage_rl.trainers.off_policy import make_eval_pool, seed_everything


# ----------------------------------------------------------------------------
# Constants derived from the existing State / Action layouts
# ----------------------------------------------------------------------------

_OBS_DIMS: list[int] = [
    State.NUM_HR,
    State.NUM_SYSBP,
    State.NUM_OXYG,
    State.NUM_GLUC,
    State.NUM_ANTIB,
    State.NUM_VASO,
    State.NUM_VENT,
    State.NUM_SOC,
]
_N_OBS_VARS = len(_OBS_DIMS)
_N_ACTIONS = Action.NUM_ACTIONS_TOTAL
_SOC_IDX = State.SOC_IDX


# ----------------------------------------------------------------------------
# Obs sanitation: clip -1 masked vitals to 0 before any model/agent use.
# ----------------------------------------------------------------------------

def _sanitize_obs(obs: np.ndarray) -> np.ndarray:
    """Replace ``-1`` (masked vital) with ``0`` and return an int64 ``(V,)``.

    The ensemble model and the π/V net build one-hot inputs with
    ``F.one_hot(num_classes=d)`` which rejects negative indices. The env's
    ``Env.step`` emits ``-1`` whenever a vital is masked at the patient's
    current site of care. We collapse all such bins into the lowest bin (0)
    so the model has a stable input distribution.

    This is information-lossy (masked vs lo for a 3-bin vital like HR are
    no longer distinguishable). A follow-up could shift indices by 1 and
    expand ``OBS_DIMS`` accordingly.
    """
    arr = np.asarray(obs, dtype=np.int64).copy()
    arr[arr < 0] = 0
    return arr


def _sanitize_obs_array(obs_arr: np.ndarray) -> np.ndarray:
    """Vectorized variant of ``_sanitize_obs`` for a 2-D ``[N, V]`` array."""
    arr = np.asarray(obs_arr, dtype=np.int64).copy()
    arr[arr < 0] = 0
    return arr


# ----------------------------------------------------------------------------
# env_reward_fn / env_done_fn shortcuts (see module docstring CONCERNs)
# ----------------------------------------------------------------------------

def _env_reward_fn(obs: np.ndarray, action_idx: int) -> float:
    """Approximate per-step reward from (obs, action) alone.

    Reproduces the deterministic portion of ``MDP.calculateReward``:

    * SOC change cost (-50 if action.soc differs from current soc).
    * SOC mismatch-with-severity penalty: this requires post-step
      ``num_abnormal`` so we skip it (the value head learns this signal).
    * Treatment costs: antibiotic=-10, ventilation=-60, vasopressors=-40.

    Does NOT compute the terminal ±10000 or the per-step change-in-abnormal
    reward — those depend on next-obs and (for glucose) the hidden diabetic
    state. The planner therefore underestimates per-edge rewards; V_ψ
    closes the gap via its real-env-return targets.
    """
    obs_arr = _sanitize_obs(obs)
    current_soc = int(obs_arr[_SOC_IDX])
    act = Action(action_idx=int(action_idx))
    reward = 0.0
    if act.soc != current_soc:
        reward -= 50.0
    if act.antibiotic == 1:
        reward -= 10.0
    if act.ventilation == 1:
        reward -= 60.0
    if act.vasopressors == 1:
        reward -= 40.0
    return reward


def _env_done_fn(next_obs: np.ndarray) -> bool:
    """Termination check from the post-step observation alone.

    Builds a ``State`` from the sanitized obs (diabetic_idx=0 placeholder,
    irrelevant for ``check_absorbing_state``) and reuses the existing
    absorbing-state predicate.
    """
    obs_arr = _sanitize_obs(next_obs)
    state = State(state_categs=obs_arr.tolist(), diabetic_idx=0)
    return bool(state.check_absorbing_state())


# ----------------------------------------------------------------------------
# π/V training-record container
# ----------------------------------------------------------------------------

@dataclass
class _PVRecord:
    """One step of one episode, kept for AlphaZero π/V training.

    Attributes:
        history: ``[(obs, action), ...]`` BEFORE the current step. The π/V
            net's encoder consumes ``history + [(obs, 0)]`` so the GRU sees
            the obs at the step we're predicting at.
        obs: ``(V,)`` int observation at this step (already sanitized).
        action: Action index actually taken in the real env.
        visit_dist: Dense ``(n_actions,)`` MCTS visit distribution at this
            step, or ``None`` if running in ``no_mcts`` mode.
        return_to_go: Discounted MC return from this step to end of episode
            (computed in a second pass once the episode finishes).
    """

    history: list[tuple[np.ndarray, int]]
    obs: np.ndarray
    action: int
    visit_dist: np.ndarray | None
    return_to_go: float = 0.0


# ----------------------------------------------------------------------------
# MBPOAgent → triage_rl.agents.base.Agent adapter (so Evaluator works)
# ----------------------------------------------------------------------------

class _EvalAgentAdapter(Agent):
    """Adapter that exposes an ``MBPOAgent`` as the simple
    ``Agent.act(obs, eval_mode)`` interface.

    Maintains an internal per-episode ``history`` (the list of past
    ``(obs, action)`` pairs the MCTS planner needs) and a step counter for
    the temperature schedule. The owning trainer (or test) must call
    ``reset_episode()`` between episodes — there is no automatic
    boundary-detection.

    Used by ``OuterLoopTrainer._run_evaluation``, which loops one episode
    at a time and explicitly calls ``reset_episode()`` between them.
    """

    def __init__(self, agent: MBPOAgent) -> None:
        self._agent = agent
        self._history: list[tuple[np.ndarray, int]] = []
        self._step_in_episode: int = 0

    def reset_episode(self) -> None:
        self._history = []
        self._step_in_episode = 0

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:  # type: ignore[override]
        # Evaluator passes a float obs from the Env; sanitize and integerize.
        obs_int = _sanitize_obs(obs)
        action, _info = self._agent.act(
            obs_int, history=self._history, step_in_episode=self._step_in_episode
        )
        # Append to history AFTER the call (the call sees history before the
        # current step). Use a copy so the buffer keeps an immutable view.
        self._history.append((obs_int.copy(), int(action)))
        self._step_in_episode += 1
        return int(action)


# ----------------------------------------------------------------------------
# OuterLoopTrainer
# ----------------------------------------------------------------------------

class OuterLoopTrainer:
    """Iterative MBPO-MCTS trainer over a real env + ensemble model.

    Construct with the four configs + the ablation switch, call
    ``run()``. ``run()`` returns the seed-buffer transition count so the
    test can assert the buffer grew.
    """

    def __init__(
        self,
        outer_cfg: OuterLoopConfig,
        model_cfg: EnsembleModelConfig,
        pi_v_cfg: PiVNetConfig,
        mcts_cfg: MCTSConfig,
        no_mcts: bool,
        algo_name: str,
        device: str = "cpu",
        env_factory=None,
    ) -> None:
        self.cfg = outer_cfg
        self.model_cfg = model_cfg
        self.pi_v_cfg = pi_v_cfg
        self.mcts_cfg = mcts_cfg
        self.no_mcts = bool(no_mcts)
        self.algo_name = algo_name
        self.device = device
        self.env_factory = env_factory or (lambda: Env(p_diabetes=0.2, max_steps=100))

        # Built in ``run()``.
        self.buffer: TrajectoryBuffer | None = None
        self.agent: MBPOAgent | None = None
        self.logger: Logger | None = None
        self._heldout_obs: np.ndarray | None = None
        self._heldout_act: np.ndarray | None = None
        self._heldout_next: np.ndarray | None = None
        self._train_rng: np.random.Generator | None = None
        self._torch_rng: torch.Generator | None = None
        self._last_mean_depth: float = 0.0

    # =========================================================== run() =====

    def run(self) -> int:
        """Execute the outer loop. Returns the seed-buffer transition count
        (so the test can assert post-iter buffer growth).
        """
        seed_everything(self.cfg.seed)
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(self.cfg.out_dir)
        self._train_rng = np.random.default_rng(self.cfg.seed)
        self._torch_rng = torch.Generator(device=self.device)
        self._torch_rng.manual_seed(self.cfg.seed)

        # 1. Eval pool (deterministic, mirrors Phases 1-3).
        eval_pool = make_eval_pool(self.cfg.seed, n=self.cfg.n_eval_episodes)
        (self.cfg.out_dir / "eval_pool.json").write_text(json.dumps(eval_pool))

        # 2. Seed dataset → buffer (held out last 5K for NLL diagnostic).
        data = assemble_seed_100k(str(self.cfg.seed_dataset_path), self.cfg.seed)
        data_sanitized = {
            "obs": _sanitize_obs_array(data["obs"]),
            "action": np.asarray(data["action"], dtype=np.int64),
            "reward": np.asarray(data["reward"], dtype=np.float32),
            "next_obs": _sanitize_obs_array(data["next_obs"]),
            "terminated": np.asarray(data["terminated"]).astype(bool),
        }
        n_total = len(data_sanitized["obs"])
        n_heldout = min(5_000, max(1, n_total // 10))
        n_buffer = n_total - n_heldout
        self._heldout_obs = data_sanitized["obs"][n_buffer:]
        self._heldout_act = data_sanitized["action"][n_buffer:]
        self._heldout_next = data_sanitized["next_obs"][n_buffer:]

        buf_data = {
            k: v[:n_buffer] for k, v in data_sanitized.items()
        }
        self.buffer = TrajectoryBuffer(
            obs_dim=_N_OBS_VARS, n_actions=_N_ACTIONS
        )
        self.buffer.bulk_load_from_dict(buf_data)
        n_seed_transitions = self.buffer.n_transitions

        # 3. Build agent (random init π/V, fresh ensemble).
        self.agent = MBPOAgent(
            obs_dims=_OBS_DIMS,
            n_actions=_N_ACTIONS,
            model_cfg=self.model_cfg,
            pi_v_cfg=self.pi_v_cfg,
            mcts_cfg=self.mcts_cfg,
            env_reward_fn=_env_reward_fn,
            env_done_fn=_env_done_fn,
            seed=self.cfg.seed,
            device=self.device,
            no_mcts=self.no_mcts,
        )

        # 4. Pretrain ensemble model (one pass of n_model_steps_per_iter
        #    over each member, samples per-step from the buffer).
        self._train_model_steps(self.cfg.n_model_steps_per_iter, label="pretrain")

        # 5. i=0 eval (baseline: pretrained model + random π/V).
        self._run_evaluation(outer_iter=0)

        # 6. Build evaluator just for the reference shape — we use a custom
        #    rollout via _EvalAgentAdapter inside _run_evaluation above.

        # 7. Outer loop.
        for i in range(1, self.cfg.n_outer_iters + 1):
            t_iter = time.time()

            # A. Collect M episodes via MBPOAgent; record (history, vd, G).
            pv_records, n_collected = self._collect_episodes(self.cfg.n_episodes_per_iter)

            # B. Retrain transition model on the (now larger) buffer.
            model_nll = self._train_model_steps(
                self.cfg.n_model_steps_per_iter, label=f"iter{i}"
            )

            # C. Train π/V on the iter's pv_records.
            pv_metrics = self._train_pi_v_steps(
                pv_records, self.cfg.n_pv_steps_per_iter
            )

            # D. Eval + internals log.
            self._run_evaluation(outer_iter=i)
            heldout_nll = self._heldout_nll()
            self.logger.log_internals(
                step=i,
                metrics={
                    "outer_iter": i,
                    "model_nll_heldout": heldout_nll,
                    "model_nll_train_last": model_nll,
                    "mcts_mean_depth": float(self._last_mean_depth),
                    "policy_entropy": pv_metrics["policy_entropy"],
                    "value_mean": pv_metrics["value_mean"],
                    "n_episodes_collected": n_collected,
                    "buffer_n_transitions": self.buffer.n_transitions,
                    "iter_wall_s": time.time() - t_iter,
                },
            )

        self.logger.close()
        return n_seed_transitions

    # ============================================ collect episodes (real env)

    def _collect_episodes(
        self, n_episodes: int
    ) -> tuple[list[_PVRecord], int]:
        """Roll out ``n_episodes`` in the real env via ``self.agent.act``.

        Each step's (history, visit_dist, return_to_go) is recorded for π/V
        training. The full episode is appended to ``self.buffer`` for the
        next model-training round.

        Returns ``(pv_records, n_episodes_collected)``.
        """
        assert self.agent is not None and self.buffer is not None
        assert self._train_rng is not None
        gamma = float(self.mcts_cfg.gamma)
        pv_records: list[_PVRecord] = []
        depths: list[float] = []

        for _ep in range(n_episodes):
            env = self.env_factory()
            seed = int(self._train_rng.integers(0, 2**31 - 1))
            obs, _info = env.reset(seed=seed)
            obs = _sanitize_obs(obs)

            history: list[tuple[np.ndarray, int]] = []
            episode_transitions: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
            step_pv_records: list[_PVRecord] = []
            rewards: list[float] = []
            step_in_episode = 0

            while True:
                action, info = self.agent.act(
                    obs.copy(),
                    history=history,
                    step_in_episode=step_in_episode,
                )
                if info.get("mean_depth") is not None:
                    depths.append(float(info["mean_depth"]))

                visit_dist: np.ndarray | None
                if self.no_mcts or info.get("visit_counts") is None:
                    visit_dist = None
                else:
                    counts = np.zeros(_N_ACTIONS, dtype=np.float64)
                    total = 0
                    for a, n in info["visit_counts"].items():
                        counts[a] = n
                        total += n
                    if total > 0:
                        visit_dist = counts / total
                    else:
                        visit_dist = None

                step_pv_records.append(_PVRecord(
                    history=[(h_obs.copy(), int(h_a)) for h_obs, h_a in history],
                    obs=obs.copy(),
                    action=int(action),
                    visit_dist=visit_dist,
                    return_to_go=0.0,  # filled after the episode ends
                ))

                next_obs_raw, raw_reward, terminated, truncated, env_info = env.step(int(action))
                next_obs = _sanitize_obs(next_obs_raw)
                done = bool(terminated or truncated)
                episode_transitions.append(
                    (obs.copy(), int(action), float(raw_reward), next_obs.copy(), bool(terminated))
                )
                rewards.append(float(raw_reward) * float(self.cfg.reward_scale))

                history.append((obs.copy(), int(action)))
                obs = next_obs
                step_in_episode += 1

                if done:
                    break

            # Compute discounted returns-to-go and append to records.
            G = 0.0
            returns: list[float] = [0.0] * len(rewards)
            for t in reversed(range(len(rewards))):
                G = rewards[t] + gamma * G
                returns[t] = G
            for rec, g in zip(step_pv_records, returns):
                rec.return_to_go = float(g)

            pv_records.extend(step_pv_records)
            self.buffer.append_episode(episode_transitions)

        self._last_mean_depth = float(np.mean(depths)) if depths else 0.0
        return pv_records, n_episodes

    # ============================================================ model train

    def _train_model_steps(self, n_steps: int, label: str = "") -> float:
        """Train every ensemble member for ``n_steps`` gradient steps.

        Each step samples a fresh segment minibatch per member from the
        buffer; loss is the per-variable categorical NLL.
        Returns the last loss value (averaged across members) for logging.
        """
        assert self.agent is not None and self.buffer is not None
        assert self._train_rng is not None
        model = self.agent.model
        batch_size = max(2, self.cfg.batch_size)
        seg_len = max(2, self.cfg.segment_len)

        last_loss = float("nan")
        if n_steps <= 0 or self.buffer.n_transitions == 0:
            return last_loss

        for step in range(n_steps):
            try:
                seg = self.buffer.sample_segments(
                    n=batch_size, segment_len=seg_len, rng=self._train_rng
                )
            except ValueError:
                # No episode long enough for this segment length; halve and retry.
                if seg_len > 2:
                    seg_len = max(2, seg_len // 2)
                    continue
                return last_loss

            obs_hist = torch.from_numpy(seg["obs_hist"]).long()
            act_hist = torch.from_numpy(seg["act_hist"]).long()
            target_next = torch.from_numpy(seg["next_obs"]).long()
            losses_this_step = []
            for k in range(self.model_cfg.n_members):
                loss = model.compute_loss(obs_hist, act_hist, target_next, member_idx=k)
                model.optimizers[k].zero_grad()
                loss.backward()
                model.optimizers[k].step()
                losses_this_step.append(float(loss.item()))
            last_loss = float(np.mean(losses_this_step))

            if step % 100 == 0:
                # Print every 100 steps for sanity (matches spec §5 step 3).
                print(f"[outer_loop:{label}] model step {step}/{n_steps} "
                      f"avg_nll={last_loss:.4f}")
        return last_loss

    # ========================================================== π/V training

    def _train_pi_v_steps(
        self, pv_records: list[_PVRecord], n_steps: int
    ) -> dict[str, float]:
        """Train π/V on ``pv_records`` for ``n_steps`` gradient steps.

        Loss components (spec §5 step C):

        * ``loss_v = MSE(V_ψ(h_t), G_t)``
        * ``loss_pi``:
            - MCTS mode: cross-entropy with the MCTS visit distribution
              as soft target (``-(π_target * log_softmax(logits)).sum()``).
            - no_mcts mode: REINFORCE-style log-likelihood weighted by the
              return (``-log π(a_t | h_t) * G_t``). See CONCERN in module
              docstring.

        Returns diagnostic metrics: ``{policy_entropy, value_mean,
        loss_pi, loss_v}``.
        """
        assert self.agent is not None
        pi_v = self.agent.pi_v
        batch_size = max(2, self.cfg.batch_size)
        metrics = {
            "policy_entropy": 0.0,
            "value_mean": 0.0,
            "loss_pi": 0.0,
            "loss_v": 0.0,
        }
        if n_steps <= 0 or len(pv_records) == 0:
            return metrics

        # In MCTS mode, only records WITH visit_dist contribute to π loss.
        mcts_records = [r for r in pv_records if r.visit_dist is not None]
        records_for_pi = mcts_records if not self.no_mcts else pv_records
        records_for_v = pv_records  # V trains on every step regardless.
        if len(records_for_v) == 0:
            return metrics

        for step in range(n_steps):
            # Sample indices into the full pv_records (uniform); V trains on
            # this. For π, sample from records_for_pi (which may be the same
            # or a subset).
            v_idx = self._train_rng.integers(0, len(records_for_v), size=batch_size)
            pi_idx = self._train_rng.integers(0, len(records_for_pi), size=batch_size) \
                if records_for_pi else None

            # --- Value loss (over v_idx) ---
            obs_hist_v, act_hist_v, _ = self._records_to_batch_tensors(
                [records_for_v[i] for i in v_idx]
            )
            returns_v = torch.tensor(
                [records_for_v[i].return_to_go for i in v_idx],
                dtype=torch.float32,
            )
            logits_v, value_v = pi_v.forward(obs_hist_v, act_hist_v)
            loss_v = F.mse_loss(value_v, returns_v.to(value_v.device))

            # --- Policy loss (over pi_idx) ---
            if pi_idx is None or len(records_for_pi) == 0:
                loss_pi = torch.tensor(0.0, device=logits_v.device)
            else:
                obs_hist_p, act_hist_p, _ = self._records_to_batch_tensors(
                    [records_for_pi[i] for i in pi_idx]
                )
                logits_p, _v_p = pi_v.forward(obs_hist_p, act_hist_p)
                log_probs = F.log_softmax(logits_p, dim=-1)

                if not self.no_mcts:
                    targets = torch.tensor(
                        np.stack([records_for_pi[i].visit_dist for i in pi_idx], axis=0),
                        dtype=torch.float32, device=log_probs.device,
                    )
                    loss_pi = -(targets * log_probs).sum(dim=-1).mean()
                else:
                    # REINFORCE-style: -log π(a_t) * G_t.
                    actions = torch.tensor(
                        [records_for_pi[i].action for i in pi_idx],
                        dtype=torch.long, device=log_probs.device,
                    )
                    returns_p = torch.tensor(
                        [records_for_pi[i].return_to_go for i in pi_idx],
                        dtype=torch.float32, device=log_probs.device,
                    )
                    chosen_log_p = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                    loss_pi = -(chosen_log_p * returns_p).mean()

            loss = loss_pi + loss_v
            pi_v.optimizer.zero_grad()
            loss.backward()
            pi_v.optimizer.step()

            metrics["loss_pi"] = float(loss_pi.item())
            metrics["loss_v"] = float(loss_v.item())
            with torch.no_grad():
                probs = F.softmax(logits_v, dim=-1)
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
                metrics["policy_entropy"] = float(entropy.item())
                metrics["value_mean"] = float(value_v.mean().item())

        return metrics

    def _records_to_batch_tensors(
        self, records: list[_PVRecord]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad and stack the histories of a batch of ``_PVRecord``s.

        The π/V encoder consumes ``history + [(obs, 0)]``. Histories vary
        in length across records; we right-pad with the first obs of the
        record (a benign no-op since GRU initial-state behavior dominates)
        and a dummy action 0. Returns ``(obs_hist[B, T, V], act_hist[B, T],
        seq_len[B])``.
        """
        seqs: list[tuple[np.ndarray, np.ndarray]] = []
        for rec in records:
            obs_seq = [np.asarray(o, dtype=np.int64) for o, _ in rec.history] + \
                      [np.asarray(rec.obs, dtype=np.int64)]
            act_seq = [int(a) for _, a in rec.history] + [0]
            seqs.append((
                np.stack(obs_seq, axis=0),
                np.asarray(act_seq, dtype=np.int64),
            ))

        max_T = max(o.shape[0] for o, _ in seqs)
        B = len(seqs)
        obs_hist = np.zeros((B, max_T, _N_OBS_VARS), dtype=np.int64)
        act_hist = np.zeros((B, max_T), dtype=np.int64)
        seq_lens = np.zeros(B, dtype=np.int64)
        for i, (o, a) in enumerate(seqs):
            T = o.shape[0]
            # LEFT-pad so the final timestep is always at index max_T - 1 ->
            # the GRU's "final hidden" is built from the most recent step.
            obs_hist[i, max_T - T:] = o
            act_hist[i, max_T - T:] = a
            seq_lens[i] = T

        return (
            torch.from_numpy(obs_hist),
            torch.from_numpy(act_hist),
            torch.from_numpy(seq_lens),
        )

    # =========================================================== held-out NLL

    def _heldout_nll(self) -> float:
        """Compute per-step NLL on the held-out 5K slice (mean across
        ensemble members). Uses a single-step "history" (just the (obs, action)
        being predicted from).
        """
        assert self.agent is not None
        if self._heldout_obs is None or len(self._heldout_obs) == 0:
            return float("nan")
        # Sample up to 1000 rows for speed.
        n_eval = min(1000, len(self._heldout_obs))
        idx = self._train_rng.integers(0, len(self._heldout_obs), size=n_eval)
        obs = torch.from_numpy(self._heldout_obs[idx]).long().unsqueeze(1)  # (N, 1, V)
        act = torch.from_numpy(self._heldout_act[idx]).long().unsqueeze(1)  # (N, 1)
        nxt = torch.from_numpy(self._heldout_next[idx]).long()              # (N, V)

        model = self.agent.model
        losses = []
        with torch.no_grad():
            for k in range(self.model_cfg.n_members):
                loss = model.compute_loss(obs, act, nxt, member_idx=k)
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else float("nan")

    # ================================================================ evaluation

    def _run_evaluation(self, outer_iter: int) -> None:
        """Run the deterministic eval pool through ``self.agent`` and write
        a checkpoint row (with ``outer_iter`` set).

        We re-implement the per-episode loop (rather than using the existing
        ``Evaluator``) so we can reset the MBPOAgent's per-episode history
        between episodes and pass the proper ``history`` arg.
        """
        assert self.agent is not None and self.logger is not None
        eval_pool_path = self.cfg.out_dir / "eval_pool.json"
        eval_pool = json.loads(eval_pool_path.read_text())

        adapter = _EvalAgentAdapter(self.agent)

        # Use the existing Evaluator: build a fresh one each call wrapping
        # our adapter. The adapter's history reset is handled by patching
        # Evaluator to call ``adapter.reset_episode()`` between episodes via
        # a thin wrapper.
        logger = self.logger
        env_factory = self.env_factory

        # Implement the per-episode loop manually so we can reset history
        # cleanly between episodes (Evaluator can't do that for us).
        env = env_factory()
        rewards: list[float] = []
        lengths: list[int] = []
        terminals: list[str] = []
        clamp_counts: list[int] = []
        action_hist = np.zeros(_N_ACTIONS, dtype=np.int64)
        action_hist_sxa: dict[str, np.ndarray] = {}
        soc_dwell = np.zeros(State.NUM_SOC, dtype=np.int64)
        soc_trans = np.zeros((State.NUM_SOC, State.NUM_SOC), dtype=np.int64)

        for ep_idx, init_seed in enumerate(eval_pool):
            adapter.reset_episode()
            obs, info_reset = env.reset(seed=int(init_seed))
            prev_soc = int(info_reset["soc"])
            num_abn_now = int(info_reset["num_abnormal_vitals"])
            ep_return = 0.0
            ep_len = 0
            ep_clamped = 0
            while True:
                a = adapter.act(obs, eval_mode=True)
                key = f"soc_{prev_soc}_abn_{num_abn_now}"
                if key not in action_hist_sxa:
                    action_hist_sxa[key] = np.zeros(_N_ACTIONS, dtype=np.int64)
                action_hist_sxa[key][a] += 1
                action_hist[a] += 1
                soc_dwell[prev_soc] += 1

                obs, raw_reward, terminated, truncated, info = env.step(int(a))
                ep_return += raw_reward
                ep_len += 1
                ep_clamped += int(info["clamped"])
                new_soc = int(info["soc"])
                soc_trans[prev_soc, new_soc] += 1
                prev_soc = new_soc
                num_abn_now = int(info["num_abnormal_vitals"])

                if terminated or truncated:
                    rewards.append(float(ep_return))
                    lengths.append(int(ep_len))
                    terminals.append(str(info["terminal_reason"]))
                    clamp_counts.append(int(ep_clamped))
                    break

        n = len(rewards)
        total_steps = sum(lengths) if lengths else 0
        n_term = {"discharge": 0, "death": 0, "timeout": 0}
        for t in terminals:
            n_term[t] = n_term.get(t, 0) + 1

        aggregates: dict[str, Any] = {
            "algo": self.algo_name,
            "outer_iter": int(outer_iter),
            "n_episodes": n,
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
            "ep_length_mean": float(np.mean(lengths)) if lengths else 0.0,
            "mortality_rate": n_term["death"] / max(n, 1),
            "discharge_rate": n_term["discharge"] / max(n, 1),
            "timeout_rate": n_term["timeout"] / max(n, 1),
            "clamp_rate": float(sum(clamp_counts) / max(total_steps, 1)),
            "action_hist": action_hist.tolist(),
            "soc_dwell_fractions": (soc_dwell / max(soc_dwell.sum(), 1)).tolist(),
            "soc_transition_counts": soc_trans.tolist(),
            "action_hist_by_soc_x_abnormal": {
                k: v.tolist() for k, v in action_hist_sxa.items()
            },
        }
        # ``step`` field in the checkpoint == outer_iter so the existing
        # aggregator's step-based grouping still works.
        logger.log_checkpoint(step=int(outer_iter), aggregates=aggregates)


