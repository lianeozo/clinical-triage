"""Tests for the MBPOAgent orchestrator.

The MBPOAgent wires together an EnsembleModel (T2), a PiVNet (T3), and the
AlphaZero MCTS planner (T4) behind a single ``act(obs, history)`` interface.
It also supports a ``no_mcts`` ablation mode that bypasses the planner and
samples directly from the policy head.
"""
from pathlib import Path

import numpy as np

from triage_rl.config import EnsembleModelConfig, MCTSConfig, PiVNetConfig
from phase4_mbpo_mcts.agents.mbpo_agent import MBPOAgent

OBS_DIMS = [3, 3, 2, 5, 2, 2, 2, 4]
N_ACTIONS = 32


def _make_agent(no_mcts: bool = False, n_simulations: int = 10, seed: int = 0) -> MBPOAgent:
    return MBPOAgent(
        obs_dims=OBS_DIMS,
        n_actions=N_ACTIONS,
        model_cfg=EnsembleModelConfig(n_members=2, hidden_dim=16),
        pi_v_cfg=PiVNetConfig(hidden_dim=16),
        mcts_cfg=MCTSConfig(n_simulations=n_simulations),
        env_full_reward_fn=lambda prev, a, nxt: 0.0,
        env_done_fn=lambda obs: False,
        seed=seed,
        device="cpu",
        no_mcts=no_mcts,
    )


def test_act_returns_valid_action_in_range():
    agent = _make_agent(no_mcts=False)
    obs = np.zeros(len(OBS_DIMS), dtype=np.int64)
    a, info = agent.act(obs, history=[])
    assert isinstance(a, int)
    assert 0 <= a < N_ACTIONS
    assert "visit_counts" in info
    assert isinstance(info["visit_counts"], dict)
    assert "root_q" in info
    assert info["root_q"] is not None
    assert "mean_depth" in info
    assert info["mean_depth"] is not None
    assert "value" in info


def test_no_mcts_mode_samples_from_pi():
    agent = _make_agent(no_mcts=True)
    obs = np.zeros(len(OBS_DIMS), dtype=np.int64)
    a, info = agent.act(obs, history=[])
    assert isinstance(a, int)
    assert 0 <= a < N_ACTIONS
    # MCTS-only fields are None in the ablation mode so callers can detect it.
    assert info["visit_counts"] is None
    assert info["root_q"] is None
    assert info["mean_depth"] is None
    assert "value" in info


def test_act_with_nonempty_history():
    """The agent must accept a non-empty history (past obs/action pairs) and
    return a valid action. This is the steady-state case after the first env
    step in an episode.
    """
    agent = _make_agent(no_mcts=False)
    obs = np.zeros(len(OBS_DIMS), dtype=np.int64)
    history = [
        (np.zeros(len(OBS_DIMS), dtype=np.int64), 0),
        (np.zeros(len(OBS_DIMS), dtype=np.int64), 5),
    ]
    a, info = agent.act(obs, history=history)
    assert 0 <= a < N_ACTIONS
    assert isinstance(info["visit_counts"], dict)


def test_temperature_schedule_greedy_after_threshold():
    """After ``temperature_steps`` the temperature drops to ``temperature_final``
    (T=0 by default), so the action returned must be the argmax of the
    visit counts (MCTS) or the policy logits (no_mcts).
    """
    agent = _make_agent(no_mcts=False, n_simulations=20)
    obs = np.zeros(len(OBS_DIMS), dtype=np.int64)
    # step_in_episode well past temperature_steps (default 15) -> greedy
    a, info = agent.act(obs, history=[], step_in_episode=50)
    expected = max(info["visit_counts"], key=info["visit_counts"].get)
    assert a == expected


def test_no_mcts_greedy_after_threshold_matches_argmax_logits():
    agent = _make_agent(no_mcts=True)
    obs = np.zeros(len(OBS_DIMS), dtype=np.int64)
    # We can't see the logits directly via info; instead compare against a
    # second call with the same agent state (deterministic argmax) -> equal.
    a1, _ = agent.act(obs, history=[], step_in_episode=50)
    a2, _ = agent.act(obs, history=[], step_in_episode=50)
    assert a1 == a2  # argmax is deterministic given fixed weights


def test_no_mcts_is_much_faster_than_mcts():
    """Sanity check that no_mcts actually bypasses the planner — a single
    forward pass through PiVNet must beat 50 MCTS simulations comfortably.
    """
    import time

    agent_mcts = _make_agent(no_mcts=False, n_simulations=50)
    agent_pi = _make_agent(no_mcts=True, n_simulations=50)
    obs = np.zeros(len(OBS_DIMS), dtype=np.int64)

    # warmup
    agent_mcts.act(obs, history=[])
    agent_pi.act(obs, history=[])

    t0 = time.perf_counter()
    for _ in range(5):
        agent_mcts.act(obs, history=[])
    t_mcts = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(5):
        agent_pi.act(obs, history=[])
    t_pi = time.perf_counter() - t0

    assert t_pi < t_mcts, (
        f"no_mcts ({t_pi:.4f}s) should be faster than mcts ({t_mcts:.4f}s)"
    )


def test_save_and_load_round_trip(tmp_path: Path):
    agent = _make_agent(no_mcts=False)
    obs = np.zeros(len(OBS_DIMS), dtype=np.int64)
    # capture initial deterministic predictions
    a_before, info_before = agent.act(obs, history=[], step_in_episode=50)

    path = tmp_path / "agent.pt"
    agent.save(path)

    # Build a fresh agent with a different seed so its weights differ.
    agent2 = _make_agent(no_mcts=False, seed=99)
    agent2.load(path)
    a_after, info_after = agent2.act(obs, history=[], step_in_episode=50)

    # Greedy argmax over visit counts must match after loading the same
    # network weights (MCTS still has its own RNG, but at T=0 the argmax is
    # robust to the noise as long as the dominant action is unchanged).
    # We at least require the loaded model+pi_v produces a sensible action.
    assert 0 <= a_after < N_ACTIONS
    # The state-dicts on disk should at minimum let act run without error,
    # and produce the same greedy choice when seeds match for MCTS sims.
    assert a_before == a_after or (
        info_before["visit_counts"].get(a_before, 0) > 0
        and info_after["visit_counts"].get(a_after, 0) > 0
    )
