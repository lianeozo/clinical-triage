import gymnasium
import numpy as np
import pytest

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.env import Env


def test_reset_returns_obs_and_info():
    env = Env()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (State.NUM_STATE_VARS,)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_step_returns_five_tuple_and_info_keys():
    env = Env()
    env.reset(seed=0)
    out = env.step(0)
    assert len(out) == 5, "Gymnasium 5-tuple required"
    obs, reward, terminated, truncated, info = out
    assert isinstance(obs, np.ndarray) and obs.shape == (State.NUM_STATE_VARS,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    for k in ("agent_action", "executed_action", "clamped", "clamped_components",
              "terminal_reason", "raw_state", "num_abnormal_vitals", "soc"):
        assert k in info, f"missing info key: {k}"


def test_f4a_clamp_vent_at_async():
    """Vent at ASYNC should be infeasible → executed action has vent=0."""
    env = Env()
    env.reset(seed=0)
    # Force SOC=ASYNC: construct an action (antib=0, vent=1, vaso=0, soc=ASYNC)
    soc = State.ASYNC
    action_idx = 0 * 16 + 1 * 8 + 0 * 4 + soc   # vent=1 at ASYNC → infeasible
    # Place patient at ASYNC by stepping with soc=ASYNC action first
    env.mdp.state.soc_state = soc
    _, _, _, _, info = env.step(action_idx)
    assert info["clamped"] is True
    assert "ventilation" in info["clamped_components"]
    executed = Action(action_idx=info["executed_action"])
    assert executed.ventilation == 0


def test_step_reward_is_raw_unscaled():
    """Env returns raw reward from MDP.calculateReward (not scaled)."""
    env = Env()
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(0)
    # Raw rewards in this MDP are O(10) to O(10_000); a scaled-by-1e-4 reward would be O(1e-4)
    # to O(1). This is a weak check but catches the obvious mistake.
    assert abs(reward) >= 1.0 or reward == 0.0, f"reward {reward} suspiciously scaled"


def test_gymnasium_env_checker():
    """Passes Gymnasium's built-in env contract checker."""
    from gymnasium.utils.env_checker import check_env
    env = Env()
    # check_env raises on contract violations. skip_render_check because we don't render.
    check_env(env.unwrapped, skip_render_check=True)


def test_truncation_on_max_steps():
    """Reaching max_steps without absorbing → truncated=True, terminal_reason='timeout'."""
    env = Env(max_steps=2)
    env.reset(seed=12345)
    # Force a SOC that won't absorb quickly: ICU with non-treatment action 12 (soc=ICU)
    # Just step max_steps times with action 0 and look at the final step's flags.
    for _ in range(env.max_steps - 1):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            pytest.skip("absorbing state hit before max_steps; rerun with different seed")
    _, _, terminated, truncated, info = env.step(0)
    assert terminated is False, "should not terminate on max_steps cap"
    assert truncated is True
    assert info["terminal_reason"] == "timeout"
