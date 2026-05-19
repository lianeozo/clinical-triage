"""Gymnasium-compliant wrapper around the POMDP sepsis MDP.

F4a feasibility clamping: if the agent picks an action whose treatment
components are not available at the chosen SOC, those components are zeroed
before the MDP transitions, so no spurious treatment-cost penalty is charged.
Both the agent's chosen action and the executed (post-clamp) action are
reported via the info dict.
"""
from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.MDP import MDP, SOC_TREATMENT_FEASIBILITY
from sepsisSimDiabetes.State import State


class Env(gymnasium.Env):
    metadata = {"render_modes": []}

    # Observation: state vector of length NUM_STATE_VARS, dtype float32.
    # Masked vitals are replaced with -1 by MDP.get_observation. Global max bin
    # value across all dims is 4 (glucose); use a uniform Box for simplicity.
    observation_space = Box(low=-1.0, high=4.0,
                            shape=(State.NUM_STATE_VARS,), dtype=np.float32)
    action_space = Discrete(Action.NUM_ACTIONS_TOTAL)

    def __init__(self, p_diabetes: float = 0.2, max_steps: int = 100) -> None:
        super().__init__()
        self.p_diabetes = p_diabetes
        self.max_steps = max_steps
        self.mdp: MDP | None = None
        self._step_count = 0
        self._np_random: np.random.Generator | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # The underlying MDP uses module-level np.random.* calls, so seed the
        # global numpy RNG from our seeded generator to make rollouts reproducible
        # under the Gymnasium contract.
        global_seed = int(self.np_random.integers(0, 2**31 - 1))
        np.random.seed(global_seed)
        # If options has init_idx, the trainer/evaluator has already mapped it to a seed;
        # we just use the provided seed.
        self.mdp = MDP(init_state_idx=None, policy_array=None, p_diabetes=self.p_diabetes)
        self._step_count = 0
        obs = self.mdp.get_observation().astype(np.float32)
        info = {
            "soc": int(self.mdp.state.soc_state),
            "raw_state": self.mdp.state.get_state_vector().astype(np.float32),
            "num_abnormal_vitals": int(self.mdp.state.get_num_abnormal()),
        }
        return obs, info

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.mdp is not None, "must call reset() before step()"
        agent_action = Action(action_idx=int(action_idx))
        executed_action, clamped_components = self._clamp(agent_action)
        clamped = bool(clamped_components)

        raw_reward = float(self.mdp.transition(executed_action))
        self._step_count += 1

        absorbing = self.mdp.state.check_absorbing_state()
        # terminal_reason: discharge if absorbing AND 0 abnormal vitals; death if absorbing
        # with >=3 abnormal vitals; timeout if step cap hit without absorbing.
        if absorbing:
            num_abn = self.mdp.state.get_num_abnormal()
            terminal_reason = "death" if num_abn >= 3 else "discharge"
            terminated = True
            truncated = False
        elif self._step_count >= self.max_steps:
            terminal_reason = "timeout"
            terminated = False
            truncated = True
        else:
            terminal_reason = "none"
            terminated = False
            truncated = False

        obs = self.mdp.get_observation().astype(np.float32)
        info = {
            "agent_action": int(agent_action.get_action_idx()),
            "executed_action": int(executed_action.get_action_idx()),
            "clamped": clamped,
            "clamped_components": list(clamped_components),
            "terminal_reason": terminal_reason,
            "raw_state": self.mdp.state.get_state_vector().astype(np.float32),
            "num_abnormal_vitals": int(self.mdp.state.get_num_abnormal()),
            "soc": int(self.mdp.state.soc_state),
        }
        return obs, raw_reward, terminated, truncated, info

    def _clamp(self, agent_action: Action) -> tuple[Action, list[str]]:
        """Return (executed_action, list of zeroed component names) per F4a."""
        rules = SOC_TREATMENT_FEASIBILITY[agent_action.soc]
        antib = int(agent_action.antibiotic)
        vent = int(agent_action.ventilation)
        vaso = int(agent_action.vasopressors)
        clamped: list[str] = []
        if antib not in rules["antibiotic"]:
            antib = 0
            clamped.append("antibiotic")
        if vent not in rules["ventilation"]:
            vent = 0
            clamped.append("ventilation")
        if vaso not in rules["vasopressors"]:
            vaso = 0
            clamped.append("vasopressors")
        executed = Action(action_idx=int(agent_action.get_action_idx()))
        executed.antibiotic = antib
        executed.ventilation = vent
        executed.vasopressors = vaso
        return executed, clamped
