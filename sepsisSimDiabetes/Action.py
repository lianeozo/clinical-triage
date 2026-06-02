import numpy as np
from .State import State

class Action(object):
    NUM_PER_ACTION = np.array([
        State.NUM_ANTIB,
        State.NUM_VENT,
        State.NUM_VASO,
        State.NUM_SOC
    ], dtype=int)

    NUM_ACTIONS_TOTAL = int(np.prod(NUM_PER_ACTION))
    ANTIBIOTIC_STRING = "antibiotic"
    VENT_STRING = "ventilation"
    VASO_STRING = "vasopressors"
    SOC_STRING = "soc"
    ACTION_VEC_SIZE = 4

    PER_ACTION_LABELS = (
        ANTIBIOTIC_STRING,
        VENT_STRING,
        VASO_STRING,
        SOC_STRING,
    )

    def __init__(self, selected_actions = None, action_idx = None):
        assert (selected_actions is not None and action_idx is None) \
            or (selected_actions is None and action_idx is not None), \
            "must specify either selected_actions dict or action index"
        if selected_actions is not None:
            self.antibiotic = int(selected_actions.get(Action.ANTIBIOTIC_STRING, 0))
            self.ventilation = int(selected_actions.get(Action.VENT_STRING, 0))
            self.vasopressors = int(selected_actions.get(Action.VASO_STRING, 0))
            self.soc = int(selected_actions.get(Action.SOC_STRING, 0))

        else:
            mod_idx = action_idx
            term_base = Action.NUM_ACTIONS_TOTAL / 2  # 16
            self.antibiotic = np.floor(mod_idx / term_base).astype(int)
            mod_idx %= term_base
            term_base /= 2  # 8
            self.ventilation = np.floor(mod_idx / term_base).astype(int)
            mod_idx %= term_base
            term_base /= 2  # 4
            self.vasopressors = np.floor(mod_idx / term_base).astype(int)
            mod_idx %= term_base
            self.soc = int(mod_idx)  # 0-3

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.antibiotic == other.antibiotic and \
            self.ventilation == other.ventilation and \
            self.vasopressors == other.vasopressors and \
            self.soc == other.soc

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_action_idx(self):
        assert self.antibiotic in (0, 1)
        assert self.ventilation in (0, 1)
        assert self.vasopressors in (0, 1)
        assert self.soc in range(4)

        return (self.antibiotic * 16 +
                self.ventilation * 8 +
                self.vasopressors * 4 +
                self.soc)

    def __hash__(self):
        return self.get_action_idx()

    def get_selected_actions(self):
        """Return a dict suitable for round-trip via __init__(selected_actions=...).

        All four components are included with explicit integer values, so the
        round-trip Action(selected_actions=a.get_selected_actions()) preserves
        every field (including SOC).
        """
        return {
            Action.ANTIBIOTIC_STRING: self.antibiotic,
            Action.VENT_STRING: self.ventilation,
            Action.VASO_STRING: self.vasopressors,
            Action.SOC_STRING: self.soc,
        }

    def get_abbrev_string(self):
        '''
        AEV: antibiotics, ventilation, vasopressors + SOC level appended.
        e.g. "AE_SOC2"
        '''
        output_str = ''
        if self.antibiotic == 1:
            output_str += 'A'
        if self.ventilation == 1:
            output_str += 'E'
        if self.vasopressors == 1:
            output_str += 'V'
        output_str += f'_SOC{self.soc}'
        return output_str

    def get_action_vec(self):
        return np.array([
            [self.antibiotic],
            [self.ventilation],
            [self.vasopressors],
            [self.soc]
        ])
