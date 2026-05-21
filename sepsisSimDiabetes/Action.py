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
            "must specify either set of action strings or action index"
        if selected_actions is not None:
            if Action.ANTIBIOTIC_STRING in selected_actions:
                self.antibiotic = 1
            else:
                self.antibiotic = 0
            if Action.VENT_STRING in selected_actions:
                self.ventilation = 1
            else:
                self.ventilation = 0
            if Action.VASO_STRING in selected_actions:
                self.vasopressors = 1
            else:
                self.vasopressors = 0
            self.soc = selected_actions.get(Action.SOC_STRING, 0)

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

        Includes all four components: presence flags for the three binary
        treatments (only present when on) plus an explicit SOC key.
        """
        out = {Action.SOC_STRING: int(self.soc)}
        if self.antibiotic == 1:
            out[Action.ANTIBIOTIC_STRING] = 1
        if self.ventilation == 1:
            out[Action.VENT_STRING] = 1
        if self.vasopressors == 1:
            out[Action.VASO_STRING] = 1
        return out

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
