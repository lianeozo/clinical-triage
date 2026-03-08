import numpy as np
from .State import State

class Action(object):
    
    NUM_PER_ACTION = np.array([
        State.NUM_ANTIB,
        State.NUM_VASO,
        State.NUM_VENT,
        State.NUM_NONINV_VENT,
        State.NUM_SOC
    ], dtype=int)

    NUM_ACTIONS_TOTAL = int(np.prod(NUM_PER_ACTION))
    ANTIBIOTIC_STRING = "antibiotic"
    VENT_STRING = "ventilation"
    NONINV_VENT_STRING = "noninvasive ventilation"
    VASO_STRING = "vasopressors"
    SOC_STRING = "soc"
    ACTION_VEC_SIZE = 5

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
            if Action.NONINV_VENT_STRING in selected_actions:
                self.noninv_ventilation = 1
            else:
                self.noninv_ventilation = 0
            self.soc = selected_actions.get(Action.SOC_STRING, 0)
            
        else:
            mod_idx = action_idx
            term_base = Action.NUM_ACTIONS_TOTAL/2 #gives 40
            self.antibiotic = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 2 # 20
            self.ventilation = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 2 # 10
            self.vasopressors = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 2  # 5
            self.noninv_ventilation = int(np.floor(mod_idx / term_base))
            mod_idx %= term_base
            self.soc = int(mod_idx) # something between 0 and 4

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.antibiotic == other.antibiotic and \
            self.ventilation == other.ventilation and \
            self.vasopressors == other.vasopressors and \
            self.noninv_ventilation == other.noninv_ventilation and \
            self.soc == other.soc

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_action_idx(self):
        assert self.antibiotic in (0, 1)
        assert self.ventilation in (0, 1)
        assert self.vasopressors in (0, 1)
        assert self.noninv_ventilation in (0,1)
        assert self.soc in range(5)
        
        return (self.antibiotic * 40 +
                self.ventilation * 20 +
                self.vasopressors * 10 +
                self.noninv_ventilation * 5 +
                self.soc)

    def __hash__(self):
        return self.get_action_idx()

    def get_selected_actions(self):
        selected_actions = set()
        if self.antibiotic == 1:
            selected_actions.add(Action.ANTIBIOTIC_STRING)
        if self.ventilation == 1:
            selected_actions.add(Action.VENT_STRING)
        if self.vasopressors == 1:
            selected_actions.add(Action.VASO_STRING)
        if self.noninv_ventilation == 1:
            selected_actions.add(Action.NONINV_VENT_STRING)
        return selected_actions

    def get_abbrev_string(self):
        '''
        AEV: antibiotics, ventilation, vasopressors, N = non-invasive ventilation,
        + SOC level appended to the end.
        
        e.g. "AEV_SOC2"
        '''
        output_str = ''
        if self.antibiotic == 1:
            output_str += 'A'
        if self.ventilation == 1:
            output_str += 'E'
        if self.vasopressors == 1:
            output_str += 'V'
        if self.noninv_ventilation == 1:
            output_str += 'N'
        output_str += f'_SOC{self.soc}'
        return output_str

    def get_action_vec(self):
        return np.array([
            [self.antibiotic], 
            [self.ventilation], 
            [self.vasopressors],
            [self.noninv_ventilation],
            [self.soc]
        ])