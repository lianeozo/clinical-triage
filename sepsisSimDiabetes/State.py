import numpy as np


'''
Includes blood glucose level proxy for diabetes: 0-3
    (lo2 - counts as abnormal, lo1, normal, hi1, hi2 - counts as abnormal)
Initial distribution:
    [.05, .15, .6, .15, .05] for non-diabetics and [.01, .05, .15, .6, .19] for diabetics
'''

class State(object):

    # Site-of-care categories
    ASYNC = 0
    AMBULATORY = 1
    FACILITY = 2
    ICU = 3

# ----------------------------------------------------
# NUMBER OF BINS PER STATE VARIABLE
# ----------------------------------------------------

    # NUMBER OF DISCRETIZED BINS PER STATE VARIABLE
    NUM_HR = 3
    NUM_SYSBP = 3
    NUM_OXYG = 2
    NUM_GLUC = 5
    NUM_ANTIB = 2
    NUM_VASO = 2
    NUM_VENT = 2
    NUM_SOC = 4

    NUM_PER_STATE = np.array([
        NUM_HR,
        NUM_SYSBP,
        NUM_OXYG,
        NUM_GLUC,
        NUM_ANTIB,
        NUM_VASO,
        NUM_VENT,
        NUM_SOC,
    ], dtype=int)

    NUM_STATE_VARS = len(NUM_PER_STATE)

    # Index of soc_state inside the state vector returned by get_state_vector().
    # Order: hr, sysbp, percoxyg, glucose, antibiotic, vaso, vent, soc → SOC is at index 7.
    SOC_IDX = 7

    NUM_OBS_STATES = int(np.prod(NUM_PER_STATE))
    NUM_HID_STATES = 2  # Binary value of diabetes
    NUM_PROJ_OBS_STATES = int(NUM_OBS_STATES / NUM_GLUC)  # Marginalizing over glucose
    NUM_FULL_STATES = int(NUM_OBS_STATES * NUM_HID_STATES)

    def __init__(self,
            state_idx = None, idx_type = 'obs',
            diabetic_idx = None, state_categs = None):

        assert state_idx is not None or state_categs is not None
        assert ((diabetic_idx is not None and diabetic_idx in [0, 1]) or
                (state_idx is not None and idx_type == 'full'))

        assert idx_type in ['obs', 'full', 'proj_obs']

        if state_idx is not None:
            self.set_state_by_idx(
                    state_idx, idx_type=idx_type, diabetic_idx=diabetic_idx)
        elif state_categs is not None:
            received = len(state_categs)
            assert received == self.NUM_STATE_VARS, f"must specify {self.NUM_STATE_VARS} state variables, but instead, got {received} "
            self.hr_state = state_categs[0]
            self.sysbp_state = state_categs[1]
            self.percoxyg_state = state_categs[2]
            self.glucose_state = state_categs[3]
            self.antibiotic_state = state_categs[4]
            self.vaso_state = state_categs[5]
            self.vent_state = state_categs[6]
            self.soc_state = state_categs[7]
            self.diabetic_idx = diabetic_idx

    def check_absorbing_state(self):
        num_abnormal = self.get_num_abnormal()
        if num_abnormal >= 3:
            return True
        elif num_abnormal == 0 and not self.on_treatment():
            return True
        return False

    def mixed_radix_idx_helper(self, mod_idx, term_base, num_bins_next):
        val = np.floor(mod_idx/term_base).astype(int)
        mod_idx %= term_base
        term_base /= num_bins_next
        return val, mod_idx, term_base

    def set_state_by_idx(self, state_idx, idx_type, diabetic_idx=None):
        """set_state_by_idx

        The state index is determined by using "bit" arithmetic, with the
        complication that not every state is binary

        :param state_idx: Given index
        :param idx_type: Index type, either observed, projected or
        full
        :param diabetic_idx: If full state index not given, this is required
        """
        if idx_type == 'obs':
            term_base = State.NUM_OBS_STATES/self.NUM_HR # Starts with heart rate
        elif idx_type == 'proj_obs':
            term_base = State.NUM_PROJ_OBS_STATES/self.NUM_HR
        elif idx_type == 'full':
            term_base = State.NUM_FULL_STATES/self.NUM_HID_STATES # Starts with diab

        # Start with the given state index
        mod_idx = state_idx

        if idx_type == 'full':
            self.diabetic_idx, mod_idx, term_base = self.mixed_radix_idx_helper(mod_idx, term_base, self.NUM_HR)
        else:
            assert diabetic_idx is not None
            self.diabetic_idx = diabetic_idx

        self.hr_state, mod_idx, term_base = self.mixed_radix_idx_helper(mod_idx, term_base, self.NUM_SYSBP)
        self.sysbp_state, mod_idx, term_base = self.mixed_radix_idx_helper(mod_idx, term_base, self.NUM_OXYG)
        self.percoxyg_state, mod_idx, term_base = self.mixed_radix_idx_helper(mod_idx, term_base, self.NUM_ANTIB if idx_type == 'proj_obs' else self.NUM_GLUC)

        if idx_type == 'proj_obs':
            self.glucose_state = 2
        else:
            self.glucose_state, mod_idx, term_base = self.mixed_radix_idx_helper(mod_idx, term_base, self.NUM_ANTIB)

        self.antibiotic_state, mod_idx, term_base = self.mixed_radix_idx_helper(mod_idx, term_base, self.NUM_VASO)
        self.vaso_state, mod_idx, term_base = self.mixed_radix_idx_helper(mod_idx, term_base, self.NUM_VENT)
        self.vent_state, mod_idx, term_base = self.mixed_radix_idx_helper(mod_idx, term_base, self.NUM_SOC)
        self.soc_state, mod_idx, term_base = self.mixed_radix_idx_helper(mod_idx, term_base, 1)

    def get_state_idx(self, idx_type='obs'):
        '''
        returns integer index of state: significance order as in categorical array
        '''
        if idx_type == 'obs':
            categ_num = np.array([3,3,2,5,2,2,2,4])
            state_categs = [
                    self.hr_state,
                    self.sysbp_state,
                    self.percoxyg_state,
                    self.glucose_state,
                    self.antibiotic_state,
                    self.vaso_state,
                    self.vent_state,
                    self.soc_state,
                    ]
        elif idx_type == 'proj_obs':
            categ_num = np.array([3,3,2,2,2,2,4])
            state_categs = [
                    self.hr_state,
                    self.sysbp_state,
                    self.percoxyg_state,
                    self.antibiotic_state,
                    self.vaso_state,
                    self.vent_state,
                    self.soc_state,
                    ]
        elif idx_type == 'full':
            categ_num = np.array([2,3,3,2,5,2,2,2,4])
            state_categs = [
                    self.diabetic_idx,
                    self.hr_state,
                    self.sysbp_state,
                    self.percoxyg_state,
                    self.glucose_state,
                    self.antibiotic_state,
                    self.vaso_state,
                    self.vent_state,
                    self.soc_state,
                    ]

        sum_idx = 0
        prev_base = 1
        for i in range(len(state_categs)):
            idx = len(state_categs) - 1 - i
            sum_idx += prev_base*state_categs[idx]
            prev_base *= categ_num[idx]
        return sum_idx

    def __eq__(self, other):
        '''
        override equals: two states equal if all internal states same
        '''
        return isinstance(other, self.__class__) and \
            self.hr_state == other.hr_state and \
            self.sysbp_state == other.sysbp_state and \
            self.percoxyg_state == other.percoxyg_state and \
            self.glucose_state == other.glucose_state and \
            self.antibiotic_state == other.antibiotic_state and \
            self.vaso_state == other.vaso_state and \
            self.vent_state == other.vent_state and \
            self.soc_state == other.soc_state

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(self.get_state_vector()))

    def get_num_abnormal(self):
        '''
        returns number of abnormal conditions
        '''
        num_abnormal = 0
        if self.hr_state != 1 and self.hr_state != -1:
            num_abnormal += 1
        if self.sysbp_state != 1 and self.sysbp_state != -1:
            num_abnormal += 1
        if self.percoxyg_state != 1 and self.percoxyg_state != -1:
            num_abnormal += 1
        if self.glucose_state != 2 and self.glucose_state != -1:
            num_abnormal += 1
        return num_abnormal

    def on_treatment(self):
        '''
        returns True iff any treatment active
        '''
        if self.antibiotic_state == 0 and \
            self.vaso_state == 0 and self.vent_state == 0:
            return False
        return True

    def on_antibiotics(self):
        '''
        returns True iff antibiotics active
        '''
        return self.antibiotic_state == 1

    def on_vasopressors(self):
        '''
        returns True iff vasopressors active
        '''
        return self.vaso_state == 1

    def on_ventilation(self):
        '''
        returns True iff ventilation active
        '''
        return self.vent_state == 1

    def copy_state(self):
        return State(state_categs = [
            self.hr_state,
            self.sysbp_state,
            self.percoxyg_state,
            self.glucose_state,
            self.antibiotic_state,
            self.vaso_state,
            self.vent_state,
            self.soc_state],
            diabetic_idx=self.diabetic_idx)

    def get_state_vector(self):
        return np.array([self.hr_state,
            self.sysbp_state,
            self.percoxyg_state,
            self.glucose_state,
            self.antibiotic_state,
            self.vaso_state,
            self.vent_state,
            self.soc_state]).astype(int)
