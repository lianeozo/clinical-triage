import numpy as np
from .State import State
from .Action import Action

'''
Includes blood glucose level proxy for diabetes: 0-3
    (lo2, lo1, normal, hi1, hi2); Any other than normal is "abnormal"
Initial distribution:
    [.05, .15, .6, .15, .05] for non-diabetics and [.01, .05, .15, .6, .19] for diabetics

Effect of vasopressors on if diabetic:
    raise blood pressure: normal -> hi w.p. .9, lo -> normal w.p. .5, lo -> hi w.p. .4
    raise blood glucose by 1 w./ p. .5

Effect of vasopressors off if diabetic:
    blood pressure falls by 1 w.p. .05 instead of .1
    glucose does not fall - apply fluctuations below instead

Fluctuation in blood glucose levels (IV/insulin therapy are not possible actions):
    fluctuate w.p. .3 if diabetic
    fluctuate w.p. .1 if non-diabetic
Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4530321/

Additional fluctuation regardless of other changes
This order is applied:
    antibiotics, ventilation, non-invasive ventilation, vasopressors, fluctuations
'''

# ----------------------------------------------------
# SITE-OF-CARE SPECS
# ----------------------------------------------------

SOC_STATE_TO_DICT_VARS = {
    'o_doc': 'outp_doc_state',
    'o_nurse': 'outp_nurse_state',
    'a_doc': 'acute_doc_state',
    'a_nurse': 'acute_nurse_state',
    'a_beds': 'acute_bed_state',
    'i_doc': 'icu_doc_state',
    'i_nurse': 'icu_nurse_state',
    'i_beds': 'icu_bed_state'
}

SOC_CAPACITY_PRESSURE = {
    State.ASYNC: {
        # One patient is a negligible load for async care
        'o_doc': 0.005,
        'o_nurse': 0.005,
        # Does not affect acute/ICU
        'a_doc': 0.0,
        'a_nurse': 0.0,
        'a_beds': 0.0,
        'i_doc': 0.0,
        'i_nurse': 0.0,
        'i_beds': 0.0
    },
    
    State.AMBULATORY: {
        # Higher load than async
        'o_doc': 0.02,
        'o_nurse': 0.02,
        # Does not affect acute/ICU
        'a_doc': 0.0,
        'a_nurse': 0.0,
        'a_beds': 0.0,
        'i_doc': 0.0,
        'i_nurse': 0.0,
        'i_beds': 0.0                    
    },
    
    State.HAH: {
        # Does not affect outpatient
        'o_doc': 0.0,
        'o_nurse': 0.0,
        # Affects acute but not ICU
        'a_doc': 0.05,
        'a_nurse': 0.05,
        'a_beds': 0.0, # bed is in the home
        'i_doc': 0.0,
        'i_nurse': 0.0,
        'i_beds': 0.0
    },
    
    State.FACILITY: {
        # Does not affect outpatient
        'o_doc': 0.0,
        'o_nurse': 0.0,
        # Affects acute but not ICU
        'a_doc': 0.05,
        'a_nurse': 0.06, # More nurses needed than doctors on average
        'a_beds': 0.05,
        'i_doc': 0.0,
        'i_nurse': 0.0,
        'i_beds': 0.0
    },
    
    State.ICU: {
        # Does not affect outpatient
        'o_doc': 0.0,
        'o_nurse': 0.0,
        # Affects ICU but not acute
        'a_doc': 0.0,
        'a_nurse': 0.0, # More nurses needed than doctors on average
        'a_beds': 0.0,
        'i_doc': 0.1,
        'i_nurse': 0.15,
        'i_beds': 0.2 # Often much fewer ICU beds than inpatient
    }   
}

SOC_RESOURCE_REQUIREMENTS = {
    State.ASYNC: dict(o_doc=1,  o_nurse=0, a_doc=0, a_nurse=0, a_beds=0, i_doc=0, i_nurse=0, i_beds=0),
    State.AMBULATORY: dict(o_doc=1,  o_nurse=1, a_doc=0, a_nurse=0, a_beds=0, i_doc=0, i_nurse=0, i_beds=0),
    State.HAH: dict(o_doc=0,  o_nurse=0, a_doc=1, a_nurse=1, a_beds=0, i_doc=0, i_nurse=0, i_beds=0),
    State.FACILITY: dict(o_doc=0,  o_nurse=0, a_doc=1, a_nurse=1, a_beds=1, i_doc=0, i_nurse=0, i_beds=0),
    State.ICU: dict(o_doc=0,  o_nurse=0, a_doc=0, a_nurse=0, a_beds=0, i_doc=1, i_nurse=1, i_beds=1),
}

SOC_TREATMENT_FEASIBILITY = {
    State.ASYNC: {
        "ventilation": [0], # impossible
        "noninvasive ventilation": [0], # impossible 
        "vasopressors": [0], # impossible
        "antibiotic": [0,1] # possible
    },

    State.AMBULATORY: {
        "ventilation": [0],
        "noninvasive ventilation": [0],
        "vasopressors": [0],
        "antibiotic": [0,1]
    },

    State.HAH: {
        "ventilation": [0], 
        "noninvasive ventilation": [0,1], # only non-invasive but treatment as allowed
        "vasopressors": [0],
        "antibiotic": [0,1]
    },

    State.FACILITY: {
        "ventilation": [0,1],
        "vasopressors": [0,1],
        "noninvasive ventilation": [0,1], # both non-invasive and invasive allowed
        "antibiotic": [0,1]
    },

    State.ICU: {
        "ventilation": [0,1],
        "noninvasive ventilation": [0,1], # both non-invasive and invasive allowed
        "vasopressors": [0,1],
        "antibiotic": [0,1]
    }
}

VITAL_MASK_PROBS = {                 
    State.ASYNC: [0.4,  0.4,   0.4, 0.2], # hr, sysbp, percoxyg, glucose
    State.AMBULATORY: [1.0,  1.0,   1.0, 0.6],
    State.HAH: [0.8,  0.8,   0.8, 0.5],
    State.FACILITY: [1.0,  1.0,   1.0, 0.9],
    State.ICU: [1.0,  1.0,   1.0, 1.0],
}


class MDP(object):

    def __init__(self, init_state_idx=None, init_state_idx_type='obs',
            policy_array=None, policy_idx_type='obs', p_diabetes=0.2):
        '''
        initialize the simulator
        '''
        assert p_diabetes >= 0 and p_diabetes <= 1, \
                "Invalid p_diabetes: {}".format(p_diabetes)
        assert policy_idx_type in ['obs', 'full', 'proj_obs']

        # Check the policy dimensions (states x actions)
        if policy_array is not None and (not callable(policy_array))::
            assert policy_array.shape[1] == Action.NUM_ACTIONS_TOTAL
            if policy_idx_type == 'obs':
                assert policy_array.shape[0] == State.NUM_OBS_STATES
            elif policy_idx_type == 'full':
                assert policy_array.shape[0] == \
                        State.NUM_HID_STATES * State.NUM_OBS_STATES
            elif policy_idx_type == 'proj_obs':
                assert policy_array.shape[0] == State.NUM_PROJ_OBS_STATES

        # p_diabetes is used to generate random state if init_state is None
        self.p_diabetes = p_diabetes
        self.state = None

        # Only need to use init_state_idx_type if you are providing a state_idx!
        self.state = self.get_new_state(init_state_idx, init_state_idx_type)

        self.policy_array = policy_array
        self.policy_idx_type = policy_idx_type  # Used for mapping the policy to actions

    def get_new_state(self, state_idx = None, idx_type = 'obs', diabetic_idx = None):
        '''
        use to start MDP over.  A few options:

        Full specification:
        1. Provide state_idx with idx_type = 'obs' + diabetic_idx
        2. Provide state_idx with idx_type = 'full', diabetic_idx is ignored
        3. Provide state_idx with idx_type = 'proj_obs' + diabetic_idx*

        * This option will set glucose to a normal level

        Random specification
        4. State_idx, no diabetic_idx: Latter will be generated
        5. No state_idx, no diabetic_idx:  Completely random
        6. No state_idx, diabetic_idx given:  Random conditional on diabetes
        '''
        assert idx_type in ['obs', 'full', 'proj_obs']
        option = None
        if state_idx is not None:
            if idx_type == 'obs' and diabetic_idx is not None:
                option = 'spec_obs'
            elif idx_type == 'obs' and diabetic_idx is None:
                option = 'spec_obs_no_diab'
                diabetic_idx = np.random.binomial(1, self.p_diabetes)
            elif idx_type == 'full':
                option = 'spec_full'
            elif idx_type == 'proj_obs' and diabetic_idx is not None:
                option = 'spec_proj_obs'
        elif state_idx is None and diabetic_idx is None:
            option = 'random'
        elif state_idx is None and diabetic_idx is not None:
            option = 'random_cond_diab'

        assert option is not None, "Invalid specification of new state"

        if option in ['random', 'random_cond_diab'] :
            init_state = self.generate_random_state(diabetic_idx)
            # Do not start in death or discharge state
            while init_state.check_absorbing_state():
                init_state = self.generate_random_state(diabetic_idx)
        else:
            # Note that diabetic_idx will be ignored if idx_type = 'full'
            init_state = State(
                    state_idx=state_idx, idx_type=idx_type,
                    diabetic_idx=diabetic_idx)
        
        self.current_step = 0
        return init_state

    def generate_random_state(self, diabetic_idx=None):
        # Note that we will condition on diabetic idx if provided
        if diabetic_idx is None:
            diabetic_idx = np.random.binomial(1, self.p_diabetes)

        # ----------------------------------------------------
        # Vital bins
        # ----------------------------------------------------
        
        # hr and sys_bp w.p. [.25, .5, .25]
        hr_state = np.random.choice(np.arange(3), p=np.array([.25, .5, .25]))
        sysbp_state = np.random.choice(np.arange(3), p=np.array([.25, .5, .25]))
        # percoxyg w.p. [.2, .8]
        percoxyg_state = np.random.choice(np.arange(2), p=np.array([.2, .8]))
        if diabetic_idx == 0:
            glucose_state = np.random.choice(np.arange(5), \
                p=np.array([.05, .15, .6, .15, .05]))
        else:
            glucose_state = np.random.choice(np.arange(5), \
                p=np.array([.01, .05, .15, .6, .19]))

        # ----------------------------------------------------
        # Capacity bins
        # ----------------------------------------------------
        
        # Doctor & nurse capacity usually avail and zero capacity rare.
        # 0 - critical shortage, 1 - tight, 2 - moderate, 3 - largely available
        
        doc_p = np.array([.05, .20, .45, .30])
        nurse_p = np.array([.05, .20, .45, .30])
        bed_p = np.array([.10, .30, .40, .20])

        
        outp_doc_state = np.random.choice(np.arange(State.NUM_OUTP_DOC), p=doc_p)
        outp_nurse_state  = np.random.choice(np.arange(State.NUM_OUTP_NURSE), p=nurse_p)
        acute_doc_state = np.random.choice(np.arange(State.NUM_ACUTE_DOC), p=doc_p)
        acute_nurse_state = np.random.choice(np.arange(State.NUM_ACUTE_NURSE),p=nurse_p)
        acute_bed_state = np.random.choice(np.arange(State.NUM_ACUTE_BEDS), p=bed_p)
        icu_doc_state = np.random.choice(np.arange(State.NUM_ICU_DOC), p=doc_p)
        icu_nurse_state = np.random.choice(np.arange(State.NUM_ICU_NURSE), p=nurse_p)
        icu_bed_state = np.random.choice(np.arange(State.NUM_ICU_BEDS), p=bed_p)
        
        # ----------------------------------------------------
        # Site of Care bins (conditioned on patient severity)
        # ----------------------------------------------------
        
        severity = ((hr_state != 1) + (sysbp_state != 1) + (percoxyg_state != 1) + (glucose_state != 2))
        
        # Determine probability distribution of SOC depending on patient vitals
        if severity == 0:
            soc_probs = [0.5, 0.3, 0.15, 0.05, 0.0]
        elif severity == 1:
            soc_probs = [0.2, 0.4, 0.2, 0.18, 0.02]
        elif severity == 2:
            soc_probs = [0.05, 0.2, 0.2, 0.45, 0.10]
        elif severity >= 3:
            soc_probs = [0.0, 0.05, 0.10, 0.45, 0.40]
        
        soc_state = np.random.choice(np.arange(State.NUM_SOC), p=soc_probs)
        
        # -----------------------------------------------------
            
        antibiotic_state = 0
        vaso_state = 0
        vent_state = 0
        noninv_vent_state = 0
        num_switches_state = 0

        state_categs = [hr_state, sysbp_state, percoxyg_state,
                glucose_state, antibiotic_state, vaso_state, vent_state,
                soc_state, outp_doc_state, outp_nurse_state, acute_doc_state,
                acute_nurse_state, acute_bed_state, icu_doc_state, icu_nurse_state,
                icu_bed_state, noninv_vent_state, num_switches_state]

        return State(state_categs=state_categs, diabetic_idx=diabetic_idx)

    def update_capacity(self):
        soc = self.state.soc_state
        soc_pressures = SOC_CAPACITY_PRESSURE[soc]
        
         # Capacity updates according to patient
        for var, p in soc_pressures.items():
            if p > 0:
                state_var = SOC_STATE_TO_DICT_VARS[var]
                if np.random.uniform() < p:
                    cur_bin = getattr(self.state, state_var)
                    setattr(self.state, state_var, max(0, cur_bin - 1))
                
        for var, state_var in SOC_STATE_TO_DICT_VARS.items():
            cur_bin = getattr(self.state, state_var)
            change = np.random.choice([-1, 0, 1])
            setattr(self.state, state_var, int(np.clip(cur_bin + change, 0, 3)))
    
    
    def get_observation(self):
        soc = self.state.soc_state
        probs = VITAL_MASK_PROBS[soc]
        observations = self.state.get_state_vector().copy()
        
        for i, prob in enumerate(probs):
            if np.random.binomial(1, prob) == 0:
                observations[i] = -1
                
        return observations
        
    
    def transition_antibiotics_on(self):
        '''
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .5
        '''
        self.state.antibiotic_state = 1
        if self.state.hr_state == 2 and np.random.uniform(0,1) < 0.5:
            self.state.hr_state = 1
        if self.state.sysbp_state == 2 and np.random.uniform(0,1) < 0.5:
            self.state.sysbp_state = 1

    def transition_antibiotics_off(self):
        '''
        antibiotics state off
        if antibiotics was on: heart rate, sys bp: normal -> hi w.p. .1
        '''
        if self.state.antibiotic_state == 1:
            if self.state.hr_state == 1 and np.random.uniform(0,1) < 0.1:
                self.state.hr_state = 2
            if self.state.sysbp_state == 1 and np.random.uniform(0,1) < 0.1:
                self.state.sysbp_state = 2
            self.state.antibiotic_state = 0

    def transition_vent_on(self):
        '''
        ventilation state on
        percent oxygen: low -> normal w.p. .7
        '''
        self.state.vent_state = 1
        if self.state.percoxyg_state == 0 and np.random.uniform(0,1) < 0.7:
            self.state.percoxyg_state = 1

    def transition_vent_off(self):
        '''
        ventilation state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .1
        '''
        if self.state.vent_state == 1:
            if self.state.percoxyg_state == 1 and np.random.uniform(0,1) < 0.1:
                self.state.percoxyg_state = 0
            self.state.vent_state = 0

    def transition_noninv_vent_on(self):
        '''
        ventilation state on
        percent oxygen: low -> normal w.p. .75
        '''
        self.state.noninv_vent_state = 1
        
        if self.state.percoxyg_state == 0 and np.random.uniform(0,1) < 0.5:
            self.state.percoxyg_state = 1
            
    def transition_noninv_vent_off(self):
        '''
        ventilation state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .1
        '''
        if self.state.noninv_vent_state == 1:
            if self.state.percoxyg_state == 1 and np.random.uniform(0,1) < 0.2:
                self.state.percoxyg_state = 0
            self.state.noninv_vent_state = 0
    
    def transition_vaso_on(self):
        '''
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal, normal -> hi w.p. .7
        for diabetic:
            raise blood pressure: normal -> hi w.p. .9,
                lo -> normal w.p. .5, lo -> hi w.p. .4
            raise blood glucose by 1 w.p. .5
        '''
        self.state.vaso_state = 1
        if self.state.diabetic_idx == 0:
            if np.random.uniform(0,1) < 0.7:
                if self.state.sysbp_state == 0:
                    self.state.sysbp_state = 1
                elif self.state.sysbp_state == 1:
                    self.state.sysbp_state = 2
        else:
            if self.state.sysbp_state == 1:
                if np.random.uniform(0,1) < 0.9:
                    self.state.sysbp_state = 2
            elif self.state.sysbp_state == 0:
                up_prob = np.random.uniform(0,1)
                if up_prob < 0.5:
                    self.state.sysbp_state = 1
                elif up_prob < 0.9:
                    self.state.sysbp_state = 2
            if np.random.uniform(0,1) < 0.5:
                self.state.glucose_state = min(4, self.state.glucose_state + 1)

    def transition_vaso_off(self):
        '''
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .1
            for diabetics, blood pressure falls by 1 w.p. .05 instead of .1
        '''
        if self.state.vaso_state == 1:
            if self.state.diabetic_idx == 0:
                if np.random.uniform(0,1) < 0.1:
                    self.state.sysbp_state = max(0, self.state.sysbp_state - 1)
            else:
                if np.random.uniform(0,1) < 0.05:
                    self.state.sysbp_state = max(0, self.state.sysbp_state - 1)
            self.state.vaso_state = 0

    def transition_fluctuate(self, hr_fluctuate, sysbp_fluctuate, percoxyg_fluctuate, \
        glucose_fluctuate):
        '''
        all (non-treatment) states fluctuate +/- 1 w.p. .1
        exception: glucose flucuates +/- 1 w.p. .3 if diabetic
        '''
        if hr_fluctuate:
            hr_prob = np.random.uniform(0,1)
            if hr_prob < 0.1:
                self.state.hr_state = max(0, self.state.hr_state - 1)
            elif hr_prob < 0.2:
                self.state.hr_state = min(2, self.state.hr_state + 1)
        if sysbp_fluctuate:
            sysbp_prob = np.random.uniform(0,1)
            if sysbp_prob < 0.1:
                self.state.sysbp_state = max(0, self.state.sysbp_state - 1)
            elif sysbp_prob < 0.2:
                self.state.sysbp_state = min(2, self.state.sysbp_state + 1)
        if percoxyg_fluctuate:
            percoxyg_prob = np.random.uniform(0,1)
            if percoxyg_prob < 0.1:
                self.state.percoxyg_state = max(0, self.state.percoxyg_state - 1)
            elif percoxyg_prob < 0.2:
                self.state.percoxyg_state = min(1, self.state.percoxyg_state + 1)
        if glucose_fluctuate:
            glucose_prob = np.random.uniform(0,1)
            if self.state.diabetic_idx == 0:
                if glucose_prob < 0.1:
                    self.state.glucose_state = max(0, self.state.glucose_state - 1)
                elif glucose_prob < 0.2:
                    self.state.glucose_state = min(1, self.state.glucose_state + 1)
            else:
                if glucose_prob < 0.3:
                    self.state.glucose_state = max(0, self.state.glucose_state - 1)
                elif glucose_prob < 0.6:
                    self.state.glucose_state = min(4, self.state.glucose_state + 1)

    def calculateReward(self, self_state, action):
        reward = 0
        num_abnormal = self.state.get_num_abnormal()
        prev_abnormal = self_state.get_num_abnormal()
        
        #----------------------------------------------------
        # Absorbing States
        # ----------------------------------------------------
        
        # Penalize death
        if num_abnormal >= 3:
            reward -= 10_000
            return reward
        # Reward on discharge
        elif num_abnormal == 0 and not self.state.on_treatment():
            reward += 10_000
            return reward

        
        # ----------------------------------------------------
        # Feedback from patient vital sign trajectory
        # ----------------------------------------------------
        
        change_in_abnormal = prev_abnormal - num_abnormal
        reward += change_in_abnormal * 100 # reward +100/-100 (weighted) for improvement/decline
        
        # ----------------------------------------------------
        # Escalation + descalation cost
        # ----------------------------------------------------
        
        change_in_soc = self.state.soc_state - self_state.soc_state
        
        if change_in_soc != 0:
            self.state.num_switches_state = self_state.num_switches_state + 1
          
        # Penalize unnecessary escalation  
        if change_in_soc > 0 and num_abnormal == 0:
            reward -= 200 * change_in_soc
                
        # Penalize switches the patient's site of care too much
        if self.current_step >= 3:
            switch_ratio = self.state.num_switches_state / self.current_step
            if switch_ratio > 0.25:
                reward -= 150 * (switch_ratio - 0.25)
        
        
        # ----------------------------------------------------
        # Treatment Cost
        # ----------------------------------------------------
    
        if action.antibiotic == 1:
            reward -= 10
        if action.ventilation == 1:
            reward -= 60
        if action.noninv_ventilation == 1:
            reward -= 40
        if action.vasopressors == 1:
            reward -= 40

        return reward

    def transition(self, action):
        self.state = self.state.copy_state()
        start_state = self.state.copy_state()
        self.update_capacity()

        if action.antibiotic == 1:
            self.transition_antibiotics_on()
            hr_fluctuate = False
            sysbp_fluctuate = False
        elif self.state.antibiotic_state == 1:
            self.transition_antibiotics_off()
            hr_fluctuate = False
            sysbp_fluctuate = False
        else:
            hr_fluctuate = True
            sysbp_fluctuate = True

        if action.ventilation == 1:
            self.transition_vent_on()
            percoxyg_fluctuate = False
        elif self.state.vent_state == 1:
            self.transition_vent_off()
            percoxyg_fluctuate = False
        else:
            percoxyg_fluctuate = True

        glucose_fluctuate = True

        
        if action.noninv_ventilation == 1:
            self.transition_noninv_vent_on()
            percoxyg_fluctuate = False
        elif self.state.noninv_vent_state == 1:
            self.transition_noninv_vent_off()
            percoxyg_fluctuate = False
            
        if action.vasopressors == 1:
            self.transition_vaso_on()
            sysbp_fluctuate = False
            glucose_fluctuate = False
        elif self.state.vaso_state == 1:
            self.transition_vaso_off()
            sysbp_fluctuate = False

        self.transition_fluctuate(hr_fluctuate, sysbp_fluctuate, percoxyg_fluctuate, \
            glucose_fluctuate)
        
        self.state.soc_state = action.soc
        

        self.current_step += 1
        reward = self.calculateReward(start_state, action)
        return reward

    def soc_feasibility(self, soc):
        reqs = SOC_RESOURCE_REQUIREMENTS[soc]
        
        for req in reqs:
            state_var = SOC_STATE_TO_DICT_VARS[req]
            cur_bin = getattr(self.state, state_var)
            if cur_bin < reqs[req]:
                return False
        return True

    def treatment_feasibility(self, action):
        soc = self.state.soc_state
        rules = SOC_TREATMENT_FEASIBILITY[soc]

        if action.ventilation not in rules["ventilation"]:
            return False

        if action.vasopressors not in rules["vasopressors"]:
            return False

        if action.antibiotic not in rules["antibiotic"]:
            return False
        
        if action.noninv_ventilation not in rules["noninvasive ventilation"]:
            return False
        
        # Can't use both kinds of ventilation
        if action.ventilation == 1 and action.noninv_ventilation == 1:
            return False

        return True
        

    def select_actions(self):
        assert self.policy_array is not None

        #Making policy_array callable, takes obs as parameterized and then returns probabilities over the actions for PPO

        if callable(self.policy_array):
            obs = self.get_observation()
            probabilities = self.policy_array(obs)
            probabilities = np.asarray(probabilities, dtype = np.float64)

            #clip
            probabilities = np.clip(probabilities, 1e-9, 1.0)
            probabilities = probabilities / probabilities.sum()

            #random selection
            a = np.random.choice(Action.NUM_ACTIONS_TOTAL, p = probabilities)
            return Action(action_idx=int(a))

        #keeping the rest of the tabular behavior non PPO functionality
        state_idx = self.state.get_state_idx(self.policy_idx_type)
        probs = self.policy_array[state_idx]
        
        feasible_actions = []
        prob_feasible_actions = []
        
        for a in range(Action.NUM_ACTIONS_TOTAL):
            action = Action(action_idx=a)
            
            # capacity feasibility
            if not self.soc_feasibility(action.soc):
                continue

            # treatment feasibility
            if not self.treatment_feasibility(action):
                continue
        
            feasible_actions.append(a)
            prob_feasible_actions.append(probs[a])
            
        if len(feasible_actions) == 0:
                feasible_actions = [0]   # do nothing incase nothing is feasible
                prob_feasible_actions = [1.0]
        else: 
            prob_feasible_actions = np.array(prob_feasible_actions) 
            prob_feasible_actions = prob_feasible_actions / prob_feasible_actions.sum()
        
        
        
        aev_idx = np.random.choice(feasible_actions, p=prob_feasible_actions)
        return Action(action_idx = aev_idx)
