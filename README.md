# Clinical Site-of-Care Triage with a POMDP Extension of the Gumbel-Max SCM Sepsis Simulator

> **This repository extends the original Oberst & Sontag (ICML 2019) sepsis simulator into a POMDP framework for sequential site-of-care (SOC) triage decisions.**
> All modifications and additions beyond the original Oberst & Sontag codebase are the work of Liane Ozoemelam, Kevin Chen, and Tanvi Thoria (Stanford CS234, 2024–25).

---

## Overview of Our Extensions

The original simulator models a sepsis patient as a finite MDP: a clinician chooses among 8 treatment combinations (antibiotics / ventilation / vasopressors on or off), patient vitals evolve stochastically, and episodes terminate on discharge (+1) or death (−1). We preserve this core while extending it significantly along four axes.

### 1. Expanded State Space (`State.py`)

The original state encodes only patient vitals and treatment flags. We extend this to a **patient–system state space** that supports site-of-care decisions and resource constraints.

**Original state variables (unchanged):**
- Vitals bins: `hr_state` (0/1/2), `sysbp_state` (0/1/2), `percoxyg_state` (0/1)
- Glucose bin: `glucose_state` (0–4)
- Treatment flags: `antibiotic_state`, `vaso_state`, `vent_state`
- Hidden type: `diabetic_idx` (0/1)

**Added state variables (our modifications):**
- `soc_state`: current site of care — one of `{ASYNC, AMBULATORY, H@H, FACILITY, ICU}`
- Outpatient capacity: `outp_doc_state`, `outp_nurse_state` (4 levels each)
- Acute care capacity: `acute_doc_state`, `acute_nurse_state`, `acute_bed_state` (4 levels each)
- ICU capacity: `icu_doc_state`, `icu_nurse_state`, `icu_bed_state` (4 levels each)

Capacity variables evolve via a stochastic random walk that applies one roll per care tier (outpatient, acute, ICU) per timestep, rather than independently per variable, to preserve clinically meaningful within-tier coherence.

---

### 2. Redefined Action Space (`Action.py`)

**Original:** action = treatment combination (3-bit vector over antibiotics / vent / vasopressors → 8 actions).

**Our modification:** the agent's primary action is the **next site of care**:

```
at ∈ A := {ASYNC, AMBULATORY, H@H, FACILITY, ICU}
```

Treatment interventions are no longer directly chosen by the agent — they are determined by feasibility constraints imposed by the current site of care. For example, vasopressors and invasive ventilation are only available at FACILITY or ICU. This makes the action space 5-dimensional (one per SOC level) while preserving the original treatment transition logic.

---

### 3. POMDP Observation Function (`MDP.py`)

The original simulator is fully observable. We implement a **site-of-care-dependent observation function** that partially masks the latent state.

At each timestep, an observation `o_t` is generated as:

```
o_t = m(c_t) ⊙ x_t
```

where `m(c_t)` is a binary mask vector whose entries are drawn as:

```
m_{t,i} ~ Bernoulli(p_i(c_t))
```

The probability `p_i(c_t)` of observing each vital (`hr`, `sysbp`, `percoxyg`, `glucose`) depends on the current site of care. ICU admits full observability; ASYNC has the highest masking probability. Unobserved features are replaced with a `-1` sentinel in the observation vector passed to the RL agent.

---

### 4. Shaped Reward Function (`MDP.py`)

The original reward function issues ±1 terminal rewards only. We replace this with a shaped reward that reflects both patient outcomes and system resource behavior:

| Event | Reward |
|---|---|
| Death (≥3 abnormal vitals) | −10,000 |
| Discharge (all vitals normal, no treatment) | +10,000 |
| SOC escalation | −200 |
| ≥2 abnormal vitals without SOC escalation | −100 |
| Antibiotics administered | −10 |
| Ventilator used | −100 |
| Vasopressors used | −100 |

The shaping is designed to balance patient survival, appropriate escalation, and resource stewardship. Note: the low penalty on antibiotics (−10) relative to discharge reward (+10,000) leads PPO to exploit antibiotic administration as a low-cost vitals-improvement strategy — a known artifact of this reward design, not a training failure.

---

### 5. RL Agents

We train and evaluate three agents on the modified POMDP:

- **DQN** (`dqn.py`): 3-layer MLP, epsilon-greedy exploration with decay, experience replay buffer, Huber loss, soft target network updates (τ = 0.005), reward scaling by 1/10,000. Falls back to `random_select_action()` when greedy action is infeasible.
- **PPO** (`ppo.py`): Actor-critic with GAE advantage estimation, clipped surrogate objective. Feasibility of the selected action is enforced before environment step.

---

### 6. Trajectory Visualization (`render_trajectories.py`)

A rendering utility that steps through a full episode under DQN, PPO, or random policy and prints per-step vitals, SOC, capacity, treatments, reward, and terminal outcome. Capacity display is automatically filtered to the tier relevant to the current SOC.

---

## Repository Structure

```
.
├── sepsisSimDiabetes/
│   ├── State.py           # Extended state space (modified)
│   ├── Action.py          # SOC action space (modified)
│   ├── MDP.py             # POMDP transitions, observation fn, reward (modified)
│   └── DataGenerator.py   # Trajectory simulation (modified)
├── dqn.py                 # DQN agent (new)
├── ppo.py                 # PPO agent (new)
├── render_trajectories.py # Trajectory visualization (new)
├── learn_mdp_parameters.ipynb   # Original — learns MDP params from MIMIC-III
├── plots-main-paper.ipynb       # Original — replicates Oberst & Sontag figures
└── data/
    └── diab_txr_mats-replication.zip   # Original learned MDP parameters
```

Files not listed above are from the original Oberst & Sontag repository and are unmodified.

---

---

# Original README (Oberst & Sontag, ICML 2019)

## Counterfactual Off-Policy Evaluation with Gumbel-Max SCMs

### Overview

This repository contains all the code required to replicate the figures in the ICML 2019 paper. To do so from scratch, you will need to run the following steps:

- To generate the MDP parameters, run `learn_mdp_parameters.ipynb`, which will save the learned parameters in the `data` folder. This takes ~2 hours.
- Alternatively, you can use the parameters that are already learned — to do so, unzip `data/diab_txr_mats-replication.zip` locally (e.g., using `unzip diab_txr_mats-replication.zip`).
- To re-create the plots in the main paper, run `plots-main-paper.ipynb`. This assumes that you have `data/diab_txr_mats-replication.pkl`, by one of the methods above.
- To re-create the plots in the appendix, see the corresponding notebooks.

### Dependencies

This code was run using Python 3.7 in a `conda` environment. Running the following commands should cover all the dependencies of the code (e.g., installing `pandas` will install `numpy`, and so on):

```
conda install jupyter
conda install pandas
conda install seaborn
conda install tqdm
pip install pymdptoolbox
```

### Updated Simulator

As we receive suggestions for improving the realism of the sepsis simulator, we will collect them in the `sim-v2` branch of this repository, in case it is useful for others. The `master` branch will remain unchanged to facilitate reproduction of the original paper.

### Acknowledgements

First, we would like to thank Christina X Ji and [Fredrik D. Johansson](http://www.mit.edu/~fredrikj/) for their work on an earlier version of the sepsis simulator we use in this paper.

Second, for some of the code used in the posterior inference over Gumbel variables, we borrowed from Chris Maddison's blog post [here](https://cmaddis.github.io/gumbel-machinery).

Finally, in this repository (in `pymdptoolbox/`) we have the source code for the `pymdptoolbox` package from [sawcordwell/pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox), which is in turn based on the toolset described in `Chades I, Chapron G, Cros M-J, Garcia F & Sabbadin R (2014) 'MDPtoolbox: a multi-platform toolbox to solve stochastic dynamic programming problems', Ecography, vol. 37, no. 9, pp. 916–920, doi 10.1111/ecog.00888.` We reproduce it here because we needed to make a slight modification to the `mdp` class to bypass certain checks; in particular, it checks for whether or not the rows of the transition matrix sum to one, but can fail due to floating-point inaccuracies — we replace this check in the main code with an assertion using `np.allclose` instead of checking for strict equality.
