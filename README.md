# Reinforcement Learning for Clinical Site-of-Care Triage in a Sepsis Simulator

> **Stanford CS224R final project (2026).** Team: Liane Ozoemelam, Saimai Lau, Yun Dong.
>
> We study whether reinforcement learning can learn acuity-appropriate **site-of-care (SOC)
> triage** for sepsis patients — deciding both *where* a patient is treated (asynchronous →
> ambulatory → facility → ICU) and *how* (antibiotics / ventilation / vasopressors) — under
> partial observability, and we isolate what limits success: the algorithm, the reward, or the
> simulator.

### Lineage and credit

This project builds on two prior efforts:

- **Oberst & Sontag (ICML 2019)**, *Counterfactual Off-Policy Evaluation with Gumbel-Max
  Structural Causal Models* — the original Gumbel-Max SCM sepsis simulator, whose discrete
  transition dynamics were learned from the MIMIC-III critical-care database. The original
  README is preserved at the bottom of this file.
- **"From Acuity to Allocation: Learning Site-of-Care Decisions with RL"** (Kevin Chen, Liane
  Ozoemelam, Tanvi Thoria; Stanford CS234, 2024–25) — the prior course project that first added
  site-of-care structure to the simulator. The present CS224R project re-scopes that work around
  an explicit triage action space and a systematic algorithm × reward study.

---

## Overview

We extend the Gumbel-Max SCM simulator into a **triage-focused POMDP** in which the site of care
is both a state variable and an action dimension, then run a controlled study across **ten
algorithms** and **five reward variants** to attribute where the difficulty in learning good
triage actually lies.

**Headline finding.** Offline IQL-KL-F is the strongest method (13.6% mortality, 42.4%
discharge, 0% infeasible actions); offline methods beat online ones; and a lightweight
feasibility penalty drives infeasible actions to zero. Critically, the *original* reward ranks
best for nearly every algorithm — reward shaping changes behavior but does not improve outcomes —
which, together with offline RL's dominance, localizes the bottleneck to the **hand-specified
simulator dynamics** rather than the algorithm or reward. See the final report for full results.

---

## Environment (`sepsisSimDiabetes/`)

A POMDP extension of the Gumbel-Max SCM sepsis simulator. The simulator was simplified from the
prior project (removing Hospital-at-Home, resource-capacity variables, and noninvasive
ventilation) to a clean triage core.

- **State** (`State.py`): four discretized vitals (heart rate, systolic BP, oxygen saturation,
  glucose), three binary treatment flags (antibiotics, vasopressors, ventilation), and the
  current site of care `{ASYNC, AMBULATORY, FACILITY, ICU}`; a hidden diabetes variable modulates
  dynamics and is never observed.
- **Action** (`Action.py`): a joint `(site-of-care × treatments)` choice, giving
  **32 actions = 4_SOC × 2³_treatment**. Per-SOC **feasibility constraints** determine which
  treatments are available at each site (intensive interventions are unavailable below the ICU);
  infeasible action components are clamped at execution.
- **Partial observability** (`MDP.py`): the observation is a SOC-dependent masked subset of the
  true state — lower sites of care reveal fewer vitals (masked entries become a `-1` sentinel).
- **Reward variants** (`reward_variants.py`): a single parameterized reward with five variants
  (`reward0`–`reward4`); `reward0` is the original baseline (±10,000 terminals + dense shaping).
  Variants reduce the terminal scale, strengthen treatment penalties, add a severity-aware ICU
  penalty, or add per-site resource costs. `MDP.calculateReward` delegates here (default
  `reward0`).

---

## Algorithms

The study spans four algorithm families. **Implemented in this repository:**

- **Online** — DQN, PPO (`phase1_ppo_dqn/`); SAC, SAC-KL-F (feasibility penalty), SAC-KL-PPO
  (KL anchor to a frozen PPO reference) (`phase2_sac/`).
- **Offline** — IQL, IQL-KL-F (`phase3_iql/`), trained on a mixed dataset assembled from the
  online policies' trajectories.
- **Model-based (exploratory)** — MBPO + MCTS planning: a recurrent ensemble dynamics model with
  PUCT-guided tree search at decision time (`phase4_mbpo_mcts/`).
- **References** — RandomAgent and NoOpAgent (`triage_rl/agents/`).

The shared **feasibility regularizer** (`L_feas = β·E_s[Σ_{a∉F(s)} π(a|s)]`) is used by SAC-KL-F
and IQL-KL-F.

> Three additional baselines that appear in the full comparison and the report — a rule-based
> **Heuristic**, **Double DQN**, and **FactPPO** — were contributed by Liane Ozoemelam; their
> training code lives in her workspace, while their evaluation outputs are consumed by the
> plotting scripts in `analysis/`.

---

## Repository Structure

```
sepsisSimDiabetes/      # the POMDP simulator
  State.py, Action.py, MDP.py, reward_variants.py, DataGenerator.py
triage_rl/              # shared harness: config, env wrapper, evaluator, logger,
  agents/ (base, random, noop), trainers/ (off_policy, on_policy, offline)
phase1_ppo_dqn/         # DQN, PPO        (agents/, train.py, modal_app.py, presets.py)
phase2_sac/             # SAC + variants  (agents/, feasibility.py, train.py, modal_app.py)
phase3_iql/             # IQL + IQL-KL-F  (agents/, dataset.py, reward-variant rebuild, train.py)
phase4_mbpo_mcts/       # MBPO + MCTS     (agents/, trainers/outer_loop.py, train.py)
analysis/               # aggregation + report/poster figure generators
evaluation/             # reward-versioned eval aggregation + trajectory diagnostics
tests/                  # pytest suite
results/                # aggregated parquets, figures, eval_checkpoints (see note below)
data/                   # original learned MDP parameters (Oberst & Sontag)
pymdptoolbox/, mdptoolboxSrc/   # vendored MDP toolbox (original)
```

**Results convention.** Aggregated parquets, figures, and per-seed `eval_checkpoints.jsonl` are
tracked; model checkpoints, eval trajectories, per-seed training internals, and findings/writeup
docs are kept local (regenerable from the Modal volume).

---

## Running

All Python is run inside the `clinical-triage` conda environment. On machines with a polluting
`PYTHONPATH` (e.g., ROS2), prefix commands with `PYTHONNOUSERSITE=1 PYTHONPATH=`:

```bash
# tests
PYTHONNOUSERSITE=1 PYTHONPATH= conda run -n clinical-triage python -m pytest tests/ -q

# train one (algo, seed) locally (e.g. SAC, smoke preset)
PYTHONNOUSERSITE=1 PYTHONPATH= conda run -n clinical-triage \
  python -m phase2_sac.train --algo sac --preset smoke --seed 0

# scale out on Modal (one container per algo×seed; A100)
PYTHONNOUSERSITE=1 PYTHONPATH= conda run -n clinical-triage \
  modal run phase3_iql/modal_app.py --preset standard --algos iql,iql_kl_f
```

Each phase exposes a `train.py` CLI (`--algo`, `--preset {smoke,standard}`, `--seed`) and a
`modal_app.py` for distributed runs. Figures are regenerated from `analysis/` and `evaluation/`.

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
