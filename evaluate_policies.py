import numpy as np
import torch

from sepsisSimDiabetes.MDP import MDP
from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State

from dqn import DQN          
from ppo import Model


DEVICE = "cpu"

DQN_WEIGHTS = "results/dqn_weights1.pt"
PPO_WEIGHTS = "results/ppo_weights.pt"


#get indices of feasible actions
def feasible_action_indices(mdp: MDP):
    feas = []
    for a_idx in range(Action.NUM_ACTIONS_TOTAL):
        act = Action(action_idx=a_idx)
        if not mdp.soc_feasibility(act.soc):
            continue
        if not mdp.treatment_feasibility(act):
            continue
        feas.append(a_idx)
    if len(feas) == 0:
        feas = [0] 
    return np.array(feas, dtype=np.int64)


#use greedy dqn here for evaluation simplicty
@torch.no_grad()
def choose_action_greedy_dqn(mdp: MDP, dqn_model: torch.nn.Module):

    obs = mdp.get_observation().astype(np.float32)


    x = torch.tensor(obs, dtype=torch.float32, device = DEVICE).unsqueeze(0)

    q = dqn_model(x).squeeze(0).cpu().numpy() 

    feas = feasible_action_indices(mdp)
    a = int( feas[np.argmax(q[feas]) ] )

    return Action(action_idx=a)

#for ppo we do stochastic 
@torch.no_grad()
def choose_action_stochastic_ppo(mdp: MDP, ppo_model: torch.nn.Module):
    obs = mdp.get_observation().astype(np.float32)
    x = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    dist, _ = ppo_model.get_distribution(x)
    probs = dist.probs.squeeze(0).cpu().numpy()

    feas = feasible_action_indices(mdp)
    p = probs[feas].astype(np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    a = int(np.random.choice(feas, p=p))
    return Action(action_idx=a)


#same for random
def choose_action_random_feasible(mdp: MDP):
    feas = feasible_action_indices(mdp)
    a = int(np.random.choice(feas))
    return Action(action_idx=a)


#rollout episdoes for max 200 steps
def rollout_episode(policy_name: str, dqn_model=None, ppo_model=None, max_steps=200, p_diabetes=0.2):
    mdp = MDP(init_state_idx=None, policy_array=None, policy_idx_type="obs", p_diabetes=p_diabetes)
    mdp.state = mdp.get_new_state()

    total_reward = 0.0
    steps = 0

    #from the previous definitions here
    action_counts = {"abx": 0, "vent": 0, "vaso": 0}

    soc_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    soc_changes = 0

    prev_soc = getattr(mdp.state, "soc_state", None)

    for t in range(max_steps):
        # count the occupancy in each of the site of care
        cur_soc = getattr(mdp.state, "soc_state", None)

        if cur_soc is not None:

            soc_counts[int(cur_soc)] = soc_counts.get(int(cur_soc), 0) + 1

            if prev_soc is not None and int(cur_soc) != int(prev_soc):

                soc_changes += 1

            prev_soc = cur_soc

        # choose action
        if policy_name == "dqn":

            action = choose_action_greedy_dqn(mdp, dqn_model)
        elif policy_name == "ppo":

            action = choose_action_stochastic_ppo(mdp, ppo_model)
        elif policy_name == "random":

            action = choose_action_random_feasible(mdp)
        else:
            return None

        if getattr(action, "antibiotic", 0) == 1:
            action_counts["abx"] += 1
        if getattr(action, "ventilation", 0) == 1 or getattr(action, "noninvasive ventilation", 0) == 1:
            action_counts["vent"] += 1
        if getattr(action, "vasopressors", 0) == 1:
            action_counts["vaso"] += 1

        r = float(mdp.transition(action))

        total_reward += r
        steps += 1

        if mdp.state.check_absorbing_state():

            break

    # terminal state
    num_abn = mdp.state.get_num_abnormal()
    absorbing = mdp.state.check_absorbing_state()

    is_death = (absorbing and num_abn >= 3)
    is_discharge = (absorbing and num_abn < 3)

    #factors we want to track
    return {

        "return": total_reward,
        "len": steps,
        "death": int(is_death),
        "discharge": int(is_discharge),
        "soc_changes": soc_changes,
        "abx_steps": action_counts["abx"],
        "vent_steps": action_counts["vent"],
        "vaso_steps": action_counts["vaso"],

        "soc0_steps": soc_counts.get(0, 0),
        "soc1_steps": soc_counts.get(1, 0),
        "soc2_steps": soc_counts.get(2, 0),
        "soc3_steps": soc_counts.get(3, 0),
        "soc4_steps": soc_counts.get(4, 0),
    }


def evaluate(policy_name: str, N=200, max_steps=200, p_diabetes=0.2, dqn_model=None, ppo_model=None):

    rows = []
    for _ in range(N):
        rows.append(
            rollout_episode(policy_name, dqn_model=dqn_model, ppo_model=ppo_model, max_steps=max_steps, p_diabetes=p_diabetes)
        )

    # aggregate
    def mean(key): return float(np.mean([r[key] for r in rows]))
    def std(key): return float(np.std([r[key] for r in rows]))

    summary = {
        "policy": policy_name,
        "N": N,
        "avg_return": mean("return"),
        "avg_len": mean("len"),
        "death_rate": mean("death"),
        "discharge_rate": mean("discharge"),
        "avg_soc_changes": mean("soc_changes"),
        "avg_abx_steps": mean("abx_steps"),
        "avg_vent_steps": mean("vent_steps"),
        "avg_vaso_steps": mean("vaso_steps"),
    }
    return summary


def main():

    obs_dim = State.NUM_STATE_VARS
    n_actions = Action.NUM_ACTIONS_TOTAL

    dqn_model = DQN(obs_dim, 128, n_actions).to(DEVICE)
    dqn_model.load_state_dict(torch.load(DQN_WEIGHTS, map_location=DEVICE))
    dqn_model.eval()

    ppo_model = Model(obs_dim, n_actions).to(DEVICE)
    ppo_model.load_state_dict(torch.load(PPO_WEIGHTS, map_location=DEVICE))
    ppo_model.eval()

    N = 200
    T = 200

    summaries = []
    summaries.append(evaluate("random", N=N, max_steps=T, p_diabetes=0.2))
    summaries.append(evaluate("dqn", N=N, max_steps=T, p_diabetes=0.2, dqn_model=dqn_model))
    summaries.append(evaluate("ppo", N=N, max_steps=T, p_diabetes=0.2, ppo_model=ppo_model))

    print("Policy | Death% | Discharge% | AvgReturn | AvgLen | Site of Care Changes | Anti | Vent | Vaso")
    for s in summaries:
        print(
            f"{s['policy']:>6} | "
            f"{100*s['death_rate']:6.1f} | "
            f"{100*s['discharge_rate']:9.1f} | "
            f"{s['avg_return']:9.1f} | "
            f"{s['avg_len']:6.1f} | "
            f"{s['avg_soc_changes']:5.1f} | "
            f"{s['avg_abx_steps']:4.1f} | "
            f"{s['avg_vent_steps']:5.1f} | "
            f"{s['avg_vaso_steps']:5.1f}"
        )


if __name__ == "__main__":
    main()