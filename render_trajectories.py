from sepsisSimDiabetes.MDP import MDP
from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from ppo import Model
from dqn import device
import torch
import os

def render_trajectory(model, device, model_type='dqn'):
    hr_map = {0: 'low', 1: 'normal', 2: 'high'}
    bp_map = {0: 'low', 1: 'normal', 2: 'high'}
    o2_map = {0: 'low', 1: 'normal'}
    glucose_map = {0: 'very low', 1: 'low', 2: 'normal', 3: 'high', 4: 'very high'}
    soc_map = {0: 'async/remote', 1: 'ambulatory', 2: 'H@H', 3: 'facility acute', 4: 'ICU'}
    cap_map = {0: 'critical shortage', 1: 'tight', 2: 'moderate', 3: 'largely available'}

    mdp = MDP(init_state_idx=None, policy_array=None, p_diabetes=0.2)
    state = mdp.state

    output = []
    output.append("   New Episode   ")
    output.append(f"Diabetic: {bool(state.diabetic_idx)}")
    output.append("")

    for t in range(14):
        # get observation and select action based on model type
        obs = torch.tensor(mdp.get_observation(), dtype=torch.float32, device=device).unsqueeze(0)

        #adding random trajectory functionality
        if model_type == 'random':
            action = mdp.random_select_action()
        else:
            obs = torch.tensor(mdp.get_observation(), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if model_type == 'dqn':
                    action_idx = model(obs).max(1).indices.item()
                else:  # 'ppo'
                    dist, _ = model.get_distribution(obs)
                    action_idx = dist.probs.argmax().item()
            action = Action(action_idx=action_idx)
        
        reward = mdp.transition(action)
        state = mdp.state
        soc = state.soc_state

        # show only the capacity vars relevant to the current site of care
        if soc in [State.ASYNC, State.AMBULATORY]:
            capacity_str = f"Outpatient docs: {cap_map[state.outp_doc_state]}, Outpatient nurses: {cap_map[state.outp_nurse_state]}"
        elif soc in [State.HAH, State.FACILITY]:
            capacity_str = f"Acute docs: {cap_map[state.acute_doc_state]}, Acute nurses: {cap_map[state.acute_nurse_state]}, Acute beds: {cap_map[state.acute_bed_state]}"
        else:  # ICU
            capacity_str = f"ICU docs: {cap_map[state.icu_doc_state]}, ICU nurses: {cap_map[state.icu_nurse_state]}, ICU beds: {cap_map[state.icu_bed_state]}"

        output.append(f"Step {t+1}:")
        output.append(f"  Vitals    — HR: {hr_map[state.hr_state]}, BP: {bp_map[state.sysbp_state]}, O2: {o2_map[state.percoxyg_state]}, Glucose: {glucose_map[state.glucose_state]}")
        output.append(f"  Location  — {soc_map[soc]}")
        output.append(f"  Capacity  — {capacity_str}")
        output.append(f"  Treatment — Antibiotics: {bool(action.antibiotic)}, Vasopressors: {bool(action.vasopressors)}, Ventilation: {bool(action.ventilation)}, Non-invasive vent: {bool(action.noninv_ventilation)}")
        output.append(f"  Reward    — {reward:.1f}")
        output.append("")

        if mdp.state.check_absorbing_state():
            outcome = "RECOVERED" if state.get_num_abnormal() == 0 else "DIED"
            output.append(f"Episode ended at step {t+1}: {outcome}")
            output.append("")
            break

    return '\n'.join(output)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    ppo_model = Model(State.NUM_STATE_VARS, Action.NUM_ACTIONS_TOTAL)
    ppo_model.load_state_dict(torch.load('results/ppo_weights.pt', map_location=device))
    ppo_model.to(device)
    ppo_model.eval()

    all_output = ''
    for i in range(10):
        all_output += render_trajectory(ppo_model, device, model_type='ppo')

    with open('results/ppo_trajectories.txt', 'w') as f:
        f.write(all_output)

    print("saved to results/ppo_trajectories.txt")