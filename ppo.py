import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sepsisSimDiabetes.DataGenerator import DataGenerator
from sepsisSimDiabetes.State import State
from sepsisSimDiabetes.Action import Action 
from dqn import plot_rewards, save_results




#actor critic model
class Model(nn.Module): 
    def __init__(self, obs_dimensions, n_actions, hidden_dimensions=128):
        super().__init__()
        self.structure = nn.Sequential(
            nn.Linear(obs_dimensions, hidden_dimensions),
            nn.Tanh(), 
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.Tanh(),

        )

        #getting the logits
        self.policy_start = nn.Linear(hidden_dimensions, n_actions)
        self.value_start = nn.Linear(hidden_dimensions, 1)

    def forward(self, obs):

        x = self.structure(obs)
        logits = self.policy_start(x)
        values = self.value_start(x).squeeze(-1)

        return logits, values
    
    def get_distribution(self, obs):
        #call the forward pass
        logits, values =self.forward(obs)

        distribution = torch.distributions.Categorical(logits = logits)
        return distribution, values
    

#define a callable policy so that we can parameterize by obs and sample actions
class CallablePolicy:
    def __init__(self, model, device="cpu"):

        self.model = model
        self.device = device

    @torch.no_grad()
    def __call__(self, obs_array):
        obs = torch.tensor(obs_array, dtype=torch.float32, device = self.device).unsqueeze(0)
        distribution, values = self.model.get_distribution(obs)

        return distribution.probs.squeeze(0).cpu().numpy()
    

#take DataGenerator outputs and return an array of transitions needed for ppo
def rollout_processing(iter_states, iter_actions, iter_rewards, iter_lengths):
    
    N, Tp1, obs_dimensions = iter_states.shape
    obs, next_obs, act, re, finished = [], [], [], [], []

    for i in range(N):

        L = int(iter_lengths[i,0])

        for t in range(L):
            obs.append(iter_states[i, t].astype(np.float32))

            next_obs.append(iter_states[i, t+1].astype(np.float32))

            act.append(int(iter_actions[i, t, 0]))
            re.append(float(iter_rewards[i, t, 0]))
            finished.append(1.0 if (t == L-1) else 0.0)

        obs_arr = np.stack(obs)
        next_obs_arr = np.stack(next_obs)
        act_arr = np.array(act, dtype=np.int64) 
        rewards_arr = np.array(re, dtype=np.float32)
        finished_arr = np.array(finished, dtype=np.float32)

    return obs_arr, next_obs_arr, act_arr, rewards_arr, finished_arr


def calc_gae(rewards, finished, values, next_values, gamma=0.99, lambda_param=0.95):
    #takes in arrays of shape S 
    
    S = rewards.shape[0]
    advantage = np.zeros(S, dtype=np.float32)

    gae = 0.0

    for t in reversed(range(S)):
        mask = 1.0 - finished[t]

        delta = rewards[t] + gamma * next_values[t] * mask - values[t]

        #calc of gae
        gae = delta + gamma * lambda_param * mask * gae

        advantage[t] = gae

    returns = advantage + values
    return advantage, returns 


#ppo training logic 
def train_ppo(
        iters = 200, 
        rollout_episodes = 64, 
        max_steps = 200, 
        gamma = 0.99, 
        lambda_param=0.95, 
        clip_epsilon = 0.2, 
        lr = 3e-4, 
        training_epochs = 4, 
        minibatch_size = 128, 
        value_coef = 0.5, 
        entropy_coef = 0.01, 
        max_grad_norm = 1.0, 
        reward_scaling = 1e-4, device="cpu"
    ):

    obs_dimensions = State.NUM_STATE_VARS
    n_actions = Action.NUM_ACTIONS_TOTAL

    model = Model(obs_dimensions, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    dg = DataGenerator()
    #for plot
    episode_rewards = []

    for it in range(iters):

        callable_policy = CallablePolicy(model, device=device)

        iter_states, iter_actions, iter_lengths, iter_rewards, _ = dg.simulate(
            rollout_episodes, 
            max_steps, 
            policy=callable_policy,
            policy_idx_type="obs", 
            p_diabetes=0.2, 
            use_tqdm=False
        )

        obs_arr, next_obs_arr, act_arr, re_arr, finished_arr = rollout_processing(
            iter_states, iter_actions, iter_rewards, iter_lengths
        )

        eps_returns = [iter_rewards[i, :int(iter_lengths[i , 0]), 0].sum() for i in range(rollout_episodes)]
        avg_ep_ret = float(np.mean(eps_returns))
        avg_ep_len = float(np.mean(iter_lengths[:, 0]))

        episode_rewards.append(avg_ep_ret)
        plot_rewards(episode_rewards, show_result=False)

        re_arr = re_arr * 1e-4

        obs = torch.tensor(obs_arr, dtype=torch.float32, device=device)
        next_obs = torch.tensor(next_obs_arr, dtype=torch.float32, device=device)
        actions = torch.tensor(act_arr, dtype=torch.int64, device=device)
        dones = torch.tensor(finished_arr, dtype=torch.float32, device=device)

        with torch.no_grad():
            dist, values = model.get_distribution(obs)
            old_logp = dist.log_prob(actions)
            _, next_values = model.get_distribution(next_obs)


        adv_np, ret_np = calc_gae(
            rewards=re_arr,
            finished=finished_arr,
            values=values.cpu().numpy(),
            next_values=next_values.cpu().numpy(),
            gamma=gamma,
            lambda_param=lambda_param
        )
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        adv = torch.tensor(adv_np, dtype=torch.float32, device=device)
        rets = torch.tensor(ret_np, dtype=torch.float32, device=device)

        B = obs.shape[0]
        idx = torch.arange(B, device=device)

        for _ in range(training_epochs):
            perm = idx[torch.randperm(B)]

            for start in range(0, B, minibatch_size):
                mb = perm[start:start + minibatch_size]

                mb_obs = obs[mb]
                mb_actions = actions[mb]
                mb_old_logp = old_logp[mb]
                mb_adv = adv[mb]
                mb_rets = rets[mb]

                dist, v = model.get_distribution(mb_obs)
                if v.dim() == 2 and v.size(-1) == 1:
                    v = v.squeeze(-1)
                logp = dist.log_prob(mb_actions)
                ratio = torch.exp(logp - mb_old_logp)

                clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                pg_loss = -(torch.min(ratio * mb_adv, clipped * mb_adv)).mean()

                v_loss = functional.mse_loss(v, mb_rets)
                ent = dist.entropy().mean()

                loss = pg_loss + value_coef * v_loss - entropy_coef * ent

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        print(
            f"iter={it:03d} | steps={B} | avg_ep_len={avg_ep_len:.1f} "
            f"| avg_ep_return={avg_ep_ret:.2f}"
        )

    plot_rewards(episode_rewards, show_result=True)
    save_results(episode_rewards, model_name='ppo')
    return model, episode_rewards

        






