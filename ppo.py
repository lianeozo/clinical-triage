import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from sepsisSimDiabetes.DataGenerator import DataGenerator
from sepsisSimDiabetes.State import State
from sepsisSimDiabetes.Action import Action 



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

        return (np.stack(obs), np.stack(next_obs), np.array(act, dtype=np.int64), np.array(re, dtype=np.float32), np.array(finished, dtype=np.float32))
    


