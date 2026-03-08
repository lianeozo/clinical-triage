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