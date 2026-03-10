import numpy as np
import random
import os
import json
from collections import namedtuple, deque
import math
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm
from sepsisSimDiabetes.MDP import MDP
from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
# EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, hidden_dim, n_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_observations, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, obs):
        output = self.network(obs)
        return output

def select_action(state, policy_net, mdp, steps_done, eps_decay, random_baseline):
    if random_baseline:
        return torch.tensor([[mdp.random_select_action().get_action_idx()]], device = device, dtype=torch.long)
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / eps_decay)

    if sample > eps_threshold:
        with torch.no_grad():
            action_idx = policy_net(state).max(1).indices.item()
            action = Action(action_idx=action_idx)
        if not mdp.treatment_feasibility(action) or not mdp.soc_feasibility(action.soc):
                action = mdp.random_select_action()
        return torch.tensor([[action.get_action_idx()]], device=device, dtype=torch.long)
    else:
        return torch.tensor([[mdp.random_select_action().get_action_idx()]], device=device, dtype=torch.long)

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken.
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(dim=1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def plot_rewards(episode_rewards, show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Rewards Result')
    else:
        plt.clf()
        plt.title('Training Rewards...')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot per-episode reward
    xs = np.arange(len(rewards_t))
    plt.plot(xs, rewards_t.numpy(), alpha=0.3, color='blue')
    
    # plt.plot(rewards_t.numpy())
    if len(rewards_t) >= 20:
        means = rewards_t.unfold(0, 20, 1).mean(1).view(-1)
        std_dev = rewards_t.unfold(0, 20, 1).std(1).view(-1)
        # means = torch.cat((torch.zeros(19), means))
        # std_dev = torch.cat((torch.zeros(19), std_dev))
        
        xs = np.arange(19, len(rewards_t))
        plt.fill_between(xs, (means+std_dev).numpy(), (means-std_dev).numpy(), facecolor='blue', alpha=0.25)
        plt.plot(xs, means.numpy(), color='blue', linewidth=2)

    plt.plot([], [], alpha=0.3, color='blue', label='Per-episode reward')
    plt.plot([], [], color='blue', linewidth=2, label='Rolling mean (20 ep)')
    plt.fill_between([], [], [], facecolor='blue', alpha=0.25, label='Rolling std (20 ep)')
    plt.legend(loc='upper left')
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
def save_results(episode_rewards, model_name='dqn', save_dir='results', random_baseline=False):
    os.makedirs(save_dir, exist_ok=True)
    if random_baseline:
        model_name = 'random'
    results = {
        'model': model_name,
        'episode_rewards': [float(r) for r in episode_rewards],
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'final_20_mean': float(np.mean(episode_rewards[-20:])),
        'final_20_std': float(np.std(episode_rewards[-20:])),
    }

    with open(f'{save_dir}/{model_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    plot_rewards(episode_rewards, show_result=True)
    fig = plt.figure(1)
    fig.savefig(f'{save_dir}/{model_name}_rewards.png', dpi=150, bbox_inches='tight')

    print(f"\n{model_name} training done")
    print(f"mean reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"last 20 eps: {results['final_20_mean']:.2f} +/- {results['final_20_std']:.2f}")
    print(f"saved to {save_dir}/")

def train_dqn(memory, policy_net, target_net, optimizer, num_episodes, T, random_baseline=False):
    steps_done = 0
    eps_decay = int(num_episodes * T * 0.8)
    episode_durations = []
    episode_rewards = []
    
    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        done = False
        mdp = MDP(init_state_idx=None, policy_array=None, p_diabetes=0.2)
        observation = torch.tensor(mdp.get_observation(), dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        
        for t in range(T):
            action = select_action(observation, policy_net, mdp, steps_done, eps_decay, random_baseline)
            reward = mdp.transition(Action(action_idx=action.item()))
            total_reward += reward
            scaled_reward = reward / 10000 # try scaling reward to stabilize training ?
            scaled_reward = torch.tensor(scaled_reward, dtype=torch.float32, device=device).unsqueeze(0)
            next_observation = torch.tensor(mdp.get_observation(), dtype=torch.float32, device=device).unsqueeze(0)
    
            # Store the transition in memory
            memory.push(observation, action, next_observation, scaled_reward)
            optimize_model(memory, policy_net, target_net, optimizer)
            steps_done += 1
    
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            done = mdp.state.check_absorbing_state()
    
            if done or (t == T - 1):
                episode_durations.append(t + 1)
                episode_rewards.append(total_reward)
                if (i_episode % 20 == 0):
                    plot_rewards(episode_rewards)
                break
    
    print('Complete')
    save_results(episode_rewards, model_name='dqn', random_baseline=random_baseline)
    return policy_net, target_net
