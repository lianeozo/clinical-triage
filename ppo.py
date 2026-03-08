import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from MDP import MDP
from State import State
from Action import Action


# ---------------------------------------------------------
# GYM ENVIRONMENT WRAPPER
# ---------------------------------------------------------

class TriageEnv(gym.Env):

    def __init__(self, max_steps=20):

        super().__init__()

        self.max_steps = max_steps
        self.step_count = 0

        self.mdp = MDP(
            init_state_idx=None,
            policy_array=None,
            policy_idx_type="obs",
            p_diabetes=0.2
        )

        # observation = state vector from simulator
        self.observation_space = spaces.Box(
            low=0,
            high=5,
            shape=(11,),
            dtype=np.float32
        )

        # actions correspond to Action(action_idx)
        self.action_space = spaces.Discrete(Action.NUM_ACTIONS_TOTAL)


    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.mdp.state = self.mdp.get_new_state()
        self.step_count = 0

        obs = self.mdp.state.get_state_vector().astype(np.float32)

        return obs, {}


    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------

    def step(self, action_idx):

        action = Action(action_idx=int(action_idx))

        # enforce feasibility
        if not self.mdp.soc_feasibility(action.soc) or \
           not self.mdp.treatment_feasibility(action):

            reward = -0.2
        else:
            reward = self.mdp.transition(action)

        obs = self.mdp.state.get_state_vector().astype(np.float32)

        self.step_count += 1

        terminated = reward != 0
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

#make environment
def make_single_env(max_steps=20):
    def _init():
        env = TriageEnv(max_steps =max_steps)
        env = Monitor(env)

        return env
    return _init


# ---------------------------------------------------------
# TRAIN PPO
# ---------------------------------------------------------

def train(
        total_timesteps=1000000, 
        n_envs=8, 
        max_steps=20, 
        model_path="ppo_triage_model",
        log_dir="./ppo_triage_logs/", 
):

    env = make_vec_env(make_single_env(max_steps=max_steps), n_envs=n_envs)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10, 
        gamma=0.99,
        gae_lambda=0.95, 
        clip_range=0.2, 
        ent_coef=0.01, 
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir, 
        policy_kwargs= dict( net_arch=dict(pi=[128, 128], vf=[128,128])),
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(model_path)

    print("Training complete. Model saved.")
    env.close()
    return model


# ---------------------------------------------------------
# EVALUATE POLICY
# ---------------------------------------------------------

def evaluate(model_path="ppo_triage_model",
    n_eval_episodes=20,
    max_steps=20,
    deterministic=True,):
    

    env = Monitor(TriageEnv(max_steps=max_steps))

    model = PPO.load(model_path)

    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=n_eval_episodes, 
        deterministic=deterministic,
    )

    print("Evaluate over ", n_eval_episodes, " episodes:")
    print("Mean reward: ", mean_reward, "\nstd reward: ", std_reward)

    obs, _ = env.reset()

    total_reward = 0

    for step in range(max_steps):

        action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, done, trunc, _ = env.step(action)

        total_reward += reward

        if done or trunc:
            break

    print("Episode reward:", total_reward)
    env.close()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    train(
        total_timesteps=1000000,
        n_envs=8, 
        max_steps=20,
        model_path="ppo_triage_model", 
        log_dir="./ppo_triage_logs/",
    )

    evaluate(
        model_path="ppo_triage_model",
        n_eval_episodes=20,
        max_steps=20,
        deterministic=True,
    )