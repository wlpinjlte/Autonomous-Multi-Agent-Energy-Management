import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import NormalizeObservation, DatetimeWrapper
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import TransformObservation
from gymnasium.spaces import Box
import numpy as np

env_id = 'Eplus-5zone-hot-continuous-stochastic-v1'
env = gym.make(env_id)


env = DatetimeWrapper(env)
# rescal to [-1, 1]
env = NormalizeObservation(env)


new_obs_space = Box(
    low=env.observation_space.low, 
    high=env.observation_space.high, 
    dtype=np.float32
)

env = TransformObservation(
    env, 
    func=lambda obs: np.array(obs, dtype=np.float32), 
    observation_space=new_obs_space
)

check_env(env)
print("Env loaded")
num_episodes = 10
dataset_states = []
dataset_actions = []
dataset_next_states = []

for ep in range(num_episodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    step_count = 0
    while not (terminated or truncated):
        action = env.action_space.sample()
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        dataset_states.append(obs)
        dataset_actions.append(action)
        dataset_next_states.append(next_obs)
        
        obs = next_obs
        step_count += 1
        
        if step_count % 5000 == 0:
            print(f"Done step: {step_count}...")

    print(f"Finished episodes {ep + 1}")

states_array = np.array(dataset_states)
actions_array = np.array(dataset_actions)
next_states_array = np.array(dataset_next_states)

np.savez('/app/data/lstm_dataset.npz', 
         states=states_array, 
         actions=actions_array, 
         next_states=next_states_array)
print("Saved data to lstm_dataset.npz")

env.close()