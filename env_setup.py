import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import NormalizeObservation, DatetimeWrapper
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import TransformObservation
from gymnasium.spaces import Box
import numpy as np

env_id = 'Eplus-5zone-hot-continuous-v1'
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


obs, info = env.reset()
print(f"\nObservations (demation: {obs.shape}):\n{obs}")
 
action = env.action_space.sample()

next_obs, reward, terminated, truncated, info = env.step(action)

print(f"\nAction: {action}")
print(f"Reward: {reward}")

env.close()