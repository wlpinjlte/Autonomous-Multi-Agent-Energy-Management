import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import NormalizeObservation, DatetimeWrapper

env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
env = DatetimeWrapper(env)
env = NormalizeObservation(env)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

print("INFO DICT KEYS:", info.keys())
print("INFO DICT:", info)
env.close()
