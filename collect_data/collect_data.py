import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import NormalizeObservation, DatetimeWrapper
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import TransformObservation
from gymnasium.spaces import Box
import numpy as np

class OccupancyNoiseWrapper(gym.Wrapper):
    def __init__(self, env, noise_std=2.0):
        super().__init__(env)
        try:
            obs_vars = self.env.get_wrapper_attr('observation_variables')
            self.occ_idx = obs_vars.index('people_occupant')
        except (AttributeError, ValueError):
            self.occ_idx = 14
        self.noise_std = noise_std
        
        self.current_event_noise = 0.0
        self.event_steps_remaining = 0

    def reset(self, **kwargs):
        self.current_event_noise = 0.0
        self.event_steps_remaining = 0
        obs, info = self.env.reset(**kwargs)
        obs = self._apply_noise(obs, info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._apply_noise(obs, info)
        return obs, reward, terminated, truncated, info

    def _apply_noise(self, obs, info):
        info['true_occupancy_unnorm'] = float(obs[self.occ_idx])
        hour = info.get('hour', 12)
        day = info.get('day', 1)
        month = info.get('month', 1)
        
        import datetime
        try:
            dt = datetime.date(1991, month, day)
            is_weekend = dt.weekday() >= 5
        except ValueError:
            is_weekend = False

        if self.event_steps_remaining > 0:
            self.event_steps_remaining -= 1
            if self.event_steps_remaining == 0:
                self.current_event_noise = 0.0
        else:
            if is_weekend:
                if np.random.rand() < 0.02 and 8 <= hour <= 18:
                    self.current_event_noise = np.random.uniform(5, 15)
                    self.event_steps_remaining = np.random.randint(16, 32)
            elif hour < 6 or hour >= 20:
                if np.random.rand() < 0.05:
                    self.current_event_noise = np.random.uniform(3, 8)
                    self.event_steps_remaining = np.random.randint(4, 10)
            else:
                if np.random.rand() < 0.10:
                    self.current_event_noise = np.random.normal(0, self.noise_std * 2)
                    self.event_steps_remaining = np.random.randint(4, 8)

        noisy_obs = np.array(obs, dtype=np.float32)
        white_noise = np.random.normal(0, 1.0) if not is_weekend and (6 <= hour < 20) else 0.0
        total_noise = self.current_event_noise + white_noise
        noisy_obs[self.occ_idx] = max(0.0, float(noisy_obs[self.occ_idx] + total_noise))
        return noisy_obs


env_id = 'Eplus-5zone-hot-continuous-stochastic-v1'

# Changing zone from ZONE-1/SPACE1-1 (side zone) to SPACE5-1 (middle zone)
temp_env = gym.make(env_id)
try:
    default_vars = temp_env.get_wrapper_attr('variables')
    default_acts = temp_env.get_wrapper_attr('actuators')
    default_meters = temp_env.get_wrapper_attr('meters')
except AttributeError:
    default_vars = temp_env.unwrapped.variables
    default_acts = temp_env.unwrapped.actuators
    default_meters = temp_env.unwrapped.meters
temp_env.close()

new_vars = {k: (v[0], str(v[1]).replace('SPACE1-1', 'SPACE5-1').replace('ZONE-1', 'SPACE5-1')) if isinstance(v, tuple) and len(v) == 2 else v for k, v in default_vars.items()}
new_acts = {k: (v[0], v[1], str(v[2]).replace('SPACE1-1', 'SPACE5-1').replace('ZONE-1', 'SPACE5-1')) if isinstance(v, tuple) and len(v) == 3 else v for k, v in default_acts.items()}

env = gym.make(env_id, variables=new_vars, actuators=new_acts, meters=default_meters)


env = DatetimeWrapper(env)
env = OccupancyNoiseWrapper(env, noise_std=2.0) # Noise before normalization
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

state_columns = env.get_wrapper_attr('observation_variables')
action_columns = env.get_wrapper_attr('action_variables')

np.savez('/app/data/lstm_dataset.npz', 
         states=states_array, 
         actions=actions_array, 
         next_states=next_states_array,
         state_columns=state_columns,
         action_columns=action_columns,
         obs_mean=env.get_wrapper_attr('obs_rms').mean,
         obs_var=env.get_wrapper_attr('obs_rms').var,
         obs_count=env.get_wrapper_attr('obs_rms').count)
print("Saved data to lstm_dataset.npz")

env.close()