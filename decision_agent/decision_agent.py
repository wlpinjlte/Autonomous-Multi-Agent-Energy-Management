import collections
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import sinergym
import os
from sinergym.utils.wrappers import NormalizeObservation, DatetimeWrapper
from gymnasium.wrappers import TransformObservation
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

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
        
        # Get isolated random number generator for given environment
        rng = self.env.np_random
        
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
                if rng.random() < 0.02 and 8 <= hour <= 18:
                    self.current_event_noise = rng.uniform(5, 15)
                    self.event_steps_remaining = rng.integers(16, 32)
            elif hour < 6 or hour >= 20:
                if rng.random() < 0.05:
                    self.current_event_noise = rng.uniform(3, 8)
                    self.event_steps_remaining = rng.integers(4, 10)
            else:
                if rng.random() < 0.10:
                    self.current_event_noise = rng.normal(0, self.noise_std * 2)
                    self.event_steps_remaining = rng.integers(4, 8)

        noisy_obs = np.array(obs, dtype=np.float32)
        white_noise = rng.normal(0, 1.0) if not is_weekend and (6 <= hour < 20) else 0.0
        total_noise = self.current_event_noise + white_noise
        noisy_obs[self.occ_idx] = max(0.0, float(noisy_obs[self.occ_idx] + total_noise))
        return noisy_obs

class HVACPredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, num_layers=2):
        super(HVACPredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class OccupancyPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(OccupancyPredictorMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        return self.net(x)

class DualPredictorObservationWrapper(gym.Wrapper):
    def __init__(self, env, lstm_model_path, occ_path, seq_length=5):
        super().__init__(env)
        self.seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        self.lstm_model = HVACPredictorLSTM(input_dim=obs_dim + act_dim, hidden_dim=64).to(self.device)
        self.lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=self.device, weights_only=True))
        self.lstm_model.eval()

        self.occ_model = OccupancyPredictorMLP(input_dim=obs_dim).to(self.device)
        self.occ_model.load_state_dict(torch.load(occ_path, map_location=self.device, weights_only=True))
        self.occ_model.eval()
        
        self.history = collections.deque(maxlen=self.seq_length - 1)
        self.current_obs = None
        
        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high
        self.observation_space = Box(
            low=np.append(obs_low, [-5e7, -5e7, -5e7, -5e7]),
            high=np.append(obs_high, [5e7, 5e7, 5e7, 5e7]),
            dtype=np.float32
        )

    def _get_augmented_obs(self, obs):
        seq = list(self.history)
        if len(seq) > 0:
            last_action = seq[-1][obs.shape[0]:]
        else:
            last_action = (self.env.action_space.low + self.env.action_space.high) / 2.0
            
        seq.append(np.concatenate((obs, last_action)))
        x_lstm = torch.tensor(np.array(seq), dtype=torch.float32).unsqueeze(0).to(self.device)
        x_occ = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_temp = self.lstm_model(x_lstm).cpu().numpy()[0]
            pred_occ = self.occ_model(x_occ).cpu().numpy()[0]
            
        return np.append(obs, [*pred_occ, *pred_temp]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        
        neutral_action = (self.env.action_space.low + self.env.action_space.high) / 2.0
        for _ in range(self.seq_length - 1):
            self.history.append(np.concatenate((obs, neutral_action)))
            
        self.current_obs = obs
        return self._get_augmented_obs(obs), info

    def step(self, action):
        self.history.append(np.concatenate((self.current_obs, action)))
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_obs = obs

        return self._get_augmented_obs(obs), reward, terminated, truncated, info


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        try:
            obs_vars = self.env.get_wrapper_attr('observation_variables')
            self.temp_idx = obs_vars.index('air_temperature')
            self.occ_idx = obs_vars.index('people_occupant')
        except (AttributeError, ValueError):
            self.temp_idx = 12
            self.occ_idx = 14
            
        self.w_energy = float(os.getenv("WEIGHT_ENERGY", "0.80"))
        self.w_comfort_now = float(os.getenv("WEIGHT_COMFORT_NOW", "0.05"))
        self.w_comfort_future = float(os.getenv("WEIGHT_COMFORT_FUTURE", "0.15"))
        print(f"[REWARD WRAPPER] Started with weights -> Energy: {self.w_energy}, Comfort: {self.w_comfort_now}, Future: {self.w_comfort_future}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        power_w = info.get('total_power_demand', 0.0)
        lambda_energy = 1e-4 
        energy_penalty = power_w * lambda_energy

        # 2. Read unnormalized values taking into account LSTM/MLP predictors
        try:
            obs_rms = self.get_wrapper_attr('obs_rms')
            epsilon = self.get_wrapper_attr('epsilon')
            
            # Unnormalize temperature
            mean_temp = obs_rms.mean[self.temp_idx]
            std_temp = np.sqrt(obs_rms.var[self.temp_idx] + epsilon)
            current_temp = float(obs[self.temp_idx]) * std_temp + mean_temp
            future_temp_t1 = float(obs[-2]) * std_temp + mean_temp
            future_temp_t2 = float(obs[-1]) * std_temp + mean_temp
        except AttributeError:
            current_temp = float(obs[self.temp_idx])
            future_temp_t1 = float(obs[-2])
            future_temp_t2 = float(obs[-1])
        
        # 3. Increase temperature strictness
        target_temp = 23.5
        deadband = 0.5
        
        diff_now = current_temp - target_temp
        diff_t1 = future_temp_t1 - target_temp
        diff_t2 = future_temp_t2 - target_temp
        
        current_occ_weight = float(np.clip((obs[self.occ_idx] + 1.0) / 2.0, 0.0, 1.0))
        pred_occ_t1_weight = float(np.clip((obs[-4] + 1.0) / 2.0, 0.0, 1.0))
        pred_occ_t2_weight = float(np.clip((obs[-3] + 1.0) / 2.0, 0.0, 1.0))

        delta_now = float(max(0.0, abs(diff_now) - deadband)) 
        delta_t1 = float(max(0.0, abs(diff_t1) - deadband))
        delta_t2 = float(max(0.0, abs(diff_t2) - deadband))
        
        comfort_now_penalty = delta_now * current_occ_weight
        comfort_future_penalty_t1 = delta_t1 * pred_occ_t1_weight
        comfort_future_penalty_t2 = delta_t2 * pred_occ_t2_weight
        comfort_future_penalty = (comfort_future_penalty_t1 + comfort_future_penalty_t2) / 2.0

        custom_reward = - (
            self.w_energy * energy_penalty + 
            self.w_comfort_now * comfort_now_penalty + 
            self.w_comfort_future * comfort_future_penalty
        )
       
        info['custom_energy_cost'] = power_w / 1000.0
        info['custom_comfort_penalty'] = comfort_now_penalty
        info['custom_future_penalty'] = comfort_future_penalty

        return obs, custom_reward, terminated, truncated, info

def make_env(env_id, rank, seed=0):
    def _init():
        # Changing zone from ZONE-1/SPACE1-1 to SPACE5-1
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
        env = OccupancyNoiseWrapper(env, noise_std=2.0)
        env = NormalizeObservation(env)
        
        try:
            data = np.load('data/lstm_dataset.npz')
            obs_rms = env.get_wrapper_attr('obs_rms')
            obs_rms.mean = data['obs_mean']
            obs_rms.var = data['obs_var']
            obs_rms.count = 1e8
        except Exception as e:
            print(f"[{rank}] Could not load normalization calibration: {e}")
        
        new_obs_space = Box(low=env.observation_space.low, high=env.observation_space.high, dtype=np.float32)
        env = TransformObservation(env, func=lambda obs: np.array(obs, dtype=np.float32), observation_space=new_obs_space)

        env = DualPredictorObservationWrapper(env, lstm_model_path='data/hvac_lstm_temperature_model.pth', occ_path='data/hvac_occupancy_model.pth')
        env = CustomRewardWrapper(env)
        
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    env_id = 'Eplus-5zone-hot-continuous-stochastic-v1'
    num_cpu = 6
    print(f"Initializing {num_cpu} parallel environments...")
    
    dummy_env = make_env(env_id, 0)()
    check_env(dummy_env)
    print("Architecture positively verified.")
    
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    print("Running multi-threaded SAC training.")

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_hvac_tensorboard/")
    model.learn(total_timesteps=350000, log_interval=4)

    model.save("data/sac_hvac_agent_with_lstm")
    print("Training completed!")

if __name__ == "__main__":
    main()