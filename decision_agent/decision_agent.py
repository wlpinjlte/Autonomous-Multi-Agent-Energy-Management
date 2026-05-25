import collections
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import NormalizeObservation, DatetimeWrapper
from gymnasium.wrappers import TransformObservation
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class OccupancyNoiseWrapper(gym.Wrapper):
    def __init__(self, env, noise_std=2.0):
        super().__init__(env)
        try:
            obs_vars = self.env.get_wrapper_attr('observation_variables')
            self.occ_idx = obs_vars.index('people_occupant')
        except (AttributeError, ValueError):
            self.occ_idx = 14
        self.noise_std = noise_std

    def reset(self, **kwargs):
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

        if is_weekend or hour < 6 or hour >= 24:
            if np.random.rand() < 0.05:
                current_noise = np.random.normal(0, self.noise_std)
            else:
                current_noise = 0.0
        else:
            current_noise = np.random.normal(0, self.noise_std)
            
        noisy_obs = np.array(obs, dtype=np.float32)
        noisy_obs[self.occ_idx] = max(0.0, float(noisy_obs[self.occ_idx] + current_noise))
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
        seq.append(np.concatenate((obs, np.zeros(self.env.action_space.shape))))
        x_lstm = torch.tensor(np.array(seq), dtype=torch.float32).unsqueeze(0).to(self.device)
        x_occ = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_temp = self.lstm_model(x_lstm).cpu().numpy()[0]
            pred_occ = self.occ_model(x_occ).cpu().numpy()[0]
            
        return np.append(obs, [*pred_occ, *pred_temp]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        
        for _ in range(self.seq_length - 1):
            self.history.append(np.concatenate((obs, np.zeros(self.env.action_space.shape))))
            
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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Normalizacja Energii (Estymowane maksymalne zużycie to 20kW)
        MAX_POWER = 20.0
        energy_cost = info.get('total_power_demand', 0.0) / 1000.0
        energy_cost_norm = min(1.0, energy_cost / MAX_POWER)

        # Odczyt nieznormalizowanych wartości (kod z try/except pozostaje z oryginału)
        try:
            obs_rms = self.get_wrapper_attr('obs_rms')
            epsilon = self.get_wrapper_attr('epsilon')
            mean_temp = obs_rms.mean[self.temp_idx]
            std_temp = np.sqrt(obs_rms.var[self.temp_idx] + epsilon)
            
            current_temp = float(obs[self.temp_idx]) * std_temp + mean_temp
            future_temp_t1 = float(obs[-2]) * std_temp + mean_temp
            future_temp_t2 = float(obs[-1]) * std_temp + mean_temp
        except AttributeError:
            current_temp = float(obs[self.temp_idx])
            future_temp_t1 = float(obs[-2])
            future_temp_t2 = float(obs[-1])
            
        true_occ_unnorm = info.get('true_occupancy_unnorm', None)
        if true_occ_unnorm is not None:
            try:
                obs_rms = self.get_wrapper_attr('obs_rms')
                epsilon = self.get_wrapper_attr('epsilon')
                true_occ_norm = (true_occ_unnorm - obs_rms.mean[self.occ_idx]) / np.sqrt(obs_rms.var[self.occ_idx] + epsilon)
                current_occ_weight = float(np.clip((true_occ_norm + 1.0) / 2.0, 0.0, 1.0))
            except AttributeError:
                current_occ_weight = 1.0 if true_occ_unnorm > 0 else 0.0
        else:
            current_occ_weight = float(np.clip((obs[self.occ_idx] + 1.0) / 2.0, 0.0, 1.0))
            
        pred_occ_t1_weight = float(np.clip((obs[-4] + 1.0) / 2.0, 0.0, 1.0))
        pred_occ_t2_weight = float(np.clip((obs[-3] + 1.0) / 2.0, 0.0, 1.0))
        
        target_temp = 23.5
        deadband = 0.5
        alpha = 0.8  # Współczynnik nachylenia kary wykładniczej
        
        # 2. Błędy bezwzględne
        delta_now = float(max(0.0, abs(current_temp - target_temp) - deadband))
        delta_t1 = float(max(0.0, abs(future_temp_t1 - target_temp) - deadband))
        delta_t2 = float(max(0.0, abs(future_temp_t2 - target_temp) - deadband))
        
        # 3. Transformacja Wykładnicza
        exp_comfort_now = np.exp(alpha * delta_now) - 1.0
        exp_comfort_t1 = np.exp(alpha * delta_t1) - 1.0
        exp_comfort_t2 = np.exp(alpha * delta_t2) - 1.0
        
        # 4. Maskowanie z dolnym limitem (Base load threshold)
        BASE_OCC_LIMIT = 0.15
        w_occ_now = max(BASE_OCC_LIMIT, current_occ_weight)
        w_occ_t1 = max(BASE_OCC_LIMIT, pred_occ_t1_weight)
        w_occ_t2 = max(BASE_OCC_LIMIT, pred_occ_t2_weight)
        
        comfort_now_penalty = exp_comfort_now * w_occ_now
        comfort_future_penalty_t1 = exp_comfort_t1 * w_occ_t1
        comfort_future_penalty_t2 = exp_comfort_t2 * w_occ_t2
        comfort_future_penalty = (comfort_future_penalty_t1 + comfort_future_penalty_t2) / 2.0

        # 5. Zbalansowane Wagi Ostateczne
        w_energy = 0.5
        w_comfort_now = 0.35
        w_comfort_future = 0.15
        
        custom_reward = - (
            w_energy * energy_cost_norm + 
            w_comfort_now * comfort_now_penalty + 
            w_comfort_future * comfort_future_penalty
        )
       
        info['custom_energy_cost'] = energy_cost
        info['custom_comfort_penalty'] = comfort_now_penalty
        info['custom_future_penalty'] = comfort_future_penalty

        return obs, custom_reward, terminated, truncated, info

def main():
    env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
    env = DatetimeWrapper(env)
    env = OccupancyNoiseWrapper(env, noise_std=2.0)
    env = NormalizeObservation(env)
    
    try:
        data = np.load('data/lstm_dataset.npz')
        obs_rms = env.get_wrapper_attr('obs_rms')
        obs_rms.mean = data['obs_mean']
        obs_rms.var = data['obs_var']
        obs_rms.count = data['obs_count']
        print("Loaded NormalizeObservation calibration from dataset.")
    except Exception as e:
        print(f"Could not load normalization calibration: {e}")
        
    new_obs_space = Box(low=env.observation_space.low, high=env.observation_space.high, dtype=np.float32)
    env = TransformObservation(env, func=lambda obs: np.array(obs, dtype=np.float32), observation_space=new_obs_space)

    env = DualPredictorObservationWrapper(env, lstm_model_path='data/hvac_lstm_temperature_model.pth', occ_path='data/hvac_occupancy_model.pth')
    env = CustomRewardWrapper(env)

    print("Weryfikacja środowiska...")
    check_env(env)
    print("Architektura pozytywnie zweryfikowana. Uruchamiam trening PPO.")

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_hvac_tensorboard/", ent_coef=0.01)
    model.learn(total_timesteps=350000, log_interval=4)

    model.save("data/ppo_hvac_agent_with_lstm")
    print("Trening zakończony!")

if __name__ == "__main__":
    main()