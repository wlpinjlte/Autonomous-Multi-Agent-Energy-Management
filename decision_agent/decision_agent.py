import collections
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import NormalizeObservation, DatetimeWrapper
from gymnasium.wrappers import TransformObservation
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

class HVACPredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
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
            nn.Linear(hidden_dim, 1)
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
            low=np.append(obs_low, [-5e7, -5e7]),
            high=np.append(obs_high, [5e7, 5e7]),
            dtype=np.float32
        )

    def _get_augmented_obs(self, obs):
        seq = list(self.history)
        seq.append(np.concatenate((obs, np.zeros(self.env.action_space.shape))))
        x_lstm = torch.tensor(np.array(seq), dtype=torch.float32).unsqueeze(0).to(self.device)
        x_occ = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_temp = self.lstm_model(x_lstm).cpu().numpy()
            pred_occ = self.occ_model(x_occ).cpu().numpy()
            
        return np.append(obs, [pred_occ, pred_temp]).astype(np.float32)

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
        self.prev_action = None

    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        energy_cost = abs(info.get('reward_energy', float(obs[15])))

        current_temp_norm = float(obs[8])
        deadband = 0.2
        
        if abs(current_temp_norm) <= deadband:
            comfort_now_penalty_raw = 0.0
        else:
            comfort_now_penalty_raw = np.exp(abs(current_temp_norm) - deadband) - 1.0
            
        pred_occ_norm = float(obs[-2])
        future_temp_norm = float(obs[-1])
        
        if abs(future_temp_norm) <= deadband:
            comfort_future_penalty_raw = 0.0
        else:
            comfort_future_penalty_raw = np.exp(abs(future_temp_norm) - deadband) - 1.0

        occ_weight = np.clip((pred_occ_norm + 1.0) / 2.0, 0.0, 1.0)

        comfort_now_penalty = comfort_now_penalty_raw * occ_weight
        comfort_future_penalty = comfort_future_penalty_raw * occ_weight
        
        if self.prev_action is not None:
            action_smoothing_penalty = float(np.sum(np.abs(action - self.prev_action)))
        else:
            action_smoothing_penalty = 0.0
        self.prev_action = action.copy()
        
        w_energy = 0.65
        w_comfort_now = 0.20
        w_comfort_future = 0.05
        w_smoothing = 0.10
        
        custom_reward = - (w_energy * energy_cost + w_comfort_now * comfort_now_penalty + w_comfort_future * comfort_future_penalty + w_smoothing * action_smoothing_penalty)
       
        info['custom_energy_cost'] = energy_cost
        info['custom_comfort_penalty'] = comfort_now_penalty
        info['custom_future_penalty'] = comfort_future_penalty
        info['custom_smoothing_penalty'] = action_smoothing_penalty

        return obs, custom_reward, terminated, truncated, info


def main():
    env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
    
    env = DatetimeWrapper(env)
    env = NormalizeObservation(env)
    new_obs_space = Box(low=env.observation_space.low, high=env.observation_space.high, dtype=np.float32)
    env = TransformObservation(env, func=lambda obs: np.array(obs, dtype=np.float32), observation_space=new_obs_space)

    env = DualPredictorObservationWrapper(env, lstm_model_path='data/hvac_lstm_temperature_model.pth', occ_path='data/hvac_occupancy_model.pth')

    env = CustomRewardWrapper(env)

    print("Weryfikacja środowiska...")
    check_env(env)
    print("Architektura pozytywnie zweryfikowana. Uruchamiam trening SAC.")

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_hvac_tensorboard/")
    model.learn(total_timesteps=70000, log_interval=4)

    model.save("data/sac_hvac_agent_with_lstm")
    print("Trening zakończony!")

if __name__ == "__main__":
    main()