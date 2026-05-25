import sys
import os
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
import matplotlib.pyplot as plt
from enum import Enum

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


class ControlMode(Enum):
    SAC = "SAC"
    RBC_STANDARD = "RBC_STANDARD"
    RBC_OCCUPANCY = "RBC_OCCUPANCY"

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
        
        energy_cost = info.get('total_power_demand', 0.0) / 1000.0

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
        
        comfort_now_penalty_raw = float(max(0.0, abs(current_temp - target_temp) - deadband))
        comfort_future_t1_raw = float(max(0.0, abs(future_temp_t1 - target_temp) - deadband))
        comfort_future_t2_raw = float(max(0.0, abs(future_temp_t2 - target_temp) - deadband))
        
        comfort_now_penalty = comfort_now_penalty_raw * current_occ_weight
        comfort_future_penalty_t1 = comfort_future_t1_raw * pred_occ_t1_weight
        comfort_future_penalty_t2 = comfort_future_t2_raw * pred_occ_t2_weight
        
        comfort_future_penalty = (comfort_future_penalty_t1 + comfort_future_penalty_t2) / 2.0

        w_energy = 5.0
        w_comfort_now = 20.0
        w_comfort_future = 5.0
        
        custom_reward = - (
            w_energy * energy_cost + 
            w_comfort_now * comfort_now_penalty + 
            w_comfort_future * comfort_future_penalty
        )
       
        info['custom_energy_cost'] = energy_cost
        info['custom_comfort_penalty'] = comfort_now_penalty
        info['custom_future_penalty'] = comfort_future_penalty

        return obs, custom_reward, terminated, truncated, info

def run_episode(env, model=None, mode=ControlMode.SAC):
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    try:
        obs_vars = env.get_wrapper_attr('observation_variables')
        temp_idx = obs_vars.index('air_temperature')
        occ_idx = obs_vars.index('people_occupant')
    except (AttributeError, ValueError):
        temp_idx = 12
        occ_idx = 14
    
    history = {
        'temp': [],
        'energy': [],
        'comfort_penalty': [],
        'future_penalty': [],
        'reward': []
    }
    
    while not (terminated or truncated):
        try:
            obs_rms = env.get_wrapper_attr('obs_rms')
            epsilon = env.get_wrapper_attr('epsilon')
            mean_occ = obs_rms.mean[occ_idx]
            std_occ = np.sqrt(obs_rms.var[occ_idx] + epsilon)
            current_occ = float(obs[occ_idx]) * std_occ + mean_occ
        except AttributeError:
            current_occ = float(obs[occ_idx])
            
        if mode == ControlMode.RBC_STANDARD:
            action = np.array([23.5, 23.5], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
        elif mode == ControlMode.RBC_OCCUPANCY:
            if current_occ > 0.5:
                action = np.array([23.5, 23.5], dtype=np.float32)
            else:
                action = np.array([15.0, 30.0], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        try:
            obs_rms = env.get_wrapper_attr('obs_rms')
            epsilon = env.get_wrapper_attr('epsilon')
            mean_temp = obs_rms.mean[temp_idx]
            std_temp = np.sqrt(obs_rms.var[temp_idx] + epsilon)
            unnorm_temp = float(obs[temp_idx]) * std_temp + mean_temp
        except AttributeError:
            unnorm_temp = float(obs[temp_idx])
            
        history['temp'].append(unnorm_temp)
        history['energy'].append(info.get('custom_energy_cost', 0.0))
        history['comfort_penalty'].append(info.get('custom_comfort_penalty', 0.0))
        history['future_penalty'].append(info.get('custom_future_penalty', 0.0))
        history['reward'].append(reward) 
        
    return history

def main():
    eplus_path = os.environ.get('EPLUS_PATH', '/usr/local/EnergyPlus-25-1-0')
    if eplus_path not in sys.path:
        sys.path.insert(0, eplus_path)

    env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
    env = DatetimeWrapper(env)
    env = OccupancyNoiseWrapper(env, noise_std=2.0)
    env = NormalizeObservation(env)
    
    try:
        data = np.load('data/lstm_dataset.npz')
        obs_rms = env.get_wrapper_attr('obs_rms')
        obs_rms.mean = data['obs_mean']
        obs_rms.var = data['obs_var']
        # Set count very high so evaluation doesn't change the normalization
        obs_rms.count = 1e8 
        print("Loaded NormalizeObservation calibration from dataset.")
    except Exception as e:
        print(f"Could not load normalization calibration: {e}")
        
    new_obs_space = Box(low=env.observation_space.low, high=env.observation_space.high, dtype=np.float32)
    env = TransformObservation(env, func=lambda obs: np.array(obs, dtype=np.float32), observation_space=new_obs_space)
    env = DualPredictorObservationWrapper(env, lstm_model_path='data/hvac_lstm_temperature_model.pth', occ_path='data/hvac_occupancy_model.pth')
    env = CustomRewardWrapper(env)
    print("Ładowanie wytrenowanego agenta PPO...")
    model = PPO.load("data/ppo_hvac_agent_with_lstm", env=env)

    print("\n[1/3] Ewaluacja modelu PPO + LSTM (Proszę czekać, symulacja roczna)...")
    sac_history = run_episode(env, model=model, mode=ControlMode.SAC)

    print("\n[2/3] Ewaluacja tradycyjnego termostatu RBC (Standard)...")
    rbc_history = run_episode(env, mode=ControlMode.RBC_STANDARD)

    print("\n[3/3] Ewaluacja termostatu RBC (Zależny od obecności)...")
    rbc_occ_history = run_episode(env, mode=ControlMode.RBC_OCCUPANCY)

    env.close()

    sac_total_energy = np.sum(sac_history['energy'])
    rbc_total_energy = np.sum(rbc_history['energy'])
    rbc_occ_total_energy = np.sum(rbc_occ_history['energy'])
    
    sac_total_penalty = np.sum(sac_history['comfort_penalty'])
    rbc_total_penalty = np.sum(rbc_history['comfort_penalty'])
    rbc_occ_total_penalty = np.sum(rbc_occ_history['comfort_penalty'])

    sac_mean_reward = np.mean(sac_history['reward'])
    sac_energy = np.sum(sac_history['energy'])
    rbc_energy = np.sum(rbc_history['energy'])
    rbc_occ_energy = np.sum(rbc_occ_history['energy'])
    
    sac_comfort = np.sum(sac_history['comfort_penalty'])
    rbc_comfort = np.sum(rbc_history['comfort_penalty'])
    rbc_occ_comfort = np.sum(rbc_occ_history['comfort_penalty'])

    sac_reward = np.mean(sac_history['reward'])
    rbc_reward = np.mean(rbc_history['reward'])
    rbc_occ_reward = np.mean(rbc_occ_history['reward'])
    
    sac_future = np.sum(sac_history.get('future_penalty', [0]))
    rbc_future = np.sum(rbc_history.get('future_penalty', [0]))
    rbc_occ_future = np.sum(rbc_occ_history.get('future_penalty', [0]))

    os.makedirs('data', exist_ok=True)
    with open('data/rl_vs_baseline_metrics.txt', 'w', encoding='utf-8') as f:
        metrics_text = f"--- PORÓWNANIE: PPO+LSTM vs Tradycyjny Termostat (RBC) vs RBC Occupancy ---\n\n"
        metrics_text += f"1. Skumulowany koszt energii (Mniej = Lepiej):\n"
        metrics_text += f"   PPO Agent: {sac_energy:.2f}\n"
        metrics_text += f"   RBC Standard: {rbc_energy:.2f}\n"
        metrics_text += f"   RBC Occupancy: {rbc_occ_energy:.2f}\n"
        metrics_text += f"   Zysk (PPO vs Standard): {((rbc_energy - sac_energy) / rbc_energy) * 100:.2f}%\n"
        metrics_text += f"   Zysk (PPO vs Occupancy): {((rbc_occ_energy - sac_energy) / rbc_occ_energy) * 100:.2f}%\n\n"
        
        metrics_text += f"2. Skumulowana kara za dyskomfort cieplny (Mniej = Lepiej):\n"
        metrics_text += f"   PPO Agent: {sac_comfort:.2f}\n"
        metrics_text += f"   RBC Standard: {rbc_comfort:.2f}\n"
        metrics_text += f"   RBC Occupancy: {rbc_occ_comfort:.2f}\n"
        metrics_text += f"   Zysk (PPO vs Standard): {((rbc_comfort - sac_comfort) / (rbc_comfort + 1e-5)) * 100:.2f}%\n"
        metrics_text += f"   Zysk (PPO vs Occupancy): {((rbc_occ_comfort - sac_comfort) / (rbc_occ_comfort + 1e-5)) * 100:.2f}%\n\n"
        
        metrics_text += f"3. Średnia nagroda na krok (Więcej = Lepiej):\n"
        metrics_text += f"   PPO Agent: {sac_reward:.4f}\n"
        metrics_text += f"   RBC Standard: {rbc_reward:.4f}\n"
        metrics_text += f"   RBC Occupancy: {rbc_occ_reward:.4f}\n\n"

        metrics_text += f"4. Szczegóły Kar (Future Penalty):\n"
        metrics_text += f"   PPO Agent: Future={sac_future:.2f}\n"
        metrics_text += f"   RBC Standard: Future={rbc_future:.2f}\n"
        metrics_text += f"   RBC Occupancy: Future={rbc_occ_future:.2f}\n"
        f.write(metrics_text)

    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(np.cumsum(sac_history['energy']), label="PPO + LSTM", color="royalblue", linewidth=2)
    plt.plot(np.cumsum(rbc_history['energy']), label="RBC Standard", color="crimson", linestyle="--", linewidth=2)
    plt.plot(np.cumsum(rbc_occ_history['energy']), label="RBC Occupancy", color="forestgreen", linestyle=":", linewidth=2)
    plt.title("Skumulowane Zużycie Energii (Wysterowanie HVAC)")
    plt.xlabel("Kroki symulacji (czas)")
    plt.ylabel("Jednostki energii")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(np.cumsum(sac_history['comfort_penalty']), label="SAC + LSTM", color="royalblue", linewidth=2)
    plt.plot(np.cumsum(rbc_history['comfort_penalty']), label="RBC Standard", color="crimson", linestyle="--", linewidth=2)
    plt.plot(np.cumsum(rbc_occ_history['comfort_penalty']), label="RBC Occupancy", color="forestgreen", linestyle=":", linewidth=2)
    plt.title("Skumulowana Kara za Brak Komfortu Cieplnego")
    plt.xlabel("Kroki symulacji (czas)")
    plt.ylabel("Skumulowana kara (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.hist(sac_history['temp'], bins=50, alpha=0.6, label="SAC + LSTM", color="royalblue", density=True)
    plt.hist(rbc_history['temp'], bins=50, alpha=0.5, label="RBC Standard", color="crimson", density=True)
    plt.hist(rbc_occ_history['temp'], bins=50, alpha=0.4, label="RBC Occupancy", color="forestgreen", density=True)
    plt.title("Rozkład Rzeczywistych Temperatur w Pomieszczeniu")
    plt.xlabel("Temperatura w st. Celsjusza (20 = ideał)")
    plt.ylabel("Gęstość")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_limit = min(1000, len(sac_history['temp']))
    plt.subplot(2, 2, 4)
    plt.plot(sac_history['temp'][:plot_limit], label="SAC + LSTM", color="royalblue", alpha=0.9, linewidth=1.5)
    plt.plot(rbc_history['temp'][:plot_limit], label="RBC Standard", color="crimson", linestyle="--", alpha=0.8, linewidth=1.5)
    plt.plot(rbc_occ_history['temp'][:plot_limit], label="RBC Occupancy", color="forestgreen", linestyle=":", alpha=0.8, linewidth=1.5)
    plt.title(f"Profil Temperatur Wewnętrznych (Pierwsze {plot_limit} kroków)")
    plt.xlabel("Kroki symulacji")
    plt.ylabel("Temperatura w st. Celsjusza")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/evaluation_rl_vs_rbc.png', dpi=300)
    plt.close()

    print("\n[✔] Zapisano raport tekstowy do 'data/rl_vs_baseline_metrics.txt'.")
    print("[✔] Zapisano arkusz wykresów do 'data/evaluation_rl_vs_rbc.png'.\n")

if __name__ == "__main__":
    main()