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
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

class HVACPredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(HVACPredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class LSTMObservationWrapper(gym.Wrapper):
    def __init__(self, env, lstm_model_path, seq_length=5, input_dim=22):
        super().__init__(env)
        self.seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.lstm_model = HVACPredictorLSTM(input_dim=input_dim, hidden_dim=64).to(self.device)
        self.lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=self.device, weights_only=True))
        self.lstm_model.eval()
        
        self.history = collections.deque(maxlen=self.seq_length - 1)
        self.current_obs = None
        
        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high
        self.observation_space = Box(
            low=np.append(obs_low, -5e7),
            high=np.append(obs_high, 5e7),
            dtype=np.float32
        )

    def _get_augmented_obs(self, obs):
        seq = list(self.history)
        seq.append(np.concatenate((obs, np.zeros(self.env.action_space.shape))))
        x = torch.tensor(np.array(seq), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_temp = self.lstm_model(x).cpu().numpy()
        return np.append(obs, pred_temp).astype(np.float32)

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

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        energy_cost = float(np.sum(np.abs(action)))
        
        current_temp_norm = float(obs[2])
        comfort_now_penalty = current_temp_norm ** 2
        future_temp_norm = float(obs[-1])
        comfort_future_penalty = future_temp_norm ** 2
        
        w_energy = 0.40
        w_comfort_now = 0.40
        w_comfort_future = 0.20
        custom_reward = - (w_energy * energy_cost + w_comfort_now * comfort_now_penalty + w_comfort_future * comfort_future_penalty)
        
        info['custom_energy_cost'] = energy_cost
        info['custom_comfort_penalty'] = comfort_now_penalty
        info['custom_future_penalty'] = comfort_future_penalty
        return obs, custom_reward, terminated, truncated, info

def run_episode(env, model=None, is_rbc=False):
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    history = {
        'temp': [],
        'energy': [],
        'comfort_penalty': [],
        'reward': []
    }
    
    while not (terminated or truncated):
        if is_rbc:
            action = np.array([21.0, 22.0], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
        else:
            action, _ = model.predict(obs, deterministic=True)
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        history['temp'].append(float(obs[2]))
        history['energy'].append(info.get('custom_energy_cost', 0.0))
        history['comfort_penalty'].append(info.get('custom_comfort_penalty', 0.0))
        history['reward'].append(reward) 
        
    return history

def main():
    eplus_path = os.environ.get('EPLUS_PATH', '/usr/local/EnergyPlus-25-1-0')
    if eplus_path not in sys.path:
        sys.path.insert(0, eplus_path)

    env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
    env = DatetimeWrapper(env)
    env = NormalizeObservation(env)
    new_obs_space = Box(low=env.observation_space.low, high=env.observation_space.high, dtype=np.float32)
    env = TransformObservation(env, func=lambda obs: np.array(obs, dtype=np.float32), observation_space=new_obs_space)
    env = LSTMObservationWrapper(env, lstm_model_path='data/hvac_lstm_temperature_model.pth')
    env = CustomRewardWrapper(env)

    print("Ładowanie wytrenowanego agenta SAC...")
    model = SAC.load("data/sac_hvac_agent_with_lstm", env=env)

    print("\n[1/2] Ewaluacja modelu SAC + LSTM (Proszę czekać, symulacja roczna)...")
    sac_history = run_episode(env, model=model, is_rbc=False)

    print("\n[2/2] Ewaluacja tradycyjnego termostatu RBC (Baseline)...")
    rbc_history = run_episode(env, is_rbc=True)

    env.close()

    sac_total_energy = np.sum(sac_history['energy'])
    rbc_total_energy = np.sum(rbc_history['energy'])
    
    sac_total_penalty = np.sum(sac_history['comfort_penalty'])
    rbc_total_penalty = np.sum(rbc_history['comfort_penalty'])

    sac_mean_reward = np.mean(sac_history['reward'])
    rbc_mean_reward = np.mean(rbc_history['reward'])

    os.makedirs('data', exist_ok=True)
    with open('data/rl_vs_baseline_metrics.txt', 'w', encoding='utf-8') as f:
        f.write("--- PORÓWNANIE: SAC+LSTM vs Tradycyjny Termostat (RBC) ---\n\n")
        f.write("1. Skumulowany koszt energii (Mniej = Lepiej):\n")
        f.write(f"   SAC Agent: {sac_total_energy:.2f}\n")
        f.write(f"   RBC Baseline: {rbc_total_energy:.2f}\n")
        f.write(f"   Zysk: {((rbc_total_energy - sac_total_energy) / rbc_total_energy * 100):.2f}%\n\n")
        
        f.write("2. Skumulowana kara za dyskomfort cieplny (Mniej = Lepiej):\n")
        f.write(f"   SAC Agent: {sac_total_penalty:.2f}\n")
        f.write(f"   RBC Baseline: {rbc_total_penalty:.2f}\n")
        f.write(f"   Zysk: {((rbc_total_penalty - sac_total_penalty) / rbc_total_penalty * 100):.2f}%\n\n")
        
        f.write("3. Średnia nagroda na krok (Więcej = Lepiej):\n")
        f.write(f"   SAC Agent: {sac_mean_reward:.4f}\n")
        f.write(f"   RBC Baseline: {rbc_mean_reward:.4f}\n")

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(np.cumsum(sac_history['energy']), label="SAC + LSTM", color="royalblue", linewidth=2)
    plt.plot(np.cumsum(rbc_history['energy']), label="Termostat (RBC)", color="crimson", linestyle="--", linewidth=2)
    plt.title("Skumulowane Zużycie Energii (Wysterowanie HVAC)")
    plt.xlabel("Kroki symulacji (czas)")
    plt.ylabel("Jednostki energii")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(np.cumsum(sac_history['comfort_penalty']), label="SAC + LSTM", color="royalblue", linewidth=2)
    plt.plot(np.cumsum(rbc_history['comfort_penalty']), label="Termostat (RBC)", color="crimson", linestyle="--", linewidth=2)
    plt.title("Skumulowana Kara za Brak Komfortu Cieplnego")
    plt.xlabel("Kroki symulacji (czas)")
    plt.ylabel("Skumulowana kara (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.hist(sac_history['temp'], bins=50, alpha=0.6, label="SAC + LSTM", color="royalblue", density=True)
    plt.hist(rbc_history['temp'], bins=50, alpha=0.5, label="Termostat (RBC)", color="crimson", density=True)
    plt.title("Rozkład Znormalizowanych Temperatur w Pomieszczeniu")
    plt.xlabel("Znormalizowana temperatura (0 = ideał)")
    plt.ylabel("Gęstość")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_limit = min(1000, len(sac_history['temp']))
    plt.subplot(2, 2, 4)
    plt.plot(sac_history['temp'][:plot_limit], label="SAC + LSTM", color="royalblue", alpha=0.9, linewidth=1.5)
    plt.plot(rbc_history['temp'][:plot_limit], label="Termostat (RBC)", color="crimson", linestyle="--", alpha=0.8, linewidth=1.5)
    plt.title(f"Profil Temperatur Wewnętrznych (Pierwsze {plot_limit} kroków)")
    plt.xlabel("Kroki symulacji")
    plt.ylabel("Znormalizowana temperatura")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/evaluation_rl_vs_rbc.png', dpi=300)
    plt.close()

    print("\n[✔] Zapisano raport tekstowy do 'data/rl_vs_baseline_metrics.txt'.")
    print("[✔] Zapisano arkusz wykresów do 'data/evaluation_rl_vs_rbc.png'.\n")

if __name__ == "__main__":
    main()