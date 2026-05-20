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
from enum import Enum

class ControlMode(Enum):
    SAC = "SAC"
    RBC_STANDARD = "RBC_STANDARD"
    RBC_OCCUPANCY = "RBC_OCCUPANCY"

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
        
        try:
            obs_vars = self.env.get_wrapper_attr('observation_variables')
            self.temp_idx = obs_vars.index('air_temperature')
            self.occ_idx = obs_vars.index('people_occupant')
            self.energy_idx = obs_vars.index('HVAC_electricity_demand_rate')
        except (AttributeError, ValueError):
            self.temp_idx = 12
            self.occ_idx = 14
            self.energy_idx = 18

    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        # 1. Wysyłamy akcję w 100% bezpośrednio bez modyfikacji, chroniąc spójność MDP w buforze SAC
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Pobranie fizycznego zużycia energii w Watach
        raw_energy = info.get('total_power_demand')
        if raw_energy is None:
            try:
                obs_rms = self.get_wrapper_attr('obs_rms')
                epsilon = self.get_wrapper_attr('epsilon')
                mean_energy = obs_rms.mean[self.energy_idx]
                std_energy = np.sqrt(obs_rms.var[self.energy_idx] + epsilon)
                raw_energy = float(obs[self.energy_idx]) * std_energy + mean_energy
            except AttributeError:
                raw_energy = float(obs[self.energy_idx])

        energy_cost = abs(raw_energy) / 1000.0

        # Odnormalizowanie temperatur
        try:
            obs_rms = self.get_wrapper_attr('obs_rms')
            epsilon = self.get_wrapper_attr('epsilon')
            mean_temp = obs_rms.mean[self.temp_idx]
            std_temp = np.sqrt(obs_rms.var[self.temp_idx] + epsilon)
            
            current_temp = float(obs[self.temp_idx]) * std_temp + mean_temp
            future_temp = float(obs[-1]) * std_temp + mean_temp
        except AttributeError:
            current_temp = float(obs[self.temp_idx])
            future_temp = float(obs[-1])
            
        target_temp = 23.5
        deadband = 0.5
        
        # Liniowa kara komfortu zamiast kwadratowej
        current_diff = abs(current_temp - target_temp)
        comfort_now_penalty_raw = float(max(0.0, current_diff - deadband))
            
        try:
            mean_occ = obs_rms.mean[self.occ_idx]
            std_occ = np.sqrt(obs_rms.var[self.occ_idx] + epsilon)
            pred_occ = float(obs[-2]) * std_occ + mean_occ
            current_occ = float(obs[self.occ_idx]) * std_occ + mean_occ
        except AttributeError:
            pred_occ = float(obs[-2])
            current_occ = float(obs[self.occ_idx])
        
        future_diff = abs(future_temp - target_temp)
        comfort_future_penalty_raw = float(max(0.0, future_diff - deadband))

        # Skalowanie wag zajętości (zgodnie z życzeniem Użytkownika): (obs + 1) / 2
        current_occ_weight = (obs[self.occ_idx] + 1.0) / 2.0
        current_occ_weight = float(np.clip(current_occ_weight, 0.0, 1.0))

        pred_occ_weight = (obs[-2] + 1.0) / 2.0
        pred_occ_weight = float(np.clip(pred_occ_weight, 0.0, 1.0))

        # Płynne skalowanie kar na podstawie wagi zajętości (bez sztywnych ograniczeń!)
        comfort_now_penalty = comfort_now_penalty_raw * current_occ_weight
        comfort_future_penalty = comfort_future_penalty_raw * pred_occ_weight

        # Miękka kara za konflikt akcji (grzanie i chłodzenie zbyt blisko siebie)
        deadband_margin = action[1] - action[0]
        if deadband_margin < 0.5:
            action_conflict_penalty = float((0.5 - deadband_margin) ** 2)
        else:
            action_conflict_penalty = 0.0
        
        # Kara za dynamiczne zmiany setpointów (gładkość)
        smoothing_penalty_val = 0.0
        if self.prev_action is not None:
            smoothing_penalty_val = float(
                ((action[0] - self.prev_action[0]) ** 2) + 
                ((action[1] - self.prev_action[1]) ** 2)
            )
        self.prev_action = np.copy(action)

        # Główne wagi fizyczne (Czyste podejście RL: Skupienie na energii)
        # 1 kW energii karze bardziej niż 1 stopień odchylenia temperatury!
        w_energy = 1.5
        w_comfort_now = 2.0
        w_comfort_future = 1.0
        
        # Wagi pomocnicze kształtujące
        w_conflict = 0.1
        w_smoothing = 0.05
        
        custom_reward = - (
            w_energy * energy_cost + 
            w_comfort_now * comfort_now_penalty + 
            w_comfort_future * comfort_future_penalty +
            w_conflict * action_conflict_penalty +
            w_smoothing * smoothing_penalty_val
        )
       
        # Zapis czystych metryk fizycznych dla ewaluacji
        info['custom_energy_cost'] = energy_cost
        info['custom_comfort_penalty'] = comfort_now_penalty
        info['custom_future_penalty'] = comfort_future_penalty
        info['custom_smoothing_penalty'] = smoothing_penalty_val
        info['action_conflict_penalty'] = action_conflict_penalty

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
        'reward': []
    }
    
    while not (terminated or truncated):
        if mode == ControlMode.RBC_STANDARD:
            # Standardowy termostat: Grzej poniżej 23.0, chłodź powyżej 24.0
            action = np.array([23.0, 24.0], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
        elif mode == ControlMode.RBC_OCCUPANCY:
            # Sprawdzenie obecności - odwracamy normalizację
            try:
                obs_rms = env.get_wrapper_attr('obs_rms')
                epsilon = env.get_wrapper_attr('epsilon')
                mean_occ = obs_rms.mean[occ_idx]
                std_occ = np.sqrt(obs_rms.var[occ_idx] + epsilon)
                occupant_count = float(obs[occ_idx]) * std_occ + mean_occ
            except AttributeError:
                occupant_count = float(obs[occ_idx])
                
            # Jeśli ilość osób >= 0.5, traktujemy to jako obecność ludzi
            if occupant_count >= 0.5:
                # Obecność: Grzej poniżej 23.0, chłodź powyżej 24.0
                action = np.array([23.0, 24.0], dtype=np.float32)
            else:
                # Brak ludzi - wyłączamy grzanie i chłodzenie poprzez maksymalne rozszerzenie deadbandu (12.0 do 30.0)
                action = np.array([12.0, 30.0], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
        else:
            action, _ = model.predict(obs, deterministic=True)

            
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
    env = DualPredictorObservationWrapper(env, lstm_model_path='data/hvac_lstm_temperature_model.pth', occ_path='data/hvac_occupancy_model.pth')
    env = CustomRewardWrapper(env)

    print("Ładowanie wytrenowanego agenta SAC...")
    model = SAC.load("data/sac_hvac_agent_with_lstm", env=env)

    print("\n[1/3] Ewaluacja modelu SAC + LSTM (Proszę czekać, symulacja roczna)...")
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
    rbc_mean_reward = np.mean(rbc_history['reward'])
    rbc_occ_mean_reward = np.mean(rbc_occ_history['reward'])

    os.makedirs('data', exist_ok=True)
    with open('data/rl_vs_baseline_metrics.txt', 'w', encoding='utf-8') as f:
        f.write("--- PORÓWNANIE: SAC+LSTM vs Tradycyjny Termostat (RBC) vs RBC Occupancy ---\n\n")
        f.write("1. Skumulowany koszt energii (Mniej = Lepiej):\n")
        f.write(f"   SAC Agent: {sac_total_energy:.2f}\n")
        f.write(f"   RBC Standard: {rbc_total_energy:.2f}\n")
        f.write(f"   RBC Occupancy: {rbc_occ_total_energy:.2f}\n")
        if rbc_total_energy != 0:
            f.write(f"   Zysk (SAC vs Standard): {((rbc_total_energy - sac_total_energy) / rbc_total_energy * 100):.2f}%\n")
        if rbc_occ_total_energy != 0:
            f.write(f"   Zysk (SAC vs Occupancy): {((rbc_occ_total_energy - sac_total_energy) / rbc_occ_total_energy * 100):.2f}%\n\n")
        else:
            f.write("\n")
        
        f.write("2. Skumulowana kara za dyskomfort cieplny (Mniej = Lepiej):\n")
        f.write(f"   SAC Agent: {sac_total_penalty:.2f}\n")
        f.write(f"   RBC Standard: {rbc_total_penalty:.2f}\n")
        f.write(f"   RBC Occupancy: {rbc_occ_total_penalty:.2f}\n")
        if rbc_total_penalty != 0:
            f.write(f"   Zysk (SAC vs Standard): {((rbc_total_penalty - sac_total_penalty) / rbc_total_penalty * 100):.2f}%\n")
        if rbc_occ_total_penalty != 0:
            f.write(f"   Zysk (SAC vs Occupancy): {((rbc_occ_total_penalty - sac_total_penalty) / rbc_occ_total_penalty * 100):.2f}%\n\n")
        else:
            f.write("\n")
        
        f.write("3. Średnia nagroda na krok (Więcej = Lepiej):\n")
        f.write(f"   SAC Agent: {sac_mean_reward:.4f}\n")
        f.write(f"   RBC Standard: {rbc_mean_reward:.4f}\n")
        f.write(f"   RBC Occupancy: {rbc_occ_mean_reward:.4f}\n")

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(np.cumsum(sac_history['energy']), label="SAC + LSTM", color="royalblue", linewidth=2)
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