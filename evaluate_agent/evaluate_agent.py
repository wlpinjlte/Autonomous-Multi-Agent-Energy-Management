import sys
import os
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
        import os
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
        
        # 3. Increase temperature strictness
        target_temp = 23.5
        deadband = 0.5
        
        diff_now = current_temp - target_temp
        diff_t1 = future_temp_t1 - target_temp
        diff_t2 = future_temp_t2 - target_temp
        
        # Option A: Asymmetric penalty
        if current_occ_weight > 0.0:
            delta_now = float(max(0.0, abs(diff_now) - deadband))
        else:
            delta_now = float(max(0.0, abs(diff_now) - deadband)) if diff_now < 0 else 0.0
            
        if pred_occ_t1_weight > 0.0:
            delta_t1 = float(max(0.0, abs(diff_t1) - deadband))
        else:
            delta_t1 = float(max(0.0, abs(diff_t1) - deadband)) if diff_t1 < 0 else 0.0
            
        if pred_occ_t2_weight > 0.0:
            delta_t2 = float(max(0.0, abs(diff_t2) - deadband))
        else:
            delta_t2 = float(max(0.0, abs(diff_t2) - deadband)) if diff_t2 < 0 else 0.0
        
        # We remove w_occ because logic is already included above
        comfort_now_penalty = delta_now
        comfort_future_penalty_t1 = delta_t1
        comfort_future_penalty_t2 = delta_t2
        comfort_future_penalty = (comfort_future_penalty_t1 + comfort_future_penalty_t2) / 2.0

        # 5. Weights balance (From environment)
        custom_reward = - (
            self.w_energy * energy_penalty + 
            self.w_comfort_now * comfort_now_penalty + 
            self.w_comfort_future * comfort_future_penalty
        )
       
        info['custom_energy_cost'] = power_w / 1000.0
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

    env_id = 'Eplus-5zone-hot-continuous-stochastic-v1'
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
        # Set count very high so evaluation doesn't change the normalization
        obs_rms.count = 1e8 
        print("Loaded NormalizeObservation calibration from dataset.")
    except Exception as e:
        print(f"Could not load normalization calibration: {e}")
        
    new_obs_space = Box(low=env.observation_space.low, high=env.observation_space.high, dtype=np.float32)
    env = TransformObservation(env, func=lambda obs: np.array(obs, dtype=np.float32), observation_space=new_obs_space)
    env = DualPredictorObservationWrapper(env, lstm_model_path='data/hvac_lstm_temperature_model.pth', occ_path='data/hvac_occupancy_model.pth')
    env = CustomRewardWrapper(env)
    print("Loading trained SAC agent...")
    model = SAC.load("data/sac_hvac_agent_with_lstm", env=env)

    print("\n[1/3] Evaluating SAC + LSTM model (Please wait, one-year simulation)...")
    sac_history = run_episode(env, model=model, mode=ControlMode.SAC)

    print("\n[2/3] Evaluating traditional RBC thermostat (Standard)...")
    rbc_history = run_episode(env, mode=ControlMode.RBC_STANDARD)

    print("\n[3/3] Evaluating RBC thermostat (Occupancy-dependent)...")
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
        metrics_text = f"--- COMPARISON: SAC+LSTM vs Traditional Thermostat (RBC) vs RBC Occupancy ---\n\n"
        metrics_text += f"1. Cumulative energy cost (Less = Better):\n"
        metrics_text += f"   SAC Agent: {sac_energy:.2f}\n"
        metrics_text += f"   RBC Standard: {rbc_energy:.2f}\n"
        metrics_text += f"   RBC Occupancy: {rbc_occ_energy:.2f}\n"
        metrics_text += f"   Gain (SAC vs Standard): {((rbc_energy - sac_energy) / rbc_energy) * 100:.2f}%\n"
        metrics_text += f"   Gain (SAC vs Occupancy): {((rbc_occ_energy - sac_energy) / rbc_occ_energy) * 100:.2f}%\n\n"
        
        metrics_text += f"2. Cumulative thermal discomfort penalty (Less = Better):\n"
        metrics_text += f"   SAC Agent: {sac_comfort:.2f}\n"
        metrics_text += f"   RBC Standard: {rbc_comfort:.2f}\n"
        metrics_text += f"   RBC Occupancy: {rbc_occ_comfort:.2f}\n"
        metrics_text += f"   Gain (SAC vs Standard): {((rbc_comfort - sac_comfort) / max(rbc_comfort, 1e-5)) * 100:.2f}%\n"
        metrics_text += f"   Gain (SAC vs Occupancy): {((rbc_occ_comfort - sac_comfort) / max(rbc_occ_comfort, 1e-5)) * 100:.2f}%\n\n"
        
        metrics_text += f"3. Mean reward per step (More = Better):\n"
        metrics_text += f"   SAC Agent: {sac_reward:.4f}\n"
        metrics_text += f"   RBC Standard: {rbc_reward:.4f}\n"
        metrics_text += f"   RBC Occupancy: {rbc_occ_reward:.4f}\n\n"
        
        metrics_text += f"4. Penalty Details (Future Penalty):\n"
        metrics_text += f"   SAC Agent: Future={sac_future:.2f}\n"
        metrics_text += f"   RBC Standard: Future={rbc_future:.2f}\n"
        metrics_text += f"   RBC Occupancy: Future={rbc_occ_future:.2f}\n"
        f.write(metrics_text)

    w_sac = np.ones_like(sac_history['temp']) / len(sac_history['temp'])
    w_rbc = np.ones_like(rbc_history['temp']) / len(rbc_history['temp'])
    w_rbc_occ = np.ones_like(rbc_occ_history['temp']) / len(rbc_occ_history['temp'])
    plot_limit = min(1000, len(sac_history['temp']))

    # --- Individual Plots ---
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(sac_history['energy']), label="SAC + LSTM + MLP", color="royalblue", linewidth=2)
    plt.plot(np.cumsum(rbc_history['energy']), label="Standardowe RBC", color="crimson", linestyle="--", linewidth=2)
    plt.plot(np.cumsum(rbc_occ_history['energy']), label="RBC bazujące na obecności", color="forestgreen", linestyle=":", linewidth=2)
    plt.title("Skumulowane zużycie energii (Sterowanie HVAC)")
    plt.xlabel("Kroki symulacji (15 min)")
    plt.ylabel("Energia (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/evaluation_energy.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(sac_history['comfort_penalty']), label="SAC + LSTM + MLP", color="royalblue", linewidth=2)
    plt.plot(np.cumsum(rbc_history['comfort_penalty']), label="Standardowe RBC", color="crimson", linestyle="--", linewidth=2)
    plt.plot(np.cumsum(rbc_occ_history['comfort_penalty']), label="RBC bazujące na obecności", color="forestgreen", linestyle=":", linewidth=2)
    plt.title("Skumulowana kara za dyskomfort termiczny")
    plt.xlabel("Kroki symulacji (15 min)")
    plt.ylabel("Skumulowana kara (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/evaluation_comfort.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(sac_history['temp'], bins=50, alpha=0.6, label="SAC + LSTM + MLP", color="royalblue", weights=w_sac)
    plt.hist(rbc_history['temp'], bins=50, alpha=0.5, label="Standardowe RBC", color="crimson", weights=w_rbc)
    plt.hist(rbc_occ_history['temp'], bins=50, alpha=0.4, label="RBC bazujące na obecności", color="forestgreen", weights=w_rbc_occ)
    plt.title("Rozkład rzeczywistych temperatur wewnątrz")
    plt.xlabel("Temperatura w stopniach Celsjusza (23.5 = idealna)")
    plt.ylabel("Częstotliwość względna")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/evaluation_temperature_dist.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(sac_history['temp'][:plot_limit], label="SAC + LSTM + MLP", color="royalblue", alpha=0.9, linewidth=1.5)
    plt.plot(rbc_history['temp'][:plot_limit], label="Standardowe RBC", color="crimson", linestyle="--", alpha=0.8, linewidth=1.5)
    plt.plot(rbc_occ_history['temp'][:plot_limit], label="RBC bazujące na obecności", color="forestgreen", linestyle=":", alpha=0.8, linewidth=1.5)
    plt.title(f"Profil temperatury wewnątrz (Pierwsze {plot_limit} kroków)")
    plt.xlabel("Kroki symulacji (15 min)")
    plt.ylabel("Temperatura w stopniach Celsjusza")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/evaluation_temperature_profile.png', dpi=300)
    plt.close()

    # --- Combined Plot ---
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(np.cumsum(sac_history['energy']), label="SAC + LSTM + MLP", color="royalblue", linewidth=2)
    plt.plot(np.cumsum(rbc_history['energy']), label="Standardowe RBC", color="crimson", linestyle="--", linewidth=2)
    plt.plot(np.cumsum(rbc_occ_history['energy']), label="RBC bazujące na obecności", color="forestgreen", linestyle=":", linewidth=2)
    plt.title("Skumulowane zużycie energii (Sterowanie HVAC)")
    plt.xlabel("Kroki symulacji (15 min)")
    plt.ylabel("Energia (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(np.cumsum(sac_history['comfort_penalty']), label="SAC + LSTM + MLP", color="royalblue", linewidth=2)
    plt.plot(np.cumsum(rbc_history['comfort_penalty']), label="Standardowe RBC", color="crimson", linestyle="--", linewidth=2)
    plt.plot(np.cumsum(rbc_occ_history['comfort_penalty']), label="RBC bazujące na obecności", color="forestgreen", linestyle=":", linewidth=2)
    plt.title("Skumulowana kara za dyskomfort termiczny")
    plt.xlabel("Kroki symulacji (15 min)")
    plt.ylabel("Skumulowana kara (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.hist(sac_history['temp'], bins=50, alpha=0.6, label="SAC + LSTM + MLP", color="royalblue", weights=w_sac)
    plt.hist(rbc_history['temp'], bins=50, alpha=0.5, label="Standardowe RBC", color="crimson", weights=w_rbc)
    plt.hist(rbc_occ_history['temp'], bins=50, alpha=0.4, label="RBC bazujące na obecności", color="forestgreen", weights=w_rbc_occ)
    plt.title("Rozkład rzeczywistych temperatur wewnątrz")
    plt.xlabel("Temperatura w stopniach Celsjusza (23.5 = idealna)")
    plt.ylabel("Częstotliwość względna")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(sac_history['temp'][:plot_limit], label="SAC + LSTM + MLP", color="royalblue", alpha=0.9, linewidth=1.5)
    plt.plot(rbc_history['temp'][:plot_limit], label="Standardowe RBC", color="crimson", linestyle="--", alpha=0.8, linewidth=1.5)
    plt.plot(rbc_occ_history['temp'][:plot_limit], label="RBC bazujące na obecności", color="forestgreen", linestyle=":", alpha=0.8, linewidth=1.5)
    plt.title(f"Profil temperatury wewnątrz (Pierwsze {plot_limit} kroków)")
    plt.xlabel("Kroki symulacji (15 min)")
    plt.ylabel("Temperatura w stopniach Celsjusza")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/evaluation_rl_vs_rbc.png', dpi=300)
    plt.close()

    print("\n[✔] Saved text report to 'data/rl_vs_baseline_metrics.txt'.")
    print("[✔] Saved individual plots and combined sheet to 'data/'.\n")

if __name__ == "__main__":
    main()