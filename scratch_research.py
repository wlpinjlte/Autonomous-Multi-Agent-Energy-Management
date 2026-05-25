import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import DatetimeWrapper
import datetime

env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
env = DatetimeWrapper(env)

print("Starting research...")
for ep in range(2):
    obs, info = env.reset()
    print(f"\n--- Episode {ep+1} ---")
    print(f"RESET INFO: {info}")
    
    # Run for 7 days (7 * 24 * 4 = 672 steps)
    for i in range(672):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check first few steps to see the starting day
        if i < 4:
            month = info.get('month', 1)
            day = info.get('day', 1)
            hour = info.get('hour', 0)
            
            # Sinergym actually uses EnergyPlus. 
            # In EnergyPlus, the start day of the week can be defined in the IDF file.
            # Let's see if we can deduce the day of the week from the noise wrapper logic.
            dt = datetime.date(2001, month, day)
            is_weekend_2001 = dt.weekday() >= 5
            
            print(f"Step {i}: Month={month}, Day={day}, Hour={hour}, is_weekend_2001={is_weekend_2001}")
    
env.close()
