import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/lstm_dataset.npz')
states = data['states']

# Occupancy is at index 14
occupancy = states[:1344, 14] # 14 days * 96 steps

plt.figure(figsize=(15, 5))
plt.plot(occupancy, label='Occupancy (Noisy)')
plt.axvline(x=96*5, color='r', linestyle='--', label='End of Day 5')
plt.axvline(x=96*7, color='g', linestyle='--', label='End of Day 7')
plt.axvline(x=96*12, color='r', linestyle='--')
plt.axvline(x=96*14, color='g', linestyle='--')
plt.legend()
plt.title('Occupancy for First 14 Days (1344 steps)')
plt.savefig('scratch/occupancy_14days.png')
print("Saved plot to scratch/occupancy_14days.png")
