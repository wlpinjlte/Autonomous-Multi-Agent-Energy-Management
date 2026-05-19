import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os
import matplotlib.pyplot as plt
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error

class OccupancyPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(OccupancyPredictorMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class OccDataset(Dataset):
    def __init__(self, states, next_states):
        self.states = states
        self.next_states = next_states
        
        self.inputs = self.states
        self.input_dim = self.inputs.shape[1]
        
        self.occ_idx = 13 
        
        print(f"Wymiar wejściowy MLP: {self.input_dim}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        target_occ = self.next_states[idx, self.occ_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor([target_occ], dtype=torch.float32)

def train_occupancy_model():
    BATCH_SIZE = 256
    EPOCHS = 15
    LEARNING_RATE = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Urządzenie docelowe: {device}")
    
    print(f"Ładowanie danych z data/lstm_dataset.npz dla predyktora zajętości...")
    data = np.load('data/lstm_dataset.npz')
    states = data['states']
    next_states = data['next_states']
    
    # Chronological train/test split (80% / 20%)
    split_idx = int(len(states) * 0.8)
    
    train_states, test_states = states[:split_idx], states[split_idx:]
    train_next_states, test_next_states = next_states[:split_idx], next_states[split_idx:]

    train_dataset = OccDataset(train_states, train_next_states)
    test_dataset = OccDataset(test_states, test_next_states)
    
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = OccupancyPredictorMLP(input_dim=train_dataset.input_dim).to(device)
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTrwa uczenie modelu zajętości...")
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoka {epoch+1}/{EPOCHS}, Strata (MSE): {total_loss / len(dataloader):.6f}")

    print(f"Trening zakończony w: {(time.time() - start_time):.2f} s.")

    os.makedirs('data', exist_ok=True)
    torch.save(model.state_dict(), 'data/hvac_occupancy_model.pth')
    print("\nZapisano predyktor zajętości jako 'data/hvac_occupancy_model.pth'.")

    print("\nEwaluacja modelu...")
    model.eval()
    
    # eval_dataloader is already defined using test_dataset
    
    all_y_true = []
    all_y_mlp = []
    all_y_naive = []

    with torch.no_grad():
        for batch_x, batch_y in eval_dataloader:
            all_y_true.append(batch_y.cpu().numpy())
            
            pred_mlp = model(batch_x.to(device)).cpu().numpy()
            all_y_mlp.append(pred_mlp)
            
            last_occ = batch_x[:, 13].unsqueeze(-1).cpu().numpy()
            all_y_naive.append(last_occ)

    y_true = np.concatenate(all_y_true, axis=0).flatten()
    y_mlp = np.concatenate(all_y_mlp, axis=0).flatten()
    y_naive = np.concatenate(all_y_naive, axis=0).flatten()
    
    global_mean = y_true.mean()
    y_mean = np.full_like(y_true, global_mean)
    
    mae_mlp = mean_absolute_error(y_true, y_mlp)
    mse_mlp = mean_squared_error(y_true, y_mlp)
    
    mae_naive = mean_absolute_error(y_true, y_naive)
    mse_naive = mean_squared_error(y_true, y_naive)
    
    mae_mean = mean_absolute_error(y_true, y_mean)
    mse_mean = mean_squared_error(y_true, y_mean)
    
    print("\n--- PODSUMOWANIE METRYK (sktime) ---")
    print(f"MLP        - MAE: {mae_mlp:.4f}, MSE: {mse_mlp:.4f}")
    print(f"Naive Last - MAE: {mae_naive:.4f}, MSE: {mse_naive:.4f}")
    print(f"Mean       - MAE: {mae_mean:.4f}, MSE: {mse_mean:.4f}")
    
    with open('data/occupancy_evaluation_metrics.txt', 'w', encoding='utf-8') as f:
        f.write("--- METRYKI EWALUACJI ZAJĘTOŚCI ---\n")
        f.write(f"MLP        - MAE: {mae_mlp:.4f}, MSE: {mse_mlp:.4f}\n")
        f.write(f"Naive Last - MAE: {mae_naive:.4f}, MSE: {mse_naive:.4f}\n")
        f.write(f"Mean       - MAE: {mae_mean:.4f}, MSE: {mse_mean:.4f}\n")
    
    plot_limit = min(300, len(y_true))
    plt.figure(figsize=(14, 6))
    
    plt.plot(y_true[:plot_limit], label="Ground Truth (Rzeczywista Zajętość)", color="black", linewidth=2.5)
    plt.plot(y_mlp[:plot_limit], label=f"MLP (MAE: {mae_mlp:.3f})", color="royalblue", alpha=0.9, linewidth=1.5)
    plt.plot(y_naive[:plot_limit], label=f"Naive Last (MAE: {mae_naive:.3f})", color="crimson", linestyle="--", alpha=0.8)
    plt.plot(y_mean[:plot_limit], label=f"Mean Baseline", color="forestgreen", linestyle=":", alpha=0.8, linewidth=2)
    
    plt.title(f"Porównanie predykcji zajętości: MLP vs Baseline (Pierwsze {plot_limit} kroków)")
    plt.xlabel("Kroki czasowe")
    plt.ylabel("Zajętość")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/occupancy_evaluation_plot.png', dpi=300)
    plt.close()
    
    print("\n[✔] Zapisano raport 'data/occupancy_evaluation_metrics.txt'.")
    print("[✔] Zapisano wykres 'data/occupancy_evaluation_plot.png'.\n")

if __name__ == "__main__":
    train_occupancy_model()