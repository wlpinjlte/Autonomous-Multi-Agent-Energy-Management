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
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

class OccDataset(Dataset):
    def __init__(self, states, next_states, occ_idx=14):
        self.states = states
        self.next_states = next_states
        
        self.inputs = self.states
        self.input_dim = self.inputs.shape[1]
        
        self.occ_idx = occ_idx 
        
        print(f"MLP input dimension: {self.input_dim}")
        print(f"Occupancy prediction from index: {self.occ_idx}")

    def __len__(self):
        return len(self.inputs) - 1

    def __getitem__(self, idx):
        x = self.inputs[idx]
        target_occ_t1 = self.next_states[idx, self.occ_idx]
        target_occ_t2 = self.next_states[idx + 1, self.occ_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor([target_occ_t1, target_occ_t2], dtype=torch.float32)

def train_occupancy_model():
    BATCH_SIZE = 256
    EPOCHS = 15
    LEARNING_RATE = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target device: {device}")
    
    print(f"Loading data from data/lstm_dataset.npz for occupancy predictor...")
    data = np.load('data/lstm_dataset.npz', allow_pickle=True)
    states = data['states']
    next_states = data['next_states']
    
    if 'state_columns' in data:
        state_columns = data['state_columns'].tolist()
        occ_idx = state_columns.index('people_occupant')
        print(f"Dynamically found 'people_occupant' at index {occ_idx}")
    else:
        occ_idx = 14
        print("Warning: no state_columns in dataset. Using default index 14.")
    
    # Chronological train/test split (80% / 20%)
    split_idx = int(len(states) * 0.8)
    
    train_states, test_states = states[:split_idx], states[split_idx:]
    train_next_states, test_next_states = next_states[:split_idx], next_states[split_idx:]

    train_dataset = OccDataset(train_states, train_next_states, occ_idx=occ_idx)
    test_dataset = OccDataset(test_states, test_next_states, occ_idx=occ_idx)
    
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = OccupancyPredictorMLP(input_dim=train_dataset.input_dim).to(device)
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining occupancy model...")
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
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss (MSE): {total_loss / len(dataloader):.6f}")

    print(f"Training completed in: {(time.time() - start_time):.2f} s.")

    os.makedirs('data', exist_ok=True)
    torch.save(model.state_dict(), 'data/hvac_occupancy_model.pth')
    print("\nSaved occupancy predictor as 'data/hvac_occupancy_model.pth'.")

    print("\nEvaluating model...")
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
            
            last_occ = batch_x[:, test_dataset.occ_idx].unsqueeze(-1).cpu().numpy()
            all_y_naive.append(np.repeat(last_occ, 2, axis=1))

    y_true = np.concatenate(all_y_true, axis=0).flatten()
    y_mlp = np.concatenate(all_y_mlp, axis=0).flatten()
    y_naive = np.concatenate(all_y_naive, axis=0).flatten()
    
    global_mean = y_true.mean()
    y_mean = np.full_like(y_true, global_mean)
    
    # Split predictions into T+1 and T+2
    y_true_t1 = y_true[::2]
    y_mlp_t1 = y_mlp[::2]
    y_naive_t1 = y_naive[::2]
    y_mean_t1 = y_mean[::2]
    
    y_true_t2 = y_true[1::2]
    y_mlp_t2 = y_mlp[1::2]
    y_naive_t2 = y_naive[1::2]
    y_mean_t2 = y_mean[1::2]
    
    mse_mlp_t1 = mean_squared_error(y_true_t1, y_mlp_t1)
    mse_mlp_t2 = mean_squared_error(y_true_t2, y_mlp_t2)
    
    mse_naive_t1 = mean_squared_error(y_true_t1, y_naive_t1)
    mse_naive_t2 = mean_squared_error(y_true_t2, y_naive_t2)
    
    mse_mean_t1 = mean_squared_error(y_true_t1, y_mean_t1)
    mse_mean_t2 = mean_squared_error(y_true_t2, y_mean_t2)
    
    print("\n--- METRICS SUMMARY (sktime) ---")
    print(f"MLP        - MSE: {mse_mlp_t1:.4f} (T+1), {mse_mlp_t2:.4f} (T+2)")
    print(f"Naive Last - MSE: {mse_naive_t1:.4f} (T+1), {mse_naive_t2:.4f} (T+2)")
    print(f"Naive Mean - MSE: {mse_mean_t1:.4f} (T+1), {mse_mean_t2:.4f} (T+2)")
    
    with open('data/occupancy_evaluation_metrics.txt', 'w', encoding='utf-8') as f:
        f.write("--- OCCUPANCY EVALUATION METRICS ---\n")
        f.write(f"MLP        - MSE: {mse_mlp_t1:.4f} (T+1), {mse_mlp_t2:.4f} (T+2)\n")
        f.write(f"Naive Last - MSE: {mse_naive_t1:.4f} (T+1), {mse_naive_t2:.4f} (T+2)\n")
        f.write(f"Naive Mean - MSE: {mse_mean_t1:.4f} (T+1), {mse_mean_t2:.4f} (T+2)\n")
    
    plot_limit = min(672, len(y_true_t1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Subplot 1: T+1
    ax1.plot(y_true_t1[:plot_limit], label="Wartość rzeczywista (T+1)", color="black", linewidth=2.5)
    ax1.plot(y_mlp_t1[:plot_limit], label=f"MLP T+1 (MSE: {mse_mlp_t1:.3f})", color="royalblue", alpha=0.9, linewidth=1.5)
    ax1.plot(y_naive_t1[:plot_limit], label=f"Naiwne (ostatnia wartość) T+1 (MSE: {mse_naive_t1:.3f})", color="crimson", linestyle="--", alpha=0.8)
    ax1.set_title(f"Predykcja 1 krok w przód (T+1) - Pierwsze {plot_limit} kroków (ok. {plot_limit/96:.1f} dni)")
    ax1.set_ylabel("Znormalizowana liczba osób")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: T+2
    ax2.plot(y_true_t2[:plot_limit], label="Wartość rzeczywista (T+2)", color="black", linewidth=2.5)
    ax2.plot(y_mlp_t2[:plot_limit], label=f"MLP T+2 (MSE: {mse_mlp_t2:.3f})", color="darkorange", alpha=0.9, linewidth=1.5)
    ax2.plot(y_naive_t2[:plot_limit], label=f"Naiwne (ostatnia wartość) T+2 (MSE: {mse_naive_t2:.3f})", color="crimson", linestyle="--", alpha=0.8)
    ax2.set_title(f"Predykcja 2 kroki w przód (T+2) - Pierwsze {plot_limit} kroków")
    ax2.set_xlabel("Kroki symulacji (1 krok = 15 minut)")
    ax2.set_ylabel("Znormalizowana liczba osób")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/occupancy_evaluation_plot.png', dpi=300)
    plt.close()
    
    print("\n[✔] Saved report 'data/occupancy_evaluation_metrics.txt'.")
    print("[✔] Saved plot 'data/occupancy_evaluation_plot.png'.\n")

if __name__ == "__main__":
    train_occupancy_model()