import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os
import matplotlib.pyplot as plt
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error

class HVACPredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, num_layers=2):
        super(HVACPredictorLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        pred = self.fc(last_out)
        return pred

class BuildingDataset(Dataset):
    def __init__(self, states, actions, next_states, seq_length=5, indoor_temp_idx=12):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.seq_length = seq_length

        self.inputs = np.concatenate((self.states, self.actions), axis=1)
        self.input_dim = self.inputs.shape[1]
        
        self.indoor_temp_idx = indoor_temp_idx
        
        print(f"Data loaded. LSTM input dimension: {self.input_dim}")
        print(f"LSTM output dimension: 1 (Predicting normalized indoor temperature at index {self.indoor_temp_idx})")

    def __len__(self):
        return len(self.states) - self.seq_length - 1

    def __getitem__(self, idx):
        x = self.inputs[idx : idx + self.seq_length]
        
        target_temp_t1 = self.next_states[idx + self.seq_length - 1, self.indoor_temp_idx]
        target_temp_t2 = self.next_states[idx + self.seq_length, self.indoor_temp_idx]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor([target_temp_t1, target_temp_t2], dtype=torch.float32)


def train_model():
    SEQ_LENGTH = 5
    BATCH_SIZE = 256
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    EPOCHS = 15

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training on device: {device}")

    print("Loading data from data/lstm_dataset.npz...")
    data = np.load('data/lstm_dataset.npz', allow_pickle=True)
    states = data['states']
    actions = data['actions']
    next_states = data['next_states']
    
    if 'state_columns' in data:
        state_columns = data['state_columns'].tolist()
        indoor_temp_idx = state_columns.index('air_temperature')
        print(f"Dynamically found 'air_temperature' at index {indoor_temp_idx}")
    else:
        indoor_temp_idx = 12
        print("Warning: state_columns not found in dataset. Falling back to default index 12.")
    
    # Chronological train/test split (80% / 20%)
    split_idx = int(len(states) * 0.8)
    
    train_states, test_states = states[:split_idx], states[split_idx:]
    train_actions, test_actions = actions[:split_idx], actions[split_idx:]
    train_next_states, test_next_states = next_states[:split_idx], next_states[split_idx:]

    train_dataset = BuildingDataset(train_states, train_actions, train_next_states, seq_length=SEQ_LENGTH, indoor_temp_idx=indoor_temp_idx)
    test_dataset = BuildingDataset(test_states, test_actions, test_next_states, seq_length=SEQ_LENGTH, indoor_temp_idx=indoor_temp_idx)
    
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = HVACPredictorLSTM(
        input_dim=train_dataset.input_dim, 
        hidden_dim=HIDDEN_DIM, 
        num_layers=NUM_LAYERS
    ).to(device)
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining predictive model...")
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

        print(f"Epoch {epoch+1}/{EPOCHS}, Average loss (MSE): {total_loss / len(dataloader):.6f}")

    print(f"Training completed in: {(time.time() - start_time):.2f} s.")
    
    torch.save(model.state_dict(), 'data/hvac_lstm_temperature_model.pth')
    print("Saved model as 'hvac_lstm_temperature_model.pth'.")


    print("\nEvaluating model...")
    model.eval()
    
    # Use eval_dataloader which has test data
    
    all_y_true = []
    all_y_lstm = []
    all_y_naive = []

    with torch.no_grad():
        for batch_x, batch_y in eval_dataloader:
            all_y_true.append(batch_y.cpu().numpy())
            
            pred_lstm = model(batch_x.to(device)).cpu().numpy()
            all_y_lstm.append(pred_lstm)
            
            last_indoor_temp = batch_x[:, -1, test_dataset.indoor_temp_idx].unsqueeze(-1).cpu().numpy()
            all_y_naive.append(np.repeat(last_indoor_temp, 2, axis=1))

    y_true = np.concatenate(all_y_true, axis=0).flatten()
    y_lstm = np.concatenate(all_y_lstm, axis=0).flatten()
    y_naive = np.concatenate(all_y_naive, axis=0).flatten()
    
    global_mean = y_true.mean()
    y_mean = np.full_like(y_true, global_mean)
    
    mae_lstm = mean_absolute_error(y_true, y_lstm)
    mse_lstm = mean_squared_error(y_true, y_lstm)
    
    mae_naive = mean_absolute_error(y_true, y_naive)
    mse_naive = mean_squared_error(y_true, y_naive)
    
    mae_mean = mean_absolute_error(y_true, y_mean)
    mse_mean = mean_squared_error(y_true, y_mean)
    
    print("\n--- METRICS SUMMARY (sktime) ---")
    print(f"LSTM       - MAE: {mae_lstm:.4f}, MSE: {mse_lstm:.4f}")
    print(f"Naive Last - MAE: {mae_naive:.4f}, MSE: {mse_naive:.4f}")
    print(f"Mean       - MAE: {mae_mean:.4f}, MSE: {mse_mean:.4f}")
    
    os.makedirs('data', exist_ok=True)
    with open('data/evaluation_metrics.txt', 'w', encoding='utf-8') as f:
        f.write("--- EVALUATION METRICS ---\n")
        f.write(f"LSTM       - MAE: {mae_lstm:.4f}, MSE: {mse_lstm:.4f}\n")
        f.write(f"Naive Last - MAE: {mae_naive:.4f}, MSE: {mse_naive:.4f}\n")
        f.write(f"Mean       - MAE: {mae_mean:.4f}, MSE: {mse_mean:.4f}\n")
    
    y_true_t1 = y_true[::2]
    y_lstm_t1 = y_lstm[::2]
    y_naive_t1 = y_naive[::2]
    
    y_true_t2 = y_true[1::2]
    y_lstm_t2 = y_lstm[1::2]
    y_naive_t2 = y_naive[1::2]
    
    mae_lstm_t1 = mean_absolute_error(y_true_t1, y_lstm_t1)
    mae_lstm_t2 = mean_absolute_error(y_true_t2, y_lstm_t2)
    mae_naive_t1 = mean_absolute_error(y_true_t1, y_naive_t1)
    mae_naive_t2 = mean_absolute_error(y_true_t2, y_naive_t2)
    
    mse_lstm_t1 = mean_squared_error(y_true_t1, y_lstm_t1)
    mse_lstm_t2 = mean_squared_error(y_true_t2, y_lstm_t2)
    mse_naive_t1 = mean_squared_error(y_true_t1, y_naive_t1)
    mse_naive_t2 = mean_squared_error(y_true_t2, y_naive_t2)
    
    plot_limit = min(672, len(y_true_t1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Subplot 1: T+1
    ax1.plot(y_true_t1[:plot_limit], label="Ground Truth (T+1)", color="black", linewidth=2.5)
    ax1.plot(y_lstm_t1[:plot_limit], label=f"LSTM T+1 (MSE: {mse_lstm_t1:.3f})", color="royalblue", alpha=0.9, linewidth=1.5)
    ax1.plot(y_naive_t1[:plot_limit], label=f"Naive Last T+1 (MSE: {mse_naive_t1:.3f})", color="crimson", linestyle="--", alpha=0.8)
    ax1.set_title(f"Prediction 1 step ahead (T+1) - First {plot_limit} steps (approx {plot_limit/96:.1f} days)")
    ax1.set_ylabel("Normalized Indoor Temp")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: T+2
    ax2.plot(y_true_t2[:plot_limit], label="Ground Truth (T+2)", color="black", linewidth=2.5)
    ax2.plot(y_lstm_t2[:plot_limit], label=f"LSTM T+2 (MSE: {mse_lstm_t2:.3f})", color="darkorange", alpha=0.9, linewidth=1.5)
    ax2.plot(y_naive_t2[:plot_limit], label=f"Naive Last T+2 (MSE: {mse_naive_t2:.3f})", color="crimson", linestyle="--", alpha=0.8)
    ax2.set_title(f"Prediction 2 steps ahead (T+2) - First {plot_limit} steps")
    ax2.set_xlabel("Simulation steps (1 step = 15 minutes)")
    ax2.set_ylabel("Normalized Indoor Temp")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/evaluation_plot.png', dpi=300)
    plt.close()
    
    print("\n[✔] Saved report 'data/evaluation_metrics.txt'.")
    print("[✔] Saved plot 'data/evaluation_plot.png'.\n")

if __name__ == "__main__":
    train_model()