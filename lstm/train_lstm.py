import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os
import matplotlib.pyplot as plt
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error

class HVACPredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
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
    def __init__(self, npz_file, seq_length=5):
        print(f"Loading data from {npz_file}...")
        data = np.load(npz_file)
        self.states = data['states']
        self.actions = data['actions']
        self.next_states = data['next_states']
        self.seq_length = seq_length

        self.inputs = np.concatenate((self.states, self.actions), axis=1)
        self.input_dim = self.inputs.shape[1]
        
        self.indoor_temp_idx = 8
        
        print(f"Data loaded. LSTM input dimension: {self.input_dim}")
        print(f"LSTM output dimension: 1 (Predicting normalized indoor temperature only)")

    def __len__(self):
        return len(self.states) - self.seq_length

    def __getitem__(self, idx):
        x = self.inputs[idx : idx + self.seq_length]
        
        target_temp = self.next_states[idx + self.seq_length - 1, self.indoor_temp_idx]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor([target_temp], dtype=torch.float32)


def train_model():
    SEQ_LENGTH = 5
    BATCH_SIZE = 256
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    EPOCHS = 15

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training on device: {device}")

    dataset = BuildingDataset('data/lstm_dataset.npz', seq_length=SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HVACPredictorLSTM(
        input_dim=dataset.input_dim, 
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
    
    eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_y_true = []
    all_y_lstm = []
    all_y_naive = []

    with torch.no_grad():
        for batch_x, batch_y in eval_dataloader:
            all_y_true.append(batch_y.cpu().numpy())
            
            pred_lstm = model(batch_x.to(device)).cpu().numpy()
            all_y_lstm.append(pred_lstm)
            
            last_indoor_temp = batch_x[:, -1, 8].unsqueeze(-1).cpu().numpy()
            all_y_naive.append(last_indoor_temp)

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
    
    plot_limit = min(300, len(y_true))
    plt.figure(figsize=(14, 6))
    
    plt.plot(y_true[:plot_limit], label="Ground Truth", color="black", linewidth=2.5)
    plt.plot(y_lstm[:plot_limit], label=f"LSTM (MAE: {mae_lstm:.3f})", color="royalblue", alpha=0.9, linewidth=1.5)
    plt.plot(y_naive[:plot_limit], label=f"Naive Last (MAE: {mae_naive:.3f})", color="crimson", linestyle="--", alpha=0.8)
    plt.plot(y_mean[:plot_limit], label=f"Mean Baseline", color="forestgreen", linestyle=":", alpha=0.8, linewidth=2)
    
    plt.title(f"Comparison of predictive models: LSTM vs Baseline (First {plot_limit} steps)")
    plt.xlabel("Time steps")
    plt.ylabel("Normalized Indoor Temperature")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/evaluation_plot.png', dpi=300)
    plt.close()
    
    print("\n[✔] Saved report 'data/evaluation_metrics.txt'.")
    print("[✔] Saved plot 'data/evaluation_plot.png'.\n")

if __name__ == "__main__":
    train_model()