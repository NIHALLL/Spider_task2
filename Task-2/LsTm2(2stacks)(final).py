import requests
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import zipfile, io
import matplotlib.pyplot as plt




def download_climate_dataSet(save_path="jena_climate_2009_2016.csv"):
    if os.path.exists(save_path):
        print("Dataset alderdy exists")
        return save_path
    print("downloading the dataset")
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(".")
    print("Download complete.")
    return save_path

def load_and_preprocess(path):

    # Read CSV
    df = pd.read_csv(path)

    # Convert Date Time column to datetime
    df["Date Time"] = pd.to_datetime(
        df["Date Time"],
        format="%d.%m.%Y %H:%M:%S"
    )

    # Make Date Time the index
    df = df.set_index("Date Time")

    # Convert 10-minute data to hourly data
    df = df.resample("1h").mean()

    # Remove missing values
    df = df.dropna()

    # Convert DataFrame to NumPy array
    data = df.values.astype(np.float32)

    # Store the index of the temperature column
    temp_col_idx = df.columns.get_loc("T (degC)")

    # Chronological split
    n = len(data)

    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    # slicing btw 
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Normalize using training statistics only
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    std[std == 0] = 1.0

    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    test_data = (test_data - mean) / std

    return (
        train_data,
        val_data,
        test_data,
        mean,
        std,
        temp_col_idx,
    )
# Download dataset
path = download_climate_dataSet()

# Load and preprocess dataset
train_data, val_data, test_data, mean, std, temp_col_idx = load_and_preprocess(path)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # Forget Gate
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Input Gate
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Candidate Cell State
        self.candidate_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Output Gate
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev, c_prev):

        # Concatenate current input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)

        # Gates
        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.input_gate(combined))
        g_t = torch.tanh(self.candidate_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))

        # Cell State Update
        c_t = (f_t * c_prev) + (i_t * g_t)

        # Hidden State Update
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
class CustomLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size

        # One custom LSTM cell
        self.lstm_cell1 = LSTMCell(input_size, hidden_size)
        self.lstm_cell2 = LSTMCell(hidden_size, hidden_size)

        # Fully Connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        # x shape = (batch_size, sequence_length, input_size)

        batch_size = x.size(0)
        sequence_length = x.size(1)

        # Initialize hidden state and cell state
        h1 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c1 = torch.zeros(batch_size, self.hidden_size, device=x.device)

        h2 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c2 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        # Process each timestep
        for t in range(sequence_length):
        
            x_t = x[:, t, :]

            h1, c1 = self.lstm_cell1(x_t, h1, c1)

            h2, c2 = self.lstm_cell2(h1, h2, c2)
        # Predict output using the final hidden state
        output = self.fc(h2)
        return output

class JenaDataset(Dataset):

    def __init__(self, data, temp_col_idx, input_window=72, output_window=12):

        self.data = data
        self.temp_col_idx = temp_col_idx
        self.input_window = input_window
        self.output_window = output_window

    def __len__(self):

        return len(self.data) - self.input_window - self.output_window + 1

    def __getitem__(self, idx):

        # Input sequence (72 hours)
        x = self.data[idx : idx + self.input_window]

        # Target sequence (next 12 temperatures)
        y = self.data[
            idx + self.input_window :
            idx + self.input_window + self.output_window,
            self.temp_col_idx
        ]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y
# 

# Create Dataset objects
train_dataset = JenaDataset(train_data, temp_col_idx)

val_dataset = JenaDataset(val_data, temp_col_idx)

test_dataset = JenaDataset(test_data, temp_col_idx)


# Batch size
batch_size = 64


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# Verify the shapes
x, y = next(iter(train_loader))

print("Input Shape :", x.shape)
print("Target Shape:", y.shape)
# 

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {device}")
input_size = train_data.shape[1]
hidden_size = 128
output_size = 12

model = CustomLSTM(
    input_size,
    hidden_size,
    output_size
).to(device)

criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
num_epochs = 30
best_val_loss = float("inf")
train_losses = []
val_losses = []

for epoch in range(num_epochs):

    model.train()
    total_train_loss = 0.0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()
        # imp asf , to save exploding gradients 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item() * x.size(0)

    train_loss = total_train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)


    # Validation
    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            predictions = model(x)

            loss = criterion(predictions, y)

            total_val_loss += loss.item() * x.size(0)

    val_loss = total_val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_lstm.pth")

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )

print(f"\nBest Validation Loss: {best_val_loss:.6f}")

# Load the best saved model
model.load_state_dict(torch.load("best_lstm.pth", map_location=device))
model.eval()

test_huber_loss = 0.0
all_predictions = []
all_targets = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)
        y = y.to(device)

        predictions = model(x)

        loss = criterion(predictions, y)

        test_huber_loss += loss.item() * x.size(0)

        all_predictions.append(predictions.cpu())

        all_targets.append(y.cpu())

all_predictions = torch.cat(all_predictions, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# LOSSESSSSS IMPPPP-------------->>>>>>>>>>>>>>>>>>>--------
test_huber_loss /= len(test_loader.dataset)
mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
mse = torch.mean((all_predictions - all_targets) ** 2).item()
# de-normalize BEFORE computing MAE/MSE
temp_mean = mean[temp_col_idx]
temp_std = std[temp_col_idx]
predictions = all_predictions.numpy() * temp_std + temp_mean
targets = all_targets.numpy() * temp_std + temp_mean

mae = np.mean(np.abs(predictions - targets))
mse = np.mean((predictions - targets) ** 2)

print(f"Test Huber Loss : {test_huber_loss:.6f}")
print(f"Test MAE        : {mae:.6f}")
print(f"Test MSE        : {mse:.6f}")

# FORECAST PREDICTION
forecast_length = 12
plt.figure(figsize=(8,4))

plt.plot(
    range(1,13),
    targets[:12],
    marker='o',
    label="Actual"
)

plt.plot(
    range(1,13),
    predictions[:12],
    marker='x',
    label="Predicted"
)

plt.title("Forecast Example")
plt.xlabel("Hour Ahead")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)

plt.show()

predictions = predictions.flatten()
targets = targets.flatten()

plt.figure(figsize=(14,6))
plt.plot(targets[:300], label="Actual Temperature")
plt.plot(predictions[:300], label="Predicted Temperature")
plt.title("Predicted vs Actual Temperature")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(8,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Huber Loss")
plt.legend()
plt.grid(True)
plt.show()