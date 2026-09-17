"""
=============================================================
RetailPulse – Week 1, Day 6
LSTM Model for Demand Forecasting (PyTorch Lightning)
=============================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
import mlflow, joblib, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

PLOTS_DIR  = Path("../reports/plots")
MODELS_DIR = Path("../models")
plt.style.use("seaborn-v0_8-whitegrid")

# ── HYPERPARAMS ───────────────────────────────────────────────
SEQ_LEN    = 30    # look-back window (days)
HORIZON    = 30    # forecast horizon
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT    = 0.2
LR         = 1e-3
EPOCHS     = 50
BATCH_SIZE = 32

# ── 1. LOAD DATA ─────────────────────────────────────────────
daily = pd.read_parquet("../data/daily_sales.parquet")
values = daily["Revenue"].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# ── 2. SEQUENCE DATASET ──────────────────────────────────────
class RetailDataset(Dataset):
    def __init__(self, data, seq_len):
        self.X, self.y = [], []
        for i in range(len(data) - seq_len):
            self.X.append(data[i : i + seq_len])
            self.y.append(data[i + seq_len])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

split_idx = len(scaled) - HORIZON
train_data = scaled[:split_idx]
test_data  = scaled[split_idx - SEQ_LEN:]

train_ds = RetailDataset(train_data, SEQ_LEN)
test_ds  = RetailDataset(test_data,  SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ── 3. LSTM MODEL ─────────────────────────────────────────────
class LSTMForecaster(pl.LightningModule):
    def __init__(self, input_size=1, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT, lr=LR):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, 1)
        self.loss = nn.MSELoss()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def training_step(self, batch, _):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ── 4. TRAIN ─────────────────────────────────────────────────
mlflow.set_experiment("demand_forecasting")
with mlflow.start_run(run_name="lstm_baseline"):
    model = LSTMForecaster()
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=5,
    )
    trainer.fit(model, train_loader, test_loader)

    # ── 5. EVALUATE ──────────────────────────────────────────
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            p = model(x).numpy()
            preds.extend(p.flatten())
            actuals.extend(y.numpy().flatten())

    preds   = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1,1)).flatten()

    mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-9))) * 100
    rmse = np.sqrt(np.mean((actuals - preds)**2))
    mae  = np.mean(np.abs(actuals - preds))

    mlflow.log_params({"seq_len": SEQ_LEN, "hidden_dim": HIDDEN_DIM,
                       "num_layers": NUM_LAYERS, "epochs": EPOCHS})
    mlflow.log_metrics({"MAPE": round(mape,2), "RMSE": round(rmse,2), "MAE": round(mae,2)})

    print(f"\n=== LSTM Results ===")
    print(f"MAPE : {mape:.2f}%")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(actuals, label="Actual",    color="#2ecc71", linewidth=2)
    ax.plot(preds,   label="LSTM Pred", color="#e74c3c", linewidth=2, linestyle="--")
    ax.set_title("LSTM Demand Forecast – Test Set (30 days)")
    ax.set_xlabel("Day"); ax.set_ylabel("Revenue (£)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"11_lstm_forecast.png", dpi=150); plt.close()
    print("→ Saved: 11_lstm_forecast.png")

# ── 6. SAVE ───────────────────────────────────────────────────
torch.save(model.state_dict(), MODELS_DIR/"lstm_model.pt")
joblib.dump(scaler, MODELS_DIR/"lstm_scaler.pkl")
print("\n✅  Week 1, Day 6 COMPLETE – LSTM model trained")
