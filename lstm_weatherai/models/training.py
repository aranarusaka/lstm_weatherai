import os
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler
from utils.preprocessing import prepare_features, create_windows
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, mean_squared_error

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

SEQ_LEN = 72
HORIZON = 24
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
DEVICE = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")

class WeatherSeqDataset(Dataset):
    def __init__(self, arr, seq_len=SEQ_LEN, horizon=HORIZON):
        self.arr = arr.astype(np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.N = len(arr) - seq_len - horizon + 1

    def __len__(self):
        return max(0, self.N)

    def __getitem__(self, idx):
        x = self.arr[idx: idx + self.seq_len] 
        y_temp = self.arr[idx + self.seq_len: idx + self.seq_len + self.horizon, 0]
        y_precip = self.arr[idx + self.seq_len: idx + self.seq_len + self.horizon, -1]
        y = np.stack([y_temp, y_precip], axis=1).reshape(-1)
        return x, y

def train_lstm(df_hist, save_dir, n_features=None, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available in this environment. Install torch to train LSTM.")

    df = prepare_features(df_hist)
    scaler = StandardScaler()
    arr = scaler.fit_transform(df.values)

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))

    X, Y = create_windows(arr, seq_len=SEQ_LEN, horizon=HORIZON)
    if X.shape[0] == 0:
        raise RuntimeError(f"Not enough data to train LSTM: need at least {SEQ_LEN + HORIZON} rows")

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    import torch
    from torch.utils.data import TensorDataset
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    from models.lstm_model import LSTMSeq2Seq
    n_features = arr.shape[1] if n_features is None else n_features
    model = LSTMSeq2Seq(n_features=n_features, hidden_size=128, num_layers=2, horizon=HORIZON).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_temp_mse': [],
        'val_precip_mse': [],
        'val_precip_f1': [],
        'val_precip_precision': [],
        'val_precip_recall': [],
        'val_precip_accuracy': [],
        'val_confusion': [],
    }

    PRECIP_THRESHOLD = 0.1

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.view(xb.size(0), SEQ_LEN, -1).to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
        avg_train_loss = running_loss / max(1, seen)


        model.eval()
        val_running = 0.0
        val_seen = 0
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.view(xb.size(0), SEQ_LEN, -1).to(DEVICE)
                yb = yb.to(DEVICE)
                out = model(xb)
                loss = loss_fn(out, yb)
                val_running += loss.item() * xb.size(0)
                val_seen += xb.size(0)
                all_preds.append(out.cpu().numpy())
                all_trues.append(yb.cpu().numpy())

        avg_val_loss = (val_running / max(1, val_seen)) if val_seen > 0 else None

        if len(all_preds) > 0:
            preds = np.concatenate(all_preds, axis=0).reshape(-1, 2)
            trues = np.concatenate(all_trues, axis=0).reshape(-1, 2)
            F = arr.shape[1]
            dummy_preds = np.zeros((preds.shape[0], F))
            dummy_trues = np.zeros((trues.shape[0], F))
            dummy_preds[:, 0] = preds[:, 0]
            dummy_preds[:, -1] = preds[:, 1]
            dummy_trues[:, 0] = trues[:, 0]
            dummy_trues[:, -1] = trues[:, 1]

            inv_preds = scaler.inverse_transform(dummy_preds)
            inv_trues = scaler.inverse_transform(dummy_trues)

            temp_mse = mean_squared_error(inv_trues[:, 0], inv_preds[:, 0])
            precip_mse = mean_squared_error(inv_trues[:, -1], inv_preds[:, -1])
            y_true_bin = (inv_trues[:, -1] > PRECIP_THRESHOLD).astype(int)
            y_pred_bin = (inv_preds[:, -1] > PRECIP_THRESHOLD).astype(int)
            try:
                f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
                prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
                rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
                acc = accuracy_score(y_true_bin, y_pred_bin)
                cm = confusion_matrix(y_true_bin, y_pred_bin).tolist()
            except Exception:
                f1 = prec = rec = acc = 0.0
                cm = [[0, 0], [0, 0]]
        else:
            temp_mse = precip_mse = None
            f1 = prec = rec = acc = 0.0
            cm = [[0, 0], [0, 0]]

        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(float(avg_train_loss))
        metrics['val_loss'].append(float(avg_val_loss) if avg_val_loss is not None else None)
        metrics['val_temp_mse'].append(float(temp_mse) if temp_mse is not None else None)
        metrics['val_precip_mse'].append(float(precip_mse) if precip_mse is not None else None)
        metrics['val_precip_f1'].append(float(f1))
        metrics['val_precip_precision'].append(float(prec))
        metrics['val_precip_recall'].append(float(rec))
        metrics['val_precip_accuracy'].append(float(acc))
        metrics['val_confusion'].append(cm)

        print(f"[LSTM] Epoch {epoch}/{epochs} train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f} f1={f1:.4f}")

        model.train()

    model.cpu()
    torch.save(model.state_dict(), os.path.join(save_dir, "model_lstm.pt"))
    joblib.dump({'n_features': arr.shape[1], 'seq_len': SEQ_LEN, 'horizon': HORIZON}, os.path.join(save_dir, "model_lstm_meta.joblib"))
    joblib.dump(metrics, os.path.join(save_dir, "training_metrics.joblib"))
    return model

def train_model(df_hist, save_dir):

    if TORCH_AVAILABLE:
        return train_lstm(df_hist, save_dir)
    else:
        raise RuntimeError("Torch not available. Install PyTorch to use the LSTM training path.")
