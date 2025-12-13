import os
import joblib
import numpy as np
import pandas as pd
from utils.preprocessing import prepare_features

try:
    import torch
    from models.lstm_model import LSTMSeq2Seq
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

SEQ_LEN = 72
HORIZON = 24

def load_lstm_if_exists(save_dir):
    model_path = os.path.join(save_dir, "model_lstm.pt")
    meta_path = os.path.join(save_dir, "model_lstm_meta.joblib")
    scaler_path = os.path.join(save_dir, "scaler.joblib")
    if os.path.exists(model_path) and os.path.exists(meta_path) and os.path.exists(scaler_path) and TORCH_AVAILABLE:
        meta = joblib.load(meta_path)
        n_features = meta['n_features']
        horizon = meta['horizon']
        model = LSTMSeq2Seq(n_features=n_features, hidden_size=128, num_layers=2, horizon=horizon)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    return None

def predict_next_day(model, df_recent, save_dir):
 
    lstm = load_lstm_if_exists(save_dir)
    if lstm is not None:
        scaler = joblib.load(os.path.join(save_dir, "scaler.joblib"))
        df = prepare_features(df_recent)
        arr = scaler.transform(df.values)
        if len(arr) < SEQ_LEN:
            raise RuntimeError(f"Not enough recent rows for LSTM prediction (need {SEQ_LEN})")
        x = arr[-SEQ_LEN:].astype(np.float32).reshape(1, SEQ_LEN, -1)
        # run model (cpu)
        import torch
        with torch.no_grad():
            out = lstm(torch.tensor(x))
            out = out.numpy().reshape(HORIZON, 2)  # (horizon, 2)
        idx = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=HORIZON, freq='H')
        df_pred = pd.DataFrame({'temp': out[:, 0], 'precip': out[:, 1]}, index=idx)
        # inverse-scaling for temp/precip: we used scaler for all features, so invert per-row
        # Reconstruct dummy arrays to inverse transform
        inv_temps, inv_precips = [], []
        for i in range(HORIZON):
            dummy = np.zeros((1, arr.shape[1]))
            dummy[0, 0] = out[i, 0]
            dummy[0, -1] = out[i, 1]
            inv = scaler.inverse_transform(dummy)
            inv_temps.append(inv[0, 0])
            inv_precips.append(inv[0, -1])
        df_pred['temp'] = inv_temps
        df_pred['precip'] = inv_precips
        # Clamp negative precipitation to zero (model may predict slightly negative due to scaling artifacts)
        df_pred['precip'] = df_pred['precip'].clip(lower=0.0)
        return df_pred

    scaler = joblib.load(os.path.join(save_dir, "scaler.joblib"))
    df = prepare_features(df_recent)
    arr = scaler.transform(df.values)
    if len(arr) < SEQ_LEN:
        raise RuntimeError(f"Not enough recent data for prediction (need at least {SEQ_LEN} hourly rows)")
    x = arr[-SEQ_LEN:].reshape(1, -1)
    pred = model.predict(x).reshape(HORIZON, 2)
    last = df.index[-1]
    idx = pd.date_range(last + pd.Timedelta(hours=1), periods=HORIZON, freq='H')
    return pd.DataFrame({"temp": pred[:, 0], "precip": pred[:, 1]}, index=idx)
