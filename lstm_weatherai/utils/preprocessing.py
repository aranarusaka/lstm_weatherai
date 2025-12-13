import numpy as np
import pandas as pd

def prepare_features(df):
    df = df.copy()
    df = df.sort_index().resample("1H").mean()
    df = df.interpolate().ffill().bfill()

    if "temp" not in df.columns:
        raise RuntimeError("'temp' column missing")
    if "precip" not in df.columns:
        df["precip"] = 0.0
    expected_cols = ["temp", "precip", "rh", "wind_speed", "pressure", "cloud_cover"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
    df["hour_sin"] = np.sin(2*np.pi*df.index.hour/24)
    df["hour_cos"] = np.cos(2*np.pi*df.index.hour/24)
    df["day_sin"] = np.sin(2*np.pi*df.index.dayofyear/365)

    keep_cols = [
        "temp",
        "rh",
        "wind_speed",
        "pressure",
        "cloud_cover",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "precip",
    ]
    return df[keep_cols]

def create_windows(arr, seq_len, horizon):
    X, Y = [], []
    T = len(arr)
    for i in range(T - seq_len - horizon + 1):
        X.append(arr[i:i+seq_len].reshape(-1))
        Y.append(arr[i+seq_len:i+seq_len+horizon][:, [0, -1]].reshape(-1))
    return np.array(X), np.array(Y)
