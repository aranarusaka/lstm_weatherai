from models.prediction import predict_next_day
from models.training import train_model
from data.fetchers import fetch_open_meteo_history, fetch_bmkg_malang
import streamlit as st
import os
from datetime import datetime, timedelta
import joblib
import pandas as pd

st.set_page_config(page_title="Weather Predictor", layout="wide")
st.title("Weather Predictor — LSTM Weather Forecast")

SEQ_LEN = 72
PRED_HORIZON = 24
MODEL_DIR = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)
if "model_loaded" not in st.session_state:
    model_path = os.path.join(MODEL_DIR, "model_lstm.pt")
    if os.path.exists(model_path):
        st.session_state["model_loaded"] = True
        st.success("LSTM model auto-detected ✔ Ready for prediction")
    else:
        st.session_state["model_loaded"] = False

with st.sidebar:
    st.header("Model Controls")

    if st.button("Train Model"):
        if "df_hist" not in st.session_state:
            st.error("Download history first")
        else:
            data = st.session_state["df_hist"]

            if "df_bmkg" in st.session_state:
                data = data.combine_first(st.session_state["df_bmkg"])

            st.info("Training LSTM… (this may take a bit)")
            model = train_model(data, MODEL_DIR)
            st.session_state["model_loaded"] = True
            st.success("Training complete — LSTM saved!")

    if st.button("Run Prediction"):
        if not st.session_state.get("model_loaded", False):
            st.error("No model found — please train first.")
        else:
            source = st.session_state.get("df_hist")
            if source is None:
                st.error("Download historical data first (needed for LSTM context).")
            else:
                if "df_bmkg" in st.session_state:
                    source = source.combine_first(st.session_state["df_bmkg"])
                st.info("Predicting next 24 hours…")
                df_pred = predict_next_day(None, source, MODEL_DIR)
                st.line_chart(df_pred["temp"])
                st.write(df_pred)

    if st.button("View Training Metrics"):
        metrics_path = os.path.join(MODEL_DIR, "training_metrics.joblib")
        if not os.path.exists(metrics_path):
            st.warning("No training metrics found — train the model first.")
        else:
            metrics = joblib.load(metrics_path)
            df_metrics = pd.DataFrame({k: metrics[k] for k in metrics if k != 'val_confusion'})
            st.subheader("Training Metrics — per epoch")
            try:
                st.line_chart(df_metrics.set_index('epoch')[['train_loss', 'val_loss']])
            except Exception:
                pass
            try:
                st.line_chart(df_metrics.set_index('epoch')[['val_precip_f1', 'val_precip_precision', 'val_precip_recall', 'val_precip_accuracy']])
            except Exception:
                pass
            st.write("Per-epoch details:")
            st.dataframe(df_metrics)
            last_cm = metrics.get('val_confusion', [])[-1] if len(metrics.get('val_confusion', [])) > 0 else None
            if last_cm is not None:
                st.subheader("Validation Confusion Matrix (last epoch)")
                cm_df = pd.DataFrame(last_cm, index=['true_0', 'true_1'], columns=['pred_0', 'pred_1'])
                st.table(cm_df)
                
st.header("1) Download Historical Weather (Open-Meteo)")
lat = st.number_input("Latitude", value=-7.99, format="%.6f")
lon = st.number_input("Longitude", value=112.63, format="%.6f")
start = st.date_input("Start", value=(datetime.utcnow() - timedelta(days=365)).date())
end = st.date_input("End", value=(datetime.utcnow() - timedelta(days=1)).date())

if st.button("Download History"):
    try:
        df = fetch_open_meteo_history(lat, lon, start.isoformat(), end.isoformat())
        st.session_state["df_hist"] = df
        st.success(f"Downloaded {len(df)} hourly rows")
        st.dataframe(df.tail())
    except Exception as e:
        st.error(f"Open-Meteo Error: {e}")


st.header("2) Real-Time BMKG Data — Malang")

if st.button("Fetch BMKG Malang"):
    try:
        df = fetch_bmkg_malang()
        st.session_state["df_bmkg"] = df
        st.success(f"Fetched BMKG ({len(df)} rows)")
        st.dataframe(df.tail())
    except Exception as e:
        st.error(f"BMKG Error: {e}")

st.markdown("---")
st.caption("If LSTM is installed, the app trains + predicts using neural networks (PyTorch).")
