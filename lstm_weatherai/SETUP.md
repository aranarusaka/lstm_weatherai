## 1) Clone repository kalau perlu

```powershell
git clone <repo-url>
cd weatherailstm
```

## 2) Buat virtual environment

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 3) Install dependency

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Launching Streamlit

Setelah dependency terdownload semua, jalankan:

```powershell
streamlit run app.py
```



## 5) Alur kerja

- Tekan "Download History"
- (Opsional) Ambil data BMKG untuk membandingkan/kombinasi data lokal.
- Tekan "Train Model", melatih LSTM.
- Tekan "Run Prediction", menghasilkan prakiraan 24 jam dari data terakhir.


