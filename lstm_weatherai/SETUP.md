## 1) Clone repository jika perlu

```powershell
git clone <repo-url>
cd weatherailstm
```

## 2) Buat dan aktifkan virtual environment (PowerShell)

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 3) Instal dependency

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Menjalankan aplikasi Streamlit

Setelah dependency terpasang, jalankan:

```powershell
streamlit run app.py
```

Streamlit membuka halaman di browser

## 5) Alur kerja singkat di aplikasi

- Tekan "Download History" untuk mengambil data historis.
- (Opsional) Ambil data BMKG untuk membandingkan/kombinasi data lokal.
- Tekan "Train Model" untuk melatih LSTM (memakan waktu; disarankan CPU-only untuk percobaan kecil, GPU untuk training nyata).
- Tekan "Run Prediction" untuk menghasilkan prakiraan 24 jam dari data terakhir.
