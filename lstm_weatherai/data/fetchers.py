import pandas as pd
import requests

def fetch_open_meteo_history(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m,precipitation",
        "timezone": "UTC"
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    d = r.json()["hourly"]

    df = pd.DataFrame(d)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    return df.rename(columns={
        "temperature_2m": "temp",
        "relativehumidity_2m": "rh",
        "pressure_msl": "pressure",
        "windspeed_10m": "wind",
        "precipitation": "precip"
    })

def fetch_bmkg_malang():
    url = "https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4=35.07.18.2011"

    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    if "data" not in data or not isinstance(data["data"], list) or len(data["data"]) == 0:
        raise RuntimeError(f"BMKG: Unexpected structure (no data list). Keys: {list(data.keys())}")

    entry = data["data"][0]

    if "cuaca" not in entry or not isinstance(entry["cuaca"], list):
        raise RuntimeError("BMKG: No cuaca list found")
    cuaca_data = entry["cuaca"]
    if cuaca_data and isinstance(cuaca_data[0], list):
        cuaca_data = [item for sublist in cuaca_data for item in sublist]
    
    df = pd.DataFrame(cuaca_data)

    if "local_datetime" in df.columns:
        dt_col = "local_datetime"
    elif "utc_datetime" in df.columns:
        dt_col = "utc_datetime"
    else:
        raise RuntimeError(f"BMKG: No datetime column found. Columns: {df.columns.tolist()}")

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.set_index(dt_col)

    rename_map = {
        "t": "temp",             
        "hu": "rh",               
        "ws": "wind_speed",      
        "wd": "wind_dir",         
        "tp": "precip",           
        "tcc": "cloud_cover",     
        "vs_text": "visibility", 
        "weather_desc": "weather",
        "weather_desc_en": "weather_en"
    }

    df = df.rename(columns=rename_map)
    numeric_cols = ["temp", "rh", "wind_speed", "precip", "cloud_cover"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "precip" not in df.columns:
        df["precip"] = 0.0


    keep = ["temp", "precip", "rh", "wind_speed", "cloud_cover"]
    keep = [k for k in keep if k in df.columns]

    return df[keep]
