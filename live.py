"""Live air quality data (Open-Meteo / CAMS, no API key) -> same features the model was trained on."""
import numpy as np, pandas as pd, requests

# Quick-pick cities (used for the live backtest and as a search fallback)
COORDS = {"Delhi": (28.61, 77.21), "Mumbai": (19.08, 72.88), "Kolkata": (22.57, 88.36),
          "Chennai": (13.08, 80.27), "Bengaluru": (12.97, 77.59), "Hyderabad": (17.39, 78.49),
          "Lucknow": (26.85, 80.95), "Ahmedabad": (23.02, 72.57), "Patna": (25.59, 85.14),
          "Jaipur": (26.91, 75.79), "Bhopal": (23.26, 77.41), "Chandigarh": (30.73, 76.78),
          "Gurugram": (28.46, 77.03), "Amritsar": (31.63, 74.87), "Guwahati": (26.14, 91.74),
          "Visakhapatnam": (17.69, 83.22), "Thiruvananthapuram": (8.52, 76.94)}

# Approximate CPCB (India) breakpoints: pollutant concentration -> sub-index
IDX = [0, 50, 100, 200, 300, 400, 500]
BREAKS = {"PM2.5": [0, 30, 60, 90, 120, 250, 380], "PM10": [0, 50, 100, 250, 350, 430, 550],
          "NO2": [0, 40, 80, 180, 280, 400, 500], "SO2": [0, 40, 80, 380, 800, 1600, 2000],
          "CO": [0, 1, 2, 10, 17, 34, 50],          # mg/m3
          "O3": [0, 50, 100, 168, 208, 748, 900]}


def geocode(name, count=5):
    """Place name -> candidate locations in India (Open-Meteo geocoding, backed by GeoNames)."""
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search", timeout=15,
                     params={"name": name, "count": count, "country_code": "IN",
                             "language": "en", "format": "json"})
    r.raise_for_status()
    return r.json().get("results", []) or []


def fetch_hourly(lat, lon, past_days=30):
    r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=25, params={
        "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
        "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
        "past_days": past_days, "forecast_days": 2})
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    return h


def daily_table(h):
    h = h.set_index("time")
    d = pd.DataFrame({"PM2.5": h.pm2_5, "PM10": h.pm10, "NO2": h.nitrogen_dioxide,
                      "SO2": h.sulphur_dioxide, "CO": h.carbon_monoxide / 1000,  # ug/m3 -> mg/m3
                      "O3": h.ozone}).resample("D").mean()
    subs = [pd.Series(np.interp(d[k], BREAKS[k], IDX), index=d.index) for k in BREAKS]
    d["AQI"] = pd.concat(subs, axis=1).max(axis=1)
    d.index.name = "Date"
    return d


def build_rows(d):
    """Same features as training (city-agnostic model: no city_code)."""
    d = d.copy()
    for k in (1, 2, 3, 7):
        d[f"AQI_lag_{k}"] = d["AQI"].shift(k)
    s = d["AQI"].shift(1)
    d["AQI_roll_3d"] = s.rolling(3, min_periods=1).mean()
    d["AQI_roll_7d"] = s.rolling(7, min_periods=1).mean()
    d["AQI_roll_7d_std"] = s.rolling(7, min_periods=1).std()
    i = d.index
    d["Month"], d["DayOfWeek"] = i.month, i.dayofweek
    d["Is_Weekend"] = (i.dayofweek >= 5).astype(int)
    d["Is_Winter"] = i.month.isin([11, 12, 1, 2]).astype(int)
    d["Is_Festival_Season"] = (((i.month == 10) & (i.day >= 15)) | ((i.month == 11) & (i.day <= 15))).astype(int)
    return d


def best_window(h, date, width=2):
    """Cleanest `width`-hour window between 5:00 and 21:00 on `date`, plus the dirtiest hour."""
    x = h.set_index("time")["pm2_5"]
    day = x[(x.index.date == date) & (x.index.hour >= 5) & (x.index.hour <= 21)]
    r = day.rolling(width).mean().dropna()
    if r.empty:
        return None
    t = r.idxmin()
    return {"start": t - pd.Timedelta(hours=width - 1), "end": t + pd.Timedelta(hours=1),
            "pm_best": float(r.min()), "peak_hour": day.idxmax(), "pm_peak": float(day.max())}


def backtest(h, predict, feats, today):
    """Predict day d+1 from day d over the fetched past days; compare with what was observed next."""
    d = build_rows(daily_table(h))
    d["target"] = d["AQI"].shift(-1)
    d = d[d.index < today].dropna(subset=feats + ["target"])
    if d.empty:
        return None
    p = np.asarray(predict(d[feats]))
    return {"days": len(d), "mae_model": float(np.mean(np.abs(p - d["target"]))),
            "mae_baseline": float(np.mean(np.abs(d["AQI"] - d["target"])))}
