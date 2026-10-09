import os, time, joblib, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
import xgboost as xgblib
from live import COORDS, fetch_hourly, daily_table, build_rows, best_window, geocode, backtest
from advice import (activity_plan, protection, health_watch,
                    plan_ahead, authority_actions)

st.set_page_config(page_title="Air Quality + Health Advisor", layout="wide")

# ---------- Load ----------
@st.cache_resource
def load_models():
    return (joblib.load("models/xgb_mean_nocity.pkl"), joblib.load("models/xgb_q10_nocity.pkl"),
            joblib.load("models/xgb_q90_nocity.pkl"), joblib.load("models/features_nocity.pkl"))

@st.cache_data
def load_data():
    return pd.read_csv("data/processed.csv", parse_dates=["Date"])

cached_hourly = st.cache_data(ttl=1800)(fetch_hourly)      # live data reused for 30 min
cached_geocode = st.cache_data(ttl=86400)(geocode)

xgb, q10, q90, features = load_models()
df = load_data()
PLACES = {}                                                 # place label -> (lat, lon)

readable = {
    "AQI": "today's AQI", "AQI_lag_1": "yesterday's AQI", "AQI_lag_2": "AQI 2 days ago",
    "AQI_lag_3": "AQI 3 days ago", "AQI_lag_7": "AQI a week ago",
    "AQI_roll_3d": "3-day average AQI", "AQI_roll_7d": "7-day average AQI",
    "AQI_roll_7d_std": "recent AQI volatility", "PM2.5": "PM2.5 level", "PM10": "PM10 level",
    "NO2": "NO2 level", "CO": "CO level", "SO2": "SO2 level", "O3": "ozone level",
    "Month": "time of year", "DayOfWeek": "day of week", "Is_Weekend": "weekend effect",
    "Is_Winter": "winter season", "Is_Festival_Season": "festival season"}

sensitivity = {"Healthy adult": 1.0, "Runner / outdoor exerciser": 1.15, "Child": 1.2,
               "Elderly": 1.3, "Pregnant": 1.3, "Asthma / respiratory condition": 1.5,
               "Outdoor worker": 1.15}

def aqi_category(a):
    for limit, name in [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
                        (300, "Poor"), (400, "Very Poor")]:
        if a <= limit:
            return name
    return "Severe"

COLORS = {"Good": "#2e9e4f", "Satisfactory": "#8bc34a", "Moderate": "#f4c20d",
          "Poor": "#f08c00", "Very Poor": "#e03131", "Severe": "#8b0000"}

def today_ts():
    return pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()

def place_label(r):
    out = []
    for p in (r.get("name"), r.get("admin2"), r.get("admin1")):
        if p and p not in out:
            out.append(p)
    return ", ".join(out)

def resolve(name):
    if name in PLACES:
        return PLACES[name]
    if name in COORDS:
        return COORDS[name]
    res = cached_geocode(name)
    if not res:
        raise ValueError(f"Could not find place: {name}")
    return (res[0]["latitude"], res[0]["longitude"])

# ---------- Forecast functions ----------
def forecast_from_row(row, history):
    pred = float(xgb.predict(row[features])[0])
    lo, hi = float(q10.predict(row[features])[0]), float(q90.predict(row[features])[0])
    lo, hi = min(lo, pred), max(hi, pred)
    c = xgb.get_booster().predict(xgblib.DMatrix(row[features]), pred_contribs=True)[0][:-1]
    order = np.argsort(-np.abs(c))[:6]
    as_of = row.index[0] if row.index.name == "Date" else row["Date"].iloc[0]
    return {"as_of": as_of, "today": float(row["AQI"].iloc[0]), "pred": pred, "low": lo, "high": hi,
            "drivers": [(readable[features[i]], float(c[i])) for i in order], "history": history}

def get_forecast(city, date):
    rows = df[(df["City"] == city) & (df["Date"] <= pd.to_datetime(date))]
    return forecast_from_row(rows.iloc[[-1]], rows.tail(30))

def live_forecast(name):
    lat, lon = resolve(name)
    h = cached_hourly(round(lat, 2), round(lon, 2))
    d = build_rows(daily_table(h))
    d = d[d.index <= today_ts()]
    row = d.dropna(subset=features).iloc[[-1]]
    f = forecast_from_row(row, d.reset_index().tail(30))
    f["hourly"] = h
    return f

def forecast_for(city, date):
    return live_forecast(city) if date == "live" else get_forecast(city, date)

# ---------- Agent tools ----------
def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return key

def tool_get_forecast(city: str, date: str) -> dict:
    """Forecast next-day AQI for a place with a 10-90% range and top factors. date is YYYY-MM-DD, or 'live' for current data."""
    f = forecast_for(city, date)
    return {"as_of": str(pd.to_datetime(f["as_of"]).date()), "today_aqi": round(f["today"]),
            "predicted_aqi": round(f["pred"]), "range_low": round(f["low"]),
            "range_high": round(f["high"]),
            "top_factors": [{"factor": n, "effect": "raises" if v > 0 else "lowers"}
                            for n, v in f["drivers"][:3]]}

def tool_get_personal_risk(profile: str, predicted_aqi: float, range_high: float) -> dict:
    """Personal risk level for a health profile, given the predicted AQI and the upper end of its range."""
    return {"profile": profile, "air_category": aqi_category(predicted_aqi),
            "personal_risk": aqi_category(range_high * sensitivity[profile])}

def tool_get_recent_trend(city: str, date: str) -> dict:
    """AQI for the last 7 days in a place. date is YYYY-MM-DD or 'live'."""
    h = forecast_for(city, date)["history"].tail(7)
    return {"dates": [str(pd.to_datetime(d).date()) for d in h["Date"]],
            "aqi": [round(float(a)) for a in h["AQI"]]}

def tool_get_activity_plan(personal_risk: str) -> dict:
    """Go ahead / Modify / Avoid status for everyday activities at a personal risk level (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)."""
    return {a: f"{s}: {t}" for a, s, t in activity_plan(personal_risk)}

def tool_get_best_window(city: str, date: str) -> dict:
    """Cleanest 2-hour window today and tomorrow for a place, plus the worst hour (only available when date is 'live')."""
    if date != "live":
        return {"note": "Hourly data is only available in live mode."}
    lat, lon = resolve(city)
    h = cached_hourly(round(lat, 2), round(lon, 2))
    out = {}
    for label, day in [("today", today_ts().date()), ("tomorrow", (today_ts() + pd.Timedelta(days=1)).date())]:
        w = best_window(h, day)
        if w:
            out[label] = {"cleanest_window": f"{w['start']:%H:%M}-{w['end']:%H:%M}",
                          "pm25_in_window": round(w["pm_best"]),
                          "worst_hour": f"{w['peak_hour']:%H:%M}", "pm25_at_worst": round(w["pm_peak"])}
    return out

def tool_compare_cities(cities: str, profile: str, date: str) -> dict:
    """Compare next-day forecast and personal risk across places. cities is a comma-separated list such as 'Delhi,Gurugram'."""
    out = {}
    for c in [x.strip() for x in cities.split(",") if x.strip()][:4]:
        try:
            f = forecast_for(c, date)
            out[c] = {"predicted_aqi": round(f["pred"]), "range_high": round(f["high"]),
                      "personal_risk": aqi_category(f["high"] * sensitivity[profile])}
        except Exception:
            out[c] = {"error": "no data available for this place"}
    return out

AGENT_TOOLS = [tool_get_forecast, tool_get_personal_risk, tool_get_recent_trend,
               tool_get_activity_plan, tool_get_best_window, tool_compare_cities]

AGENT_SYSTEM = """You are an air quality health advisor for Indian cities and towns. Use the tools instead of guessing numbers.
You are given the user's selected place, health profile and a date value; pass that date value to tools exactly as given.
The question may mention other places: use tool_compare_cities or call tools for each place.
Typical flow: tool_get_forecast, tool_get_personal_risk, tool_get_activity_plan; when timing matters add tool_get_best_window.
Reply in the language of the question (English or Hindi), in this short format:
ANSWER: <direct answer to the question in 1-2 sentences>
RISK: <personal risk level and forecast with range>
WHY: <one line using the top factors; 'lowers' means lower than typical>
PLAN: <3 concrete steps, based on the activity plan and best window when available>
End with: General guidance, not medical advice.
If a tool says data is unavailable, say so instead of inventing numbers."""

def run_agent(city, profile, date, question):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=get_api_key(), http_options=types.HttpOptions(timeout=90000))
    config = types.GenerateContentConfig(system_instruction=AGENT_SYSTEM, tools=AGENT_TOOLS)
    prompt = (f"Selected place: {city}. Health profile: {profile}. Date value: {date}.\n"
              f"Question: {question}")
    models = ["gemini-3.8-flash"]
    try:
        for m in client.models.list():
            n = m.name.replace("models/", "")
            if "flash" in n and not any(x in n for x in ("image", "tts", "live", "audio")) and n not in models:
                models.append(n)
    except Exception:
        pass
    resp, last_err = None, None
    for model in models[:5]:
        for attempt in range(2):
            try:
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
                break
            except Exception as e:
                last_err = e
                if any(c in str(e) for c in ("503", "504", "429")):
                    time.sleep(3 * (attempt + 1))
                else:
                    break
        if resp is not None:
            break
    if resp is None:
        raise RuntimeError(f"All models busy or unavailable: {str(last_err)[:80]}")
    steps = []
    for item in (getattr(resp, "automatic_function_calling_history", None) or []):
        for part in (item.parts or []):
            if getattr(part, "function_call", None):
                steps.append(part.function_call.name)
    return resp.text, steps

cached_agent = st.cache_data(ttl=6 * 3600, show_spinner=False)(run_agent)

def offline_agent(city, profile, date):
    """Same tool chain without the LLM, used when the API is unavailable."""
    f = tool_get_forecast(city, date)
    r = tool_get_personal_risk(profile, f["predicted_aqi"], f["range_high"])
    t = tool_get_recent_trend(city, date)["aqi"]
    plan = tool_get_activity_plan(r["personal_risk"])
    trend = "worsening" if t[-1] > t[0] * 1.1 else "improving" if t[-1] < t[0] * 0.9 else "stable"
    top = f["top_factors"][0]
    steps = ["tool_get_forecast", "tool_get_personal_risk", "tool_get_recent_trend", "tool_get_activity_plan"]
    text = (f"ANSWER: For {profile} in {city}, risk is {r['personal_risk']} and the air has been {trend} this week.\n"
            f"RISK: {r['personal_risk']}, forecast {f['predicted_aqi']} AQI (range {f['range_low']}-{f['range_high']}).\n"
            f"WHY: The biggest factor is {top['factor']}, which {top['effect']} the forecast.\n"
            "PLAN: " + "; ".join(f"{a}: {s}" for a, s in list(plan.items())[:3]) + ".\n"
            "General guidance, not medical advice.")
    return text, steps

def pretty(text):
    for k in ["ANSWER:", "RISK:", "WHY:", "PLAN:"]:
        text = text.replace(k, f"\n\n**{k}**")
    return text.strip()

# ---------- Validation ----------
@st.cache_data
def hist_validation():
    t = df[df["Date"] >= "2019-01-01"].dropna(subset=features + ["Target_AQI_Next_Day"]).copy()
    y = t["Target_AQI_Next_Day"]
    p, lo, hi = xgb.predict(t[features]), q10.predict(t[features]), q90.predict(t[features])
    t["err"], t["base"] = np.abs(p - y), np.abs(t["AQI"] - y)
    t["cov"] = ((y >= lo) & (y <= hi)).astype(int)
    agg = dict(rows=("err", "size"), MAE_model=("err", "mean"), MAE_baseline=("base", "mean"),
           coverage=("cov", "mean"), mean_AQI=("Target_AQI_Next_Day", "mean"))
    by_city = t.groupby("City").agg(**agg).round(2)
    by_city["MAE_pct_of_mean"] = (by_city["MAE_model"] / by_city["mean_AQI"] * 100).round(1)
    by_season = t.groupby(t["Is_Winter"].map({1: "Winter (Nov-Feb)", 0: "Rest of year"})).agg(**agg).round(2)
    overall = {"MAE model": round(t["err"].mean(), 2), "MAE baseline": round(t["base"].mean(), 2),
               "coverage": round(t["cov"].mean(), 3), "rows": len(t)}
    return by_city, by_season, overall

@st.cache_data(ttl=3600, show_spinner=False)
def live_validation():
    rows = []
    for name, (lat, lon) in COORDS.items():
        try:
            r = backtest(cached_hourly(round(lat, 2), round(lon, 2)),
                         lambda X: xgb.predict(X), features, today_ts())
            if r:
                rows.append({"City": name, "days": r["days"], "MAE_model": round(r["mae_model"], 1),
                             "MAE_baseline": round(r["mae_baseline"], 1)})
        except Exception:
            pass
    return pd.DataFrame(rows)

# ---------- UI ----------
st.title("Air Quality Forecast + Personalized Health Advisor")
st.caption("Next-day AQI forecast with uncertainty range, explanations and an activity plan by health profile. "
           "Supports SDG 3 (Good Health) and SDG 11 (Sustainable Cities).")

mode = st.radio("Data mode", ["Live (estimated, today)", "Historical (2015-2020 station data)"], horizontal=True)
live = mode.startswith("Live")

c1, c2, c3 = st.columns(3)
profile = c2.selectbox("Health profile", list(sensitivity.keys()))
f, date_arg, city = None, "live", None

if live:
    q = c1.text_input("Search any place in India", value="Delhi")
    try:
        results = cached_geocode(q.strip()) if q.strip() else []
    except Exception:
        results = [{"name": k, "latitude": v[0], "longitude": v[1]} for k, v in COORDS.items()
                   if q.strip().lower() in k.lower()]
        c1.caption("Place search unavailable; showing quick-pick cities only.")
    if results:
        labels = [place_label(r) for r in results]
        pick = c1.selectbox("Select match", list(range(len(labels))), format_func=lambda i: labels[i])
        city = labels[pick]
        PLACES[city] = (results[pick]["latitude"], results[pick]["longitude"])
        try:
            f = live_forecast(city)
            c3.markdown("**Forecast as of**")
            c3.write("Live, today")
        except Exception as e:
            st.warning(f"Live data unavailable right now, showing historical data instead. ({str(e)[:80]})")
            live = False
    else:
        c1.warning("No match found. Try another spelling.")
        live = False

if not live:
    cities = sorted(df["City"].unique())
    city = c1.selectbox("City (historical data)", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
    cd = df[df["City"] == city]["Date"]
    default = min(max(pd.Timestamp("2019-11-05"), cd.min()), cd.max()).date()
    date = c3.date_input("Forecast as of", value=default, min_value=cd.min().date(), max_value=cd.max().date())
    date_arg = str(date)
    f = get_forecast(city, date)

air_cat = aqi_category(f["pred"])
personal = aqi_category(f["high"] * sensitivity[profile])

if live:
    st.info("Live mode uses modelled data (Copernicus CAMS via Open-Meteo, about 40 km grid outside Europe), "
            "converted to an approximate Indian AQI. It describes the regional background: nearby places can "
            "look similar, and values can differ from ground-station readings.")

m1, m2, m3 = st.columns(3)
m1.metric("Forecast for next day", f"{f['pred']:.0f} AQI", f"{f['pred'] - f['today']:+.0f} vs today",
          delta_color="inverse")
m2.metric("Likely range (10-90%)", f"{f['low']:.0f} - {f['high']:.0f}")
m3.markdown(f"**Personal risk for {profile}**")
m3.caption("Risk is based on the upper end of the forecast range, to be cautious.")
m3.markdown(f"<div style='background:{COLORS[personal]};color:white;padding:10px;"
            f"border-radius:8px;text-align:center;font-size:20px'>{personal}</div>", unsafe_allow_html=True)
st.write(f"General air quality category: **{air_cat}**  |  Data as of **{pd.to_datetime(f['as_of']).date()}**")
st.write(plan_ahead(f["today"], f["pred"]))

left, right = st.columns(2)
with left:
    st.subheader("Last 30 days and next-day forecast")
    h = f["history"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["Date"], y=h["AQI"], mode="lines+markers", name="Actual AQI"))
    fig.add_trace(go.Scatter(x=[pd.to_datetime(f["as_of"]) + pd.Timedelta(days=1)], y=[f["pred"]],
                             mode="markers", name="Forecast", marker=dict(size=12, color="red"),
                             error_y=dict(type="data", symmetric=False, array=[f["high"] - f["pred"]],
                                          arrayminus=[f["pred"] - f["low"]])))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="AQI")
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Why this forecast?")
    names, vals = [d[0] for d in f["drivers"]][::-1], [d[1] for d in f["drivers"]][::-1]
    fig2 = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                            marker_color=["#e03131" if v > 0 else "#2e9e4f" for v in vals]))
    fig2.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="Effect on forecast vs the average day (red = raises, green = lowers)")
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Recommendations ----------
st.subheader("What should you do?")
if live and "hourly" in f:
    for label, day in [("Today", today_ts().date()), ("Tomorrow", (today_ts() + pd.Timedelta(days=1)).date())]:
        w = best_window(f["hourly"], day)
        if w:
            st.write(f"🕒 **{label}: cleanest window {w['start']:%H:%M}-{w['end']:%H:%M}** "
                     f"(PM2.5 about {w['pm_best']:.0f} µg/m³), worst around {w['peak_hour']:%H:%M} "
                     f"({w['pm_peak']:.0f} µg/m³). Source data is 3-hourly, so treat times as approximate.")

st.markdown("**Activity planner** (adjusted for your profile)")
st.markdown("| Activity | Status | How |\n|---|---|---|\n" +
            "\n".join(f"| {a} | {s} | {t} |" for a, s, t in activity_plan(personal)))

colA, colB = st.columns(2)
with colA:
    st.markdown("**Protect yourself**")
    for t in protection(personal):
        st.write("•", t)
with colB:
    st.markdown(f"**Health watch: {profile}**")
    for t in health_watch(profile):
        st.write("•", t)

with st.expander("City / school advisory (for authorities and institutions)"):
    for t in authority_actions(f["pred"]):
        st.write("•", t)
    st.caption("Stages are modelled on Delhi-NCR's Graded Response Action Plan; check current official thresholds.")

# ---------- AI advisor ----------
st.subheader("Ask the AI advisor")
st.caption("Ask anything about air and your plans. The agent decides which tools to call: forecast, risk, "
           "activity plan, best time window, place comparison.")
question = st.text_input("Your question", placeholder="e.g. Can I walk with my asthmatic father tomorrow? "
                                                      "Is Gurugram better than Delhi?")
if st.button("Ask"):
    qn = question.strip() or "Give me a plan for today and tomorrow."
    try:
        with st.spinner("Agent is choosing and calling tools..."):
            text, steps = cached_agent(city, profile, date_arg, qn)
        st.caption("Agent steps (LLM-driven): " + " -> ".join(steps or ["no tools needed"]))
    except Exception:
        text, steps = offline_agent(city, profile, date_arg)
        st.caption("LLM unavailable, so the built-in agent ran the standard tool chain "
                   "(it cannot answer free-text questions): " + " -> ".join(steps))
    st.markdown(pretty(text))

# ---------- Validation ----------
with st.expander("How accurate is this forecast? (model validation)"):
    st.caption("We tested the model on 2019-2020 data it never saw during training. "
               "Lower error is better, and 'coverage' shows how often the real value fell inside the predicted range.")
    by_city, by_season, overall = hist_validation()
    st.write("Overall:", overall)
    st.markdown("**By season**")
    st.dataframe(by_season.rename_axis("Season"))
    st.markdown("**By city** (coverage should be near 0.80)")
    st.dataframe(by_city)

st.caption("Guidance is general and not medical advice. Risk multipliers are a transparent heuristic, not a "
           "clinical model. Air quality data: Copernicus Atmosphere Monitoring Service (CAMS) via Open-Meteo; "
           "place search: GeoNames via Open-Meteo.")
