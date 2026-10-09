"""Practical recommendations built from personal risk level, profile, trend and best time window."""

LEVELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# activity: (modify_at, avoid_at, what to do when it must be modified)
ACTIVITIES = {
    "Outdoor exercise (walk / run)": (2, 3, "keep it light and short, in the cleanest window"),
    "Cycling or walking to work": (3, 4, "avoid main roads, wear a well-fitted N95"),
    "Children's outdoor play / school PE": (2, 3, "shift to indoor activity, limit to 30 min"),
    "Opening windows to ventilate": (3, 4, "open briefly only in the cleanest window"),
    "Outdoor work / long time outside": (3, 4, "N95, frequent breaks indoors, plenty of water"),
}


def activity_plan(level_name):
    i = LEVELS.index(level_name)
    rows = []
    for act, (mod, avoid, tip) in ACTIVITIES.items():
        if i >= avoid:
            rows.append((act, "⛔ Avoid", "do it indoors or postpone"))
        elif i >= mod:
            rows.append((act, "⚠️ Modify", tip))
        else:
            rows.append((act, "✅ Go ahead", "normal routine is fine"))
    return rows


def protection(level_name):
    i = LEVELS.index(level_name)
    tips = []
    if i >= 3:
        tips.append("Use an N95/FFP2 respirator outdoors. It must seal around the nose and cheeks; "
                    "cloth and surgical masks filter fine particles poorly.")
        tips.append("Run a HEPA air purifier in the room where you spend most time, windows closed.")
    if i >= 2:
        tips.append("Avoid indoor sources: incense, smoking, burning waste, dry sweeping. "
                    "Use the kitchen exhaust or a window vent while cooking.")
    if i >= 4:
        tips.append("Keep outdoor trips to essentials, and rest if you feel breathless or your chest tightens.")
    return tips or ["No special protection is needed at this level."]


def health_watch(profile):
    base = {
        "Asthma / respiratory condition": [
            "Keep your reliever inhaler with you and follow the asthma action plan from your doctor.",
            "Watch for wheeze, night cough or increased inhaler use, and see a doctor if they worsen."],
        "Child": ["Watch for cough, wheeze or tiredness during play.",
                  "Prefer indoor play on poor-air days."],
        "Elderly": ["Watch for breathlessness, chest discomfort or dizziness, and seek care promptly.",
                    "Keep regular medicines on schedule and keep a family contact informed."],
        "Pregnant": ["Prefer indoor exercise on poor-air days.",
                     "Discuss any breathing difficulty or persistent cough with your doctor."],
        "Runner / outdoor exerciser": ["Hard exercise increases how much polluted air you breathe in; "
                                       "shift intense sessions indoors on poor days."],
        "Outdoor worker": ["Schedule heavy tasks in the cleanest hours and take breaks in cleaner indoor air."],
    }
    return base.get(profile, ["No profile-specific precautions beyond the general advice."])


def plan_ahead(today_aqi, pred_aqi):
    change = (pred_aqi - today_aqi) / max(today_aqi, 1)
    if change >= 0.15:
        return "📈 Air is likely to get worse tomorrow. Do outdoor errands and exercise today."
    if change <= -0.15:
        return "📉 Air is likely to improve tomorrow. Consider postponing non-urgent outdoor plans to then."
    return "➡️ Little change expected tomorrow, so plan around today's levels."


def authority_actions(pred_aqi):
    """Staged city/school actions, modelled on Delhi-NCR's GRAP (verify current thresholds before citing)."""
    if pred_aqi > 450:
        return ["Stage IV-type: restrict heavy vehicles and construction, consider school closure / online classes."]
    if pred_aqi > 400:
        return ["Stage III-type: pause construction and demolition, stop outdoor school activities."]
    if pred_aqi > 300:
        return ["Stage II-type: enforce dust control, boost public transport, schools cancel outdoor assemblies."]
    if pred_aqi > 200:
        return ["Stage I-type: dust and waste-burning enforcement, public advisory, schools reduce outdoor PE."]
    return ["No city-level action needed at this level."]
