"""
ChemSentinel v2 - Water Quality Analyzer
PySustain-2026 | Track 2: Code to Clean Water & Sanitation (SDG 6)

Run with:  streamlit run chemsentinal.py
Needs:     pip install streamlit plotly pandas numpy scikit-learn
"""

import datetime
import inspect
import io
import math
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide", page_title="ChemSentinel: Water Quality Analyzer", page_icon="💧")


# =====================================================================
# 0. SMALL UTILITIES
# =====================================================================
def _st_version():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:  # noqa: BLE001
        return (1, 0)


def stretch(fn):
    """Version-proof 'fill the container width' kwargs for Streamlit widgets."""
    if _st_version() >= (1, 50) and "width" in inspect.signature(fn).parameters:
        return {"width": "stretch"}
    if "use_container_width" in inspect.signature(fn).parameters:
        return {"use_container_width": True}
    return {}


def show_chart(fig, key=None):
    st.plotly_chart(fig, key=key, **stretch(st.plotly_chart))


def show_table(df, height=None):
    kw = stretch(st.dataframe)
    if height:
        kw["height"] = height
    st.dataframe(df, **kw)


# =====================================================================
# 1. STANDARDS, WEIGHTS AND THE WEIGHTED ARITHMETIC WQI
# =====================================================================
PARAMS = ["ph", "tds", "turbidity", "chlorine", "hardness", "fluoride"]
NAMES = {
    "ph": "pH", "tds": "TDS", "turbidity": "Turbidity",
    "chlorine": "Free chlorine", "hardness": "Total hardness", "fluoride": "Fluoride",
}
UNITS = {
    "ph": "", "tds": "mg/L", "turbidity": "NTU",
    "chlorine": "mg/L", "hardness": "mg/L as CaCO3", "fluoride": "mg/L",
}
RADAR_LABELS = ["pH deviation", "TDS", "Turbidity", "Chlorine", "Hardness", "Fluoride"]

# Sn = the standard value used in the index.
#   Drinking set: BIS IS 10500:2012 *acceptable* limits (pH, TDS, turbidity, hardness, fluoride)
#   and the WHO *health-based* guideline for free chlorine (5 mg/L). BIS only sets a minimum
#   residual (0.2 mg/L) for chlorine, so it cannot be used as an upper limit in the index.
#   pH: ideal 7.0, lower limit 6.5, upper limit 8.5 (BIS / WHO operational range).
DRINKING = {
    "ph":        {"sn": 8.5,   "ideal": 7.0, "low": 6.5},
    "tds":       {"sn": 500.0, "ideal": 0.0},
    "turbidity": {"sn": 1.0,   "ideal": 0.0},
    "chlorine":  {"sn": 5.0,   "ideal": 0.0},
    "hardness":  {"sn": 200.0, "ideal": 0.0},
    "fluoride":  {"sn": 1.0,   "ideal": 0.0},
}
# Illustrative screening limits for cooling / process water (NOT a drinking standard).
# Real plants must follow their equipment and boiler/RO manufacturer specifications.
INDUSTRIAL = {
    "ph":        {"sn": 8.0,   "ideal": 7.0, "low": 6.5},
    "tds":       {"sn": 300.0, "ideal": 0.0},
    "turbidity": {"sn": 2.0,   "ideal": 0.0},
    "chlorine":  {"sn": 1.0,   "ideal": 0.0},
    "hardness":  {"sn": 100.0, "ideal": 0.0},
    "fluoride":  {"sn": 1.5,   "ideal": 0.0},
}

QI_CAP = 500.0  # a single extreme value cannot push Qi beyond 500% of its standard

# Brown / Horton WQI classes (lower = better; 100 = right at the standard)
CATEGORIES = [
    (25, "Excellent", "#2ecc71"),
    (50, "Good", "#9bd34a"),
    (75, "Poor", "#f39c12"),
    (100, "Very Poor", "#e67e22"),
    (float("inf"), "Unfit for drinking", "#e74c3c"),
]


def categorize(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a", "#999999"
    for upper, name, colour in CATEGORIES:
        if x <= upper:
            return name, colour
    return "n/a", "#999999"


def weights(std):
    """Wi = K / Sn,  K = 1 / sum(1/Sn)  ->  weights are inversely proportional to the limit."""
    inv = {p: 1.0 / std[p]["sn"] for p in PARAMS}
    k = 1.0 / sum(inv.values())
    return {p: k * inv[p] for p in PARAMS}


def quality_rating(df, std):
    """Qi = 100 * (Vi - Videal) / (Sn - Videal).  pH is handled on both sides of 7."""
    q = pd.DataFrame(index=df.index)
    for p in PARAMS:
        v = df[p].astype(float)
        s = std[p]
        if p == "ph":
            below = 100.0 * (s["ideal"] - v) / (s["ideal"] - s["low"])
            above = 100.0 * (v - s["ideal"]) / (s["sn"] - s["ideal"])
            q[p] = np.where(v < s["ideal"], below, above)
        else:
            q[p] = 100.0 * (v - s["ideal"]) / (s["sn"] - s["ideal"])
        q[p] = q[p].clip(upper=QI_CAP).where(v.notna())
    return q


def wqi_frame(df, std):
    """WQI = sum(Wi*Qi) / sum(Wi). Missing parameters are skipped and weights re-normalised."""
    w = weights(std)
    q = quality_rating(df, std)
    wmat = pd.DataFrame({p: np.where(q[p].notna(), w[p], 0.0) for p in PARAMS}, index=df.index)
    contrib = q.fillna(0.0) * wmat
    wsum = wmat.sum(axis=1).replace(0, np.nan)
    wqi = contrib.sum(axis=1) / wsum
    return wqi, q, contrib, wmat


def hardness_from(ca, mg):
    return 2.497 * ca + 4.118 * mg


# =====================================================================
# 2. COMPLIANCE TABLES (WHO and BIS are kept strictly separate)
# =====================================================================
def who_table(v):
    rows = [
        ("pH", f"{v['ph']:.2f}", "6.5 - 8.5", "Operational range (no health-based value)", 6.5 <= v["ph"] <= 8.5),
        ("TDS", f"{v['tds']:.0f} mg/L", "<= 600 mg/L", "Taste / palatability (no health-based value)", v["tds"] <= 600),
        ("Turbidity", f"{v['turbidity']:.2f} NTU", "< 1 NTU", "Needed for effective disinfection", v["turbidity"] < 1),
        ("Free chlorine", f"{v['chlorine']:.2f} mg/L", "<= 5 mg/L", "Health-based guideline", v["chlorine"] <= 5),
        ("Fluoride", f"{v['fluoride']:.2f} mg/L", "<= 1.5 mg/L", "Health-based guideline", v["fluoride"] <= 1.5),
    ]
    df = pd.DataFrame(rows, columns=["Parameter", "Your value", "WHO guideline", "Basis", "ok"])
    df["Result"] = np.where(df["ok"], "✅ Meets", "❌ Exceeds")
    return df.drop(columns="ok")


def bis_table(v, source):
    def band(val, acc, perm):
        if val <= acc:
            return "✅ Within acceptable limit"
        if val <= perm:
            return "🟡 Above acceptable, within permissible (only if no alternate source)"
        return "❌ Exceeds permissible limit"

    ph_ok = 6.5 <= v["ph"] <= 8.5
    if source == "Tap Water":
        cl_res = "✅ Residual present" if v["chlorine"] >= 0.2 else "❌ Below minimum residual (0.2 mg/L)"
    else:
        cl_res = "n/a (source is not a chlorinated supply)"
    rows = [
        ("pH", f"{v['ph']:.2f}", "6.5 - 8.5", "No relaxation", "✅ Within limit" if ph_ok else "❌ Out of range"),
        ("TDS", f"{v['tds']:.0f} mg/L", "500", "2000", band(v["tds"], 500, 2000)),
        ("Turbidity", f"{v['turbidity']:.2f} NTU", "1", "5", band(v["turbidity"], 1, 5)),
        ("Total hardness", f"{v['hardness']:.0f} mg/L", "200", "600", band(v["hardness"], 200, 600)),
        ("Fluoride", f"{v['fluoride']:.2f} mg/L", "1.0", "1.5", band(v["fluoride"], 1.0, 1.5)),
        ("Free residual chlorine", f"{v['chlorine']:.2f} mg/L", "min 0.2", "min 1.0 (if extended)", cl_res),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Your value", "BIS acceptable", "BIS permissible", "Result"])


def exceeds_permissible(v):
    """Safety guard: BIS permissible limits / WHO health guideline. Returns list of parameter names."""
    bad = []
    if not 6.5 <= v["ph"] <= 8.5:
        bad.append("pH")
    if v["tds"] > 2000:
        bad.append("TDS")
    if v["turbidity"] > 5:
        bad.append("Turbidity")
    if v["hardness"] > 600:
        bad.append("Total hardness")
    if v["fluoride"] > 1.5:
        bad.append("Fluoride")
    if v["chlorine"] > 5:
        bad.append("Free chlorine")
    return bad


# =====================================================================
# 3. INPUT VALIDATION (blocks impossible data, flags suspicious data)
# =====================================================================
def validate_single(ca, mg, vals, source):
    errors, warns = [], []
    labelled = {**{NAMES[k]: vals[k] for k in ("ph", "tds", "turbidity", "chlorine", "fluoride")},
                "Calcium": ca, "Magnesium": mg}
    for label, val in labelled.items():
        if val is None or (isinstance(val, float) and math.isnan(val)):
            errors.append(f"{label}: value is missing.")
        elif val < 0:
            errors.append(f"{label}: a negative value ({val:g}) is not physically possible.")
    if errors:
        return errors, warns

    if vals["ph"] > 14:
        errors.append("pH cannot exceed 14.")
    dissolved = ca + mg + vals["fluoride"] + vals["chlorine"]
    if vals["tds"] < dissolved:
        errors.append(
            f"TDS ({vals['tds']:g} mg/L) is lower than the sum of dissolved Ca + Mg + F + Cl "
            f"({dissolved:.1f} mg/L). Dissolved ions are part of TDS, so this combination is physically impossible."
        )
    if errors:
        return errors, warns

    if vals["ph"] < 4 or vals["ph"] > 10.5:
        warns.append("pH is extreme for natural or treated water - check the probe calibration.")
    if vals["ph"] > 9.5 and vals["hardness"] > 300:
        warns.append("High hardness at pH > 9.5 is chemically unusual: calcium carbonate normally precipitates out.")
    if vals["turbidity"] > 1000:
        warns.append("Turbidity above 1000 NTU is beyond typical sensor range.")
    if vals["tds"] > 35000:
        warns.append("TDS above 35,000 mg/L is beyond seawater - double-check the units.")
    if vals["chlorine"] > 10:
        warns.append("Free chlorine above 10 mg/L is implausible for a water supply - check the units.")
    if vals["fluoride"] > 20:
        warns.append("Fluoride above 20 mg/L is far beyond reported natural levels - check the units.")
    if source == "Tap Water" and vals["chlorine"] == 0:
        warns.append("A tap supply with zero residual chlorine has no protection against re-contamination.")
    if source == "Tap Water" and vals["tds"] > 2000:
        warns.append("Very high TDS for a tap supply - confirm the sample source.")
    return errors, warns


# =====================================================================
# 4. SCIENTIFICALLY CAREFUL ADVICE
# =====================================================================
def household_advice(v, source):
    """List of (level, message). Advice only claims what each method really does."""
    out = []
    if v["turbidity"] > 1:
        lvl = "error" if v["turbidity"] > 5 else "warning"
        out.append((lvl, "💧 **Turbidity above 1 NTU:** particles can shield microbes from disinfection. "
                         "Let the water settle and filter it through a sediment filter (or clean cloth), and only then "
                         "disinfect. Cloudy water that has not been filtered cannot be disinfected reliably."))
    if v["chlorine"] > 5:
        out.append(("error", "💨 **Free chlorine above the WHO health guideline (5 mg/L):** use an activated-carbon filter. "
                             "Standing uncovered helps only for free chlorine (not for chloramine)."))
    if source == "Tap Water" and v["chlorine"] < 0.2:
        out.append(("warning", "🦠 **No/low residual chlorine in a tap supply:** the supply may be unprotected from "
                               "re-contamination. Boiling kills microbes (it does not remove chemicals); report the "
                               "supply to the utility."))
    if v["ph"] < 6.5:
        out.append(("warning", "🧪 **Low pH (acidic):** corrosive - can leach metals from pipes. A calcite / "
                               "neutralising filter raises pH."))
    elif v["ph"] > 8.5:
        out.append(("warning", "🧪 **High pH (alkaline):** reduces chlorine's disinfection power and promotes scaling. "
                               "Household treatment is limited - the supplier usually has to adjust pH (acid dosing)."))
    if v["tds"] > 500:
        out.append(("warning", "🪨 **TDS above 500 mg/L:** reverse osmosis or distillation lowers dissolved solids. "
                               "Boiling does NOT help - it concentrates them. TDS is a total, so test the individual "
                               "ions (nitrate, fluoride, arsenic) before judging safety."))
    if v["hardness"] > 200:
        out.append(("info", "🚿 **Hard water:** mainly a scaling / soap problem rather than a health one. "
                            "An ion-exchange softener removes it; RO also reduces it."))
    if v["fluoride"] > 1.0:
        lvl = "error" if v["fluoride"] > 1.5 else "warning"
        out.append((lvl, "🦷 **Fluoride above the BIS acceptable limit (1.0 mg/L):** long-term intake above "
                         "1.5 mg/L risks dental and skeletal fluorosis. Use RO or activated-alumina defluoridation, or an "
                         "alternate source. Boiling does NOT remove fluoride - it concentrates it."))
    return out


def industrial_directives(v):
    out = []
    if v["turbidity"] > 2:
        out.append("🌫️ **Suspended solids:** coagulation + multimedia/sand filtration upstream of other units.")
    if v["chlorine"] > 1:
        out.append("🧴 **Oxidant carry-over:** dechlorinate (activated carbon or sodium bisulfite) - "
                   "especially before RO membranes.")
    if v["tds"] > 300:
        out.append("⚡ **High dissolved solids:** reverse osmosis and/or demineralisation (ion exchange) as needed.")
    if v["hardness"] > 100:
        out.append("🔥 **Scaling hazard:** softening (ion exchange) or antiscalant dosing to protect heat exchangers and boilers.")
    if v["ph"] < 6.5 or v["ph"] > 8.0:
        out.append("🧪 **pH out of range:** acid/alkali dosing with a pH control loop before use or discharge.")
    if v["fluoride"] > 1.5:
        out.append("🧂 **Fluoride:** RO or activated-alumina removal; check discharge limits before release.")
    return out


# =====================================================================
# 5. PRESETS + SESSION STATE
# =====================================================================
SOURCES = ["Tap Water", "Well Water", "River/Lake", "Industrial Intake"]
PRESETS = {
    "Clean Tap": dict(src="Tap Water", ph=7.2, tds=150.0, turb=0.3, cl=0.5, ca=25.0, mg=6.0, fl=0.2),
    "Borewell with Fluoride": dict(src="Well Water", ph=7.9, tds=1150.0, turb=2.0, cl=0.0, ca=90.0, mg=45.0, fl=3.2),
    "Polluted River": dict(src="River/Lake", ph=6.1, tds=2400.0, turb=38.0, cl=0.0, ca=140.0, mg=70.0, fl=1.8),
}
for _k, _v in PRESETS["Clean Tap"].items():
    st.session_state.setdefault(_k, _v)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# =====================================================================
# 6. SIDEBAR
# =====================================================================
st.title("💧 ChemSentinel: Water Quality Analyzer")
st.caption("PySustain-2026 · Track 2 · SDG 6 Clean Water & Sanitation — weighted-arithmetic WQI, "
           "batch analytics, ML cross-check and live anomaly detection.")

st.sidebar.header("⚡ Quick presets")
for _name in PRESETS:
    st.sidebar.button(_name, on_click=apply_preset, args=(_name,), key=f"preset_{_name}", **stretch(st.button))

st.sidebar.header("Current Water Test")
water_source = st.sidebar.selectbox("Source", SOURCES, key="src")
ph_level = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1, key="ph",
                                   help="Acidity/alkalinity. 7 is neutral. BIS/WHO range: 6.5 - 8.5.")
tds = st.sidebar.number_input("TDS (mg/L)", min_value=0.0, step=10.0, key="tds",
                              help="Total Dissolved Solids. BIS: 500 acceptable / 2000 permissible.")
turbidity = st.sidebar.number_input("Turbidity (NTU)", min_value=0.0, step=0.1, key="turb",
                                    help="Cloudiness. BIS: 1 acceptable / 5 permissible. WHO: <1 NTU for effective disinfection.")
chlorine = st.sidebar.number_input("Free chlorine (mg/L)", min_value=0.0, step=0.1, key="cl",
                                   help="Disinfectant residual. BIS minimum 0.2 mg/L; WHO health guideline 5 mg/L.")
calcium = st.sidebar.number_input("Calcium (mg/L)", min_value=0.0, step=1.0, key="ca",
                                  help="Contributes to hardness and scale.")
magnesium = st.sidebar.number_input("Magnesium (mg/L)", min_value=0.0, step=1.0, key="mg",
                                    help="Contributes to hardness.")
fluoride = st.sidebar.number_input("Fluoride (mg/L)", min_value=0.0, step=0.1, key="fl",
                                   help="Common geogenic contaminant in Indian groundwater. BIS: 1.0 acceptable / 1.5 permissible.")
st.sidebar.write("---")
sentinel_mode = st.sidebar.checkbox("⚠️ Enable Industrial Early Warning Mode", key="industrial")

hardness = round(hardness_from(calcium, magnesium), 2)
sample = dict(ph=ph_level, tds=tds, turbidity=turbidity, chlorine=chlorine, hardness=hardness, fluoride=fluoride)
errors, warns = validate_single(calcium, magnesium, sample, water_source)
STD = INDUSTRIAL if sentinel_mode else DRINKING

# =====================================================================
# 7. MACHINE LEARNING (Random Forest on simulated, limit-labelled data)
# =====================================================================
REF_CLEAN = dict(ph=7.2, tds=150.0, turbidity=0.3, chlorine=0.5, hardness=87.0, fluoride=0.2)


@st.cache_resource(show_spinner="Training Random Forest on simulated samples…")
def train_model(n=6000, seed=42):
    """
    Simulated lab dataset. Label = 'potable' if EVERY parameter is inside the BIS *permissible* limit
    (pH 6.5-8.5, TDS<=2000, turbidity<=5, hardness<=600, fluoride<=1.5, chlorine<=5), with 4% label
    noise to mimic lab/measurement error. Swap in a real labelled dataset for production use.
    """
    rng = np.random.default_rng(seed)
    ph = np.clip(rng.normal(7.4, 0.9, n), 3.5, 11)
    hard = np.clip(rng.lognormal(np.log(220), 0.7, n), 10, 2500)
    tds = np.clip(rng.lognormal(np.log(450), 0.9, n), 20, 8000)
    tds = np.maximum(tds, hard * 0.9 + 20)  # dissolved hardness ions are part of TDS
    turb = np.clip(rng.lognormal(np.log(1.5), 1.1, n), 0.05, 400)
    chlorinated = rng.random(n) > 0.25
    cl = np.where(chlorinated, np.clip(rng.gamma(2.0, 0.35, n), 0, 8), 0.0)
    fl = np.clip(rng.lognormal(np.log(0.7), 0.8, n), 0.02, 12)
    X = pd.DataFrame(dict(ph=ph, tds=tds, turbidity=turb, chlorine=cl, hardness=hard, fluoride=fl))[PARAMS]
    ok = ((ph >= 6.5) & (ph <= 8.5) & (tds <= 2000) & (turb <= 5) & (cl <= 5) & (hard <= 600) & (fl <= 1.5))
    flip = rng.random(n) < 0.04
    y = np.where(flip, ~ok, ok).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    rf = RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=3,
                                class_weight="balanced", random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    return dict(model=rf, acc=accuracy_score(yte, pred), f1=f1_score(yte, pred), n=n,
                importance=pd.Series(rf.feature_importances_, index=PARAMS))


def ml_prob(model, df):
    return model.predict_proba(df[PARAMS])[:, 1]


def shapley_local(model, x):
    """Exact Shapley values (6 features -> 64 coalitions) against a clean-water reference."""
    n = len(PARAMS)
    rows = []
    for m in range(1 << n):
        rows.append({f: (x[f] if (m >> i) & 1 else REF_CLEAN[f]) for i, f in enumerate(PARAMS)})
    p = ml_prob(model, pd.DataFrame(rows))
    fact = [math.factorial(k) for k in range(n + 1)]
    phi = np.zeros(n)
    for i in range(n):
        for m in range(1 << n):
            if (m >> i) & 1:
                continue
            s = bin(m).count("1")
            phi[i] += fact[s] * fact[n - s - 1] / fact[n] * (p[m | (1 << i)] - p[m])
    return float(p[-1]), float(p[0]), phi


# =====================================================================
# 8. BATCH HELPERS
# =====================================================================
ALIASES = {
    "ph": ["ph"], "tds": ["tds", "totaldissolvedsolids", "solids"],
    "turbidity": ["turbidity", "turb", "ntu"],
    "chlorine": ["chlorine", "freechlorine", "residualchlorine", "cl"],
    "calcium": ["calcium", "ca"], "magnesium": ["magnesium", "mg"],
    "hardness": ["hardness", "totalhardness"], "fluoride": ["fluoride", "fl", "f"],
    "lat": ["latitude", "lat"], "lon": ["longitude", "lon", "lng", "long"],
    "date": ["date", "datetime", "timestamp", "time"],
    "id": ["sampleid", "id", "station", "site", "location", "name"],
}


def _norm(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def map_columns(df):
    normed = {_norm(c): c for c in df.columns}
    found = {}
    for key, names in ALIASES.items():
        for n in names:
            if n in normed:
                found[key] = normed[n]
                break
    return found


def prepare_batch(raw):
    cm = map_columns(raw)
    notes = []
    out = pd.DataFrame(index=raw.index)
    for key in ["ph", "tds", "turbidity"]:
        if key not in cm:
            return None, [f"Required column missing: **{key}**"]
        out[key] = pd.to_numeric(raw[cm[key]], errors="coerce")
    for key in ["chlorine", "fluoride"]:
        if key in cm:
            out[key] = pd.to_numeric(raw[cm[key]], errors="coerce")
        else:
            out[key] = np.nan
            notes.append(f"No **{key}** column — it is skipped in the WQI (weights re-normalised) and the ML check.")
    ca = pd.to_numeric(raw[cm["calcium"]], errors="coerce") if "calcium" in cm else None
    mg = pd.to_numeric(raw[cm["magnesium"]], errors="coerce") if "magnesium" in cm else None
    if "hardness" in cm:
        out["hardness"] = pd.to_numeric(raw[cm["hardness"]], errors="coerce")
    elif ca is not None and mg is not None:
        out["hardness"] = hardness_from(ca, mg)
    else:
        return None, ["Need either a **hardness** column or both **calcium** and **magnesium** columns."]
    out["_ions"] = (ca + mg) if (ca is not None and mg is not None) else 0.243 * out["hardness"]
    out["id"] = raw[cm["id"]].astype(str) if "id" in cm else "Sample " + (raw.index + 1).astype(str)
    out["date"] = pd.to_datetime(raw[cm["date"]], errors="coerce") if "date" in cm else pd.NaT
    out["lat"] = pd.to_numeric(raw[cm["lat"]], errors="coerce") if "lat" in cm else np.nan
    out["lon"] = pd.to_numeric(raw[cm["lon"]], errors="coerce") if "lon" in cm else np.nan
    return out, notes


def validate_rows(df):
    reasons = pd.Series("", index=df.index, dtype="object")

    def flag(mask, msg):
        mask = mask.fillna(False)
        reasons.loc[mask] = reasons.loc[mask] + msg + "; "

    flag(df[["ph", "tds", "turbidity", "hardness"]].isna().any(axis=1), "missing required value")
    flag((df[["ph", "tds", "turbidity", "chlorine", "hardness", "fluoride"]] < 0).any(axis=1), "negative value")
    flag(df["ph"] > 14, "pH > 14")
    flag(df["tds"] < df["_ions"].fillna(0) + df["fluoride"].fillna(0) + df["chlorine"].fillna(0),
         "TDS lower than dissolved ions")
    flag(df["turbidity"] > 1000, "turbidity beyond sensor range")
    return reasons.str.rstrip("; ")


def make_demo(n_per=50, seed=11):
    rng = np.random.default_rng(seed)
    profiles = {  # synthetic demo stations: (lat, lon, ph, tds, turb, cl, ca, mg, f)
        "Station A (tap)": (12.935, 79.140, 7.3, 200, 0.5, 0.5, 28, 7, 0.3),
        "Station B (tap)": (12.905, 79.110, 7.1, 260, 0.8, 0.4, 35, 9, 0.4),
        "Station C (borewell)": (12.960, 79.170, 7.8, 900, 1.5, 0.0, 80, 40, 2.0),
        "Station D (borewell)": (12.890, 79.160, 7.6, 650, 1.2, 0.0, 65, 30, 1.2),
        "Station E (river)": (12.925, 79.090, 7.0, 700, 15.0, 0.0, 60, 30, 0.8),
        "Station F (outfall)": (12.950, 79.100, 6.2, 1600, 22.0, 0.0, 120, 55, 1.4),
    }
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=n_per - 1)
    rows = []
    for name, (lat, lon, ph, tds, turb, cl, ca, mg, f) in profiles.items():
        for i in range(n_per):
            trend = 1 + (0.012 * i if "outfall" in name else 0)  # outfall slowly worsens
            c = ca * rng.lognormal(0, 0.1)
            m = mg * rng.lognormal(0, 0.1)
            t = tds * trend * rng.lognormal(0, 0.12)
            fl = f * rng.lognormal(0, 0.15)
            ch = max(cl * rng.lognormal(0, 0.2), 0) if cl > 0 else 0.0
            t = max(t, c + m + fl + ch + 30)
            rows.append(dict(
                Station=name, Date=(start + pd.Timedelta(days=i)).date(),
                Latitude=lat + rng.normal(0, 0.002), Longitude=lon + rng.normal(0, 0.002),
                pH=rng.normal(ph, 0.2), TDS=t, Turbidity=turb * rng.lognormal(0, 0.25) * trend,
                Chlorine=ch, Calcium=c, Magnesium=m, Fluoride=fl))
    df = pd.DataFrame(rows).round(3)
    # a few deliberately bad rows so the validation can be shown live
    bad = df.sample(4, random_state=3).index
    df.loc[bad[0], "pH"] = 15.2
    df.loc[bad[1], "Turbidity"] = -3.0
    df.loc[bad[2], "TDS"] = 40.0
    df.loc[bad[3], "pH"] = np.nan
    return df.sample(frac=1, random_state=5).reset_index(drop=True)


# =====================================================================
# 9. LIVE STREAM HELPERS
# =====================================================================
SP = ["ph", "tds", "turbidity", "chlorine"]
SP_LABEL = {"ph": "pH", "tds": "TDS (mg/L)", "turbidity": "Turbidity (NTU)", "chlorine": "Free chlorine (mg/L)"}
BASE = {"ph": (7.4, 0.04), "tds": (320.0, 5.0), "turbidity": (0.6, 0.05), "chlorine": (0.8, 0.04)}
SENSOR_RANGE = {"ph": (0, 14), "tds": (0, 5000), "turbidity": (0, 1000), "chlorine": (0, 10)}
DRIFT = {"ph": lambda t: 0.02 * math.sin(t / 60), "tds": lambda t: 5 * math.sin(t / 60),
         "turbidity": lambda t: 0.02 * math.sin(t / 40), "chlorine": lambda t: 0.03 * math.sin(t / 45)}
WINDOW = 60


def generate_reading(S):
    rng, t = S["rng"], S["t"]
    r = {p: BASE[p][0] + DRIFT[p](t) + rng.normal(0, BASE[p][1]) for p in SP}
    r["turbidity"], r["chlorine"] = max(r["turbidity"], 0.0), max(r["chlorine"], 0.0)
    truth = "normal"
    if S["event_left"] > 0:
        k = S["event_len"] - S["event_left"]
        shape = math.sin(math.pi * (k + 1) / (S["event_len"] + 1))
        r["turbidity"] += 6.0 * shape
        r["tds"] += 180.0 * shape
        r["chlorine"] = max(r["chlorine"] - 0.55 * shape, 0.0)
        r["ph"] -= 0.5 * shape
        S["event_left"] -= 1
        truth = "contamination"
    if S["fault_left"] > 0:
        p, kind = S["fault"]
        if kind == "Flatline (stuck sensor)":
            r[p] = S["frozen"]
        elif kind == "Erratic noise burst":
            r[p] += rng.normal(0, 12 * BASE[p][1])
        else:
            r[p] = -999.0
        S["fault_left"] -= 1
        truth = f"sensor fault ({p})"
    r["truth"] = truth
    return r


def robust_z(values, x):
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = max(1.4826 * mad, 1e-3 * max(abs(med), 1.0))
    return (x - med) / scale


def run_detection(S, row, method, zthr, cont):
    rows = S["rows"]
    hist = [r for r in rows if not r["flag"]][-WINDOW:]
    faults = {}
    for p in SP:
        lo, hi = SENSOR_RANGE[p]
        if not (lo <= row[p] <= hi):
            faults[p] = "out-of-range reading"
    if len(rows) >= 5:
        for p in SP:
            last = [r[p] for r in rows[-5:]] + [row[p]]
            if p not in faults and np.ptp(last) < 1e-9:
                faults[p] = "flatline"
    z = {}
    if len(hist) >= 20:
        for p in SP:
            if p not in faults:
                z[p] = robust_z(np.array([r[p] for r in hist]), row[p])
    anomalous, implicated = False, []
    if method == "Isolation Forest" and len(hist) >= 40:
        if S["if_model"] is None or S["if_age"] >= 10 or S["if_cont"] != cont:
            Xtr = np.array([[r[p] for p in SP] for r in rows if not r["flag"]][-200:])
            S["if_model"] = IsolationForest(n_estimators=100, contamination=cont, random_state=0).fit(Xtr)
            S["if_age"], S["if_cont"] = 0, cont
        S["if_age"] += 1
        anomalous = S["if_model"].predict(pd.DataFrame([[row[p] for p in SP]], columns=SP).values)[0] == -1
        implicated = [p for p in z if abs(z[p]) > 3.0]
        anomalous = bool(anomalous and implicated)  # IF must be backed by a visible sensor deviation
    else:
        implicated = [p for p in z if abs(z[p]) > zthr]
        anomalous = bool(implicated)
    if faults:
        return True, "Sensor fault", [f"{p} ({why})" for p, why in faults.items()]
    if anomalous:
        params = implicated or ["multivariate pattern"]
        kind = "Contamination event" if len(implicated) >= 2 else "Single-sensor anomaly"
        return True, kind, params
    return False, "", []


def new_stream():
    S = dict(t=0, rows=[], event_left=0, event_len=0, fault=None, fault_left=0, frozen=0.0,
             rng=np.random.default_rng(7), if_model=None, if_age=0, if_cont=None)
    for _ in range(80):  # pre-seed a clean baseline so detection works immediately
        r = generate_reading(S)
        r.update(flag=False, kind="", params=[], t=S["t"])
        S["rows"].append(r)
        S["t"] += 1
    return S


def advance_stream(S, method, zthr, cont):
    row = generate_reading(S)
    flag, kind, params = run_detection(S, row, method, zthr, cont)
    row.update(flag=flag, kind=kind, params=params, t=S["t"])
    S["rows"].append(row)
    S["t"] += 1
    if len(S["rows"]) > 600:
        S["rows"] = S["rows"][-600:]


def cb_inject_event():
    S = st.session_state.stream
    S["event_len"], S["event_left"] = 10, 10


def cb_inject_fault():
    S = st.session_state.stream
    p = st.session_state.fault_param
    S["fault"] = (p, st.session_state.fault_kind)
    S["fault_left"] = 14
    S["frozen"] = S["rows"][-1][p]


def cb_reset_stream():
    st.session_state.stream = new_stream()


if "stream" not in st.session_state:
    st.session_state.stream = new_stream()

# =====================================================================
# 10. TABS
# =====================================================================
tab_single, tab_batch, tab_ml, tab_live, tab_guide = st.tabs(
    ["🧪 Single Sample", "📂 Batch Analysis", "🤖 WQI vs ML", "📡 Live Anomaly Detection", "📖 Methods & Guide"])

# ---------------------------------------------------------------------
# TAB 1 - SINGLE SAMPLE
# ---------------------------------------------------------------------
with tab_single:
    for e in errors:
        st.error("⛔ " + e)
    for w in warns:
        st.warning("⚠️ " + w)

    if errors:
        st.info("Fix the inputs in the sidebar to see the analysis.")
    else:
        df1 = pd.DataFrame([sample])
        wqi_s, q_s, contrib_s, w_s = wqi_frame(df1, STD)
        wqi = float(wqi_s.iloc[0])
        cat, cat_col = categorize(wqi)
        bad_perm = exceeds_permissible(sample)

        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.subheader("Status & Usability")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=wqi,
                number={"valueformat": ".1f", "font": {"size": 44}},
                title={"text": ("Industrial WQI" if sentinel_mode else "Water Quality Index") +
                               "<br><span style='font-size:0.8em'>lower is better</span>"},
                gauge={"axis": {"range": [0, 150]}, "bar": {"color": "#34495e"},
                       "steps": [{"range": [0, 25], "color": "#2ecc71"}, {"range": [25, 50], "color": "#9bd34a"},
                                 {"range": [50, 75], "color": "#f1c40f"}, {"range": [75, 100], "color": "#e67e22"},
                                 {"range": [100, 150], "color": "#e74c3c"}]}))
            gauge.update_layout(height=260, margin=dict(l=20, r=20, t=70, b=10),
                                paper_bgcolor="rgba(0,0,0,0)")
            show_chart(gauge, key="gauge_single")
            st.markdown(f"**Class:** <span style='background:{cat_col};color:white;padding:3px 10px;"
                        f"border-radius:10px'>{cat}</span>", unsafe_allow_html=True)

            if sentinel_mode:
                st.warning("⚙️ INDUSTRIAL SCREENING MODE — illustrative cooling/process-water limits")
                use = {"Excellent": ("success", "🏭 **Process-ready feed:** suitable as make-up feed; electronics-grade (ultra-pure) water still needs RO + deionisation polishing."),
                       "Good": ("success", "🏭 **Utility grade:** suitable for cooling-tower make-up with the usual scale/corrosion inhibitors."),
                       "Poor": ("warning", "🏭 **Washdown grade:** equipment washing, dust suppression; not for boilers or cooling without treatment."),
                       "Very Poor": ("error", "🏭 **Treat before reuse:** exceeds screening limits on several parameters."),
                       "Unfit for drinking": ("error", "🔴 **Do not discharge untreated:** check applicable CPCB / state PCB discharge norms and treat first.")}[cat]
                getattr(st, use[0])(use[1])
                st.subheader("⚠️ Processing Directives")
                dirs = industrial_directives(sample)
                if dirs:
                    for d in dirs:
                        st.info(d)
                else:
                    st.success("✨ Intake parameters are within screening limits. No corrective action required.")
            else:
                if bad_perm:
                    st.error("🔴 **Not suitable for drinking without treatment** — exceeds BIS permissible / WHO "
                             "health limit for: " + ", ".join(bad_perm) +
                             ". May still serve non-drinking uses (flushing, cleaning) depending on the contaminant "
                             "and microbiological quality — verify before use.")
                elif cat in ("Excellent", "Good"):
                    st.success("✅ **Chemically suitable for drinking** (physico-chemical parameters tested). "
                               "Microbiological safety (coliform / E. coli) must be tested separately.")
                else:
                    st.warning("🟡 **Marginal:** within permissible limits but above BIS acceptable limits on some "
                               "parameters. Acceptable only where no better source exists.")
                st.subheader("💡 Treatment Suggestions")
                adv = household_advice(sample, water_source)
                if adv:
                    for lvl, msg in adv:
                        getattr(st, {"error": "error", "warning": "warning", "info": "info"}[lvl])(msg)
                else:
                    st.success("✨ No treatment needed for the parameters tested.")

        with col2:
            st.subheader("Parameter Radar (% of limit)")
            qv = [min(float(q_s[p].iloc[0]), 300.0) for p in PARAMS]
            r_max = max(130.0, max(qv) * 1.1)
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=[100] * 7, theta=RADAR_LABELS + [RADAR_LABELS[0]], fill="toself",
                                          name="Limit (100%)", mode="lines", line=dict(color="#2ecc71", width=2),
                                          fillcolor="rgba(46,204,113,0.1)"))
            fig.add_trace(go.Scatterpolar(r=qv + [qv[0]], theta=RADAR_LABELS + [RADAR_LABELS[0]], fill="toself",
                                          name="Current test", mode="lines+markers",
                                          line=dict(color="#e74c3c", width=3), marker=dict(size=8, color="#e74c3c"),
                                          fillcolor="rgba(231,76,60,0.45)"))
            fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                         radialaxis=dict(visible=True, range=[0, r_max], ticksuffix="%",
                                                         gridcolor="rgba(128,128,128,0.25)"),
                                         angularaxis=dict(gridcolor="rgba(128,128,128,0.25)")),
                              paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
                              margin=dict(l=30, r=30, t=50, b=20), height=430)
            show_chart(fig, key="radar_single")
            st.caption("Each axis = measured value ÷ standard × 100 (pH measured as deviation from 7). Outside the green ring = above the limit. Axes capped at 300%.")

        st.write("---")
        st.subheader("🧮 How the score was built (weighted arithmetic WQI)")
        w = weights(STD)
        total_c = float(contrib_s.sum(axis=1).iloc[0])
        calc = pd.DataFrame({
            "Parameter": [NAMES[p] for p in PARAMS],
            "Your value": [f"{sample[p]:.2f} {UNITS[p]}".strip() for p in PARAMS],
            "Standard (Sn)": [f"{STD[p]['sn']:g} {UNITS[p]}".strip() for p in PARAMS],
            "Quality rating Qi": [round(float(q_s[p].iloc[0]), 1) for p in PARAMS],
            "Weight Wi = K/Sn": [round(w[p], 4) for p in PARAMS],
            "Wi × Qi": [round(float(contrib_s[p].iloc[0]), 2) for p in PARAMS],
            "Share of WQI": [f"{100 * float(contrib_s[p].iloc[0]) / total_c:.0f}%" if total_c > 0 else "0%" for p in PARAMS],
        })
        show_table(calc)
        st.caption("WQI = Σ(Wi·Qi) / ΣWi, with Qi = 100·(Vi − Videal)/(Sn − Videal). Note how weights follow the "
                   "inverse of each limit: tightly regulated parameters (turbidity, fluoride) dominate, while TDS "
                   "and hardness carry little weight — which is why the permissible-limit safety guard above also exists.")

        st.write("---")
        st.subheader("📊 Advanced Analytics")
        a1, a2, a3 = st.columns(3)
        water_type = "Fresh Water" if tds < 1000 else ("Brackish Water" if tds < 10000 else "Saline Water")
        a1.metric("Salinity class (by TDS)", water_type)
        if ph_level < 6.5:
            tendency = "Corrosive tendency"
        elif ph_level > 8.5:
            tendency = "Scale-forming tendency"
        else:
            tendency = "Near-neutral"
        a2.metric("pH-based tendency (indicative)", tendency, help="A full Langelier index also needs alkalinity and temperature.")
        who_df = who_table(sample)
        a3.metric("WHO guideline check", f"{(who_df['Result'] == '✅ Meets').sum()}/{len(who_df)} met")

        c_who, c_bis = st.columns(2)
        with c_who:
            st.markdown("**WHO drinking-water guidelines**")
            show_table(who_df)
        with c_bis:
            st.markdown("**BIS IS 10500:2012 (India)**")
            show_table(bis_table(sample, water_source))
        st.caption("WHO and BIS are different standards and are never mixed: e.g. 500 mg/L TDS is a BIS acceptable limit, "
                   "not a WHO value (WHO has no health-based TDS guideline).")

        st.write("---")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""CHEMSENTINEL WATER QUALITY REPORT
Date/Time: {now}
Source: {water_source}
Mode: {'INDUSTRIAL SCREENING' if sentinel_mode else 'DRINKING WATER (BIS IS 10500 / WHO)'}

PARAMETERS:
- pH: {ph_level}
- TDS: {tds} mg/L
- Turbidity: {turbidity} NTU
- Free chlorine: {chlorine} mg/L
- Calcium / Magnesium: {calcium} / {magnesium} mg/L
- Total hardness (as CaCO3): {hardness} mg/L
- Fluoride: {fluoride} mg/L

WEIGHTED ARITHMETIC WQI: {wqi:.1f}  (lower is better)
CLASS: {cat.upper()}
EXCEEDS PERMISSIBLE / HEALTH LIMIT: {', '.join(bad_perm) if bad_perm else 'none'}

Note: physico-chemical screening only. Microbiological testing is required for drinking-water safety.
"""
        st.download_button("📄 Download Test Report", data=report, file_name="chemsentinel_report.txt", mime="text/plain")

# ---------------------------------------------------------------------
# TAB 2 - BATCH
# ---------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch analysis — upload hundreds of samples")
    st.caption("Accepted columns (names are matched flexibly): pH, TDS, Turbidity, Chlorine, Fluoride, Hardness *or* "
               "Calcium + Magnesium; optional Station/ID, Date, Latitude, Longitude. Uses the same scoring mode as the sidebar.")
    b1, b2, b3 = st.columns([2, 1, 1])
    with b1:
        up = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
    with b2:
        st.write("")
        if st.button("Load demo dataset (300 synthetic samples)", key="load_demo", **stretch(st.button)):
            st.session_state["batch_raw"] = make_demo()
    with b3:
        st.write("")
        tmpl = make_demo(n_per=2).head(5).to_csv(index=False)
        st.download_button("⬇️ CSV template", data=tmpl, file_name="chemsentinel_template.csv", mime="text/csv")

    raw = None
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read that CSV: {exc}")
    elif "batch_raw" in st.session_state:
        raw = st.session_state["batch_raw"]
        st.info("Showing the synthetic demo dataset (locations and values are illustrative, not real measurements). "
                "It contains 4 deliberately invalid rows to demonstrate validation.")

    if raw is None:
        st.info("Upload a CSV or load the demo dataset to begin.")
    else:
        prep, notes = prepare_batch(raw)
        if prep is None:
            for n in notes:
                st.error(n)
        else:
            for n in notes:
                st.warning(n)
            reasons = validate_rows(prep)
            valid = reasons == ""
            res = prep.copy()
            res["Validation"] = np.where(valid, "OK", reasons)
            wq, qb, cb, _ = wqi_frame(prep, STD)
            res["WQI"] = np.where(valid, wq, np.nan)
            res["Class"] = [categorize(x)[0] if v else "Invalid" for x, v in zip(res["WQI"], valid)]
            res["_colour"] = [categorize(x)[1] if v else "#999999" for x, v in zip(res["WQI"], valid)]
            model_pack = train_model()
            complete = valid & prep[PARAMS].notna().all(axis=1)
            res["ML P(potable)"] = np.nan
            if complete.any():
                res.loc[complete, "ML P(potable)"] = ml_prob(model_pack["model"], prep.loc[complete])
            res["WQI verdict"] = np.where(valid, np.where(res["WQI"] <= 50, "Potable", "Not potable"), "—")
            res["ML verdict"] = np.where(complete, np.where(res["ML P(potable)"] >= 0.5, "Potable", "Not potable"), "—")
            res["Agree?"] = np.where(complete, np.where(res["WQI verdict"] == res["ML verdict"], "✅", "⚠️"), "—")

            n_total, n_valid = len(res), int(valid.sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Samples", n_total)
            m2.metric("Valid", n_valid, delta=f"{n_total - n_valid} rejected" if n_total != n_valid else None,
                      delta_color="inverse")
            m3.metric("Mean WQI (valid)", f"{res.loc[valid, 'WQI'].mean():.1f}" if n_valid else "—")
            if complete.any():
                m4.metric("WQI ↔ ML agreement", f"{(res.loc[complete, 'Agree?'] == '✅').mean() * 100:.0f}%")
            else:
                m4.metric("WQI ↔ ML agreement", "—")

            left, right = st.columns([1, 1.6])
            with left:
                order = [c[1] for c in CATEGORIES] + ["Invalid"]
                counts = res["Class"].value_counts().reindex(order).dropna().reset_index()
                counts.columns = ["Class", "Samples"]
                cmap = {c[1]: c[2] for c in CATEGORIES}
                cmap["Invalid"] = "#999999"
                figc = px.bar(counts, x="Class", y="Samples", color="Class", color_discrete_map=cmap)
                figc.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=30, b=10),
                                   title="Samples per WQI class")
                show_chart(figc, key="batch_classes")
            with right:
                choices = ["WQI"] + [p for p in PARAMS]
                sel = st.selectbox("Trend parameter", choices, format_func=lambda x: "WQI (lower = better)" if x == "WQI" else NAMES[x],
                                   key="trend_param")
                roll = st.slider("Smoothing (rolling mean, samples)", 1, 15, 1, key="trend_roll")
                tdf = res[valid].copy()
                if tdf["date"].notna().any():
                    tdf = tdf.sort_values("date")
                    xcol, xlab = "date", "Date"
                else:
                    tdf = tdf.reset_index(drop=True)
                    tdf["order"] = np.arange(len(tdf))
                    xcol, xlab = "order", "Sample order"
                tdf["val"] = tdf.groupby("id")[sel].transform(lambda s: s.rolling(roll, min_periods=1).mean()) \
                    if roll > 1 else tdf[sel]
                figt = px.line(tdf, x=xcol, y="val", color="id", labels={"val": sel, xcol: xlab})
                if sel != "WQI":
                    figt.add_hline(y=DRINKING[sel]["sn"], line_dash="dash", line_color="#e74c3c",
                                   annotation_text="BIS acceptable limit" if sel != "chlorine" else "WHO guideline")
                else:
                    figt.add_hline(y=100, line_dash="dash", line_color="#e74c3c", annotation_text="At the standard (100)")
                figt.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="Station / ID",
                                   title=f"Trend: {'WQI' if sel == 'WQI' else NAMES[sel]}")
                show_chart(figt, key="batch_trend")

            has_geo = prep["lat"].notna().any() and prep["lon"].notna().any()
            if has_geo:
                st.subheader("🗺️ Map (coloured by WQI class)")
                mp = res[valid & prep["lat"].notna() & prep["lon"].notna()].copy()
                mp = mp.rename(columns={"lat": "latitude", "lon": "longitude"})
                mp["colour"] = mp["_colour"]
                try:
                    st.map(mp[["latitude", "longitude", "colour"]], latitude="latitude", longitude="longitude",
                           color="colour", size=40)
                except TypeError:
                    st.map(mp[["latitude", "longitude"]])
                st.caption("🟢 Excellent  🟢 Good  🟡 Poor  🟠 Very Poor  🔴 Unfit")
            else:
                st.caption("Add Latitude / Longitude columns to see the map view.")

            st.subheader("📋 Results table")
            show_cols = ["id", "date", "ph", "tds", "turbidity", "chlorine", "hardness", "fluoride",
                         "WQI", "Class", "ML P(potable)", "WQI verdict", "ML verdict", "Agree?", "Validation"]
            shown = res[show_cols].copy()
            shown["date"] = pd.to_datetime(shown["date"]).dt.date
            shown = shown.round({"WQI": 1, "ML P(potable)": 2, "hardness": 1, "ph": 2, "tds": 0, "turbidity": 2,
                                 "chlorine": 2, "fluoride": 2})
            show_table(shown, height=360)
            st.download_button("⬇️ Download scored results (CSV)", data=shown.to_csv(index=False),
                               file_name="chemsentinel_batch_results.csv", mime="text/csv")

# ---------------------------------------------------------------------
# TAB 3 - ML vs WQI
# ---------------------------------------------------------------------
with tab_ml:
    st.subheader("Rule-based WQI vs Random Forest — do they agree, and why not?")
    pack = train_model()
    model = pack["model"]
    st.caption(f"Random Forest (250 trees) trained on {pack['n']:,} **simulated** samples labelled 'potable' when every "
               f"parameter is inside the BIS *permissible* limit (4% label noise). Hold-out accuracy "
               f"**{pack['acc'] * 100:.1f}%**, F1 **{pack['f1']:.2f}**. The WQI uses the stricter BIS *acceptable* limits — "
               "so the two methods are expected to disagree for water that sits in the relaxation band. "
               "Replace the simulator with a real labelled dataset for production use.")
    if errors:
        st.info("Fix the inputs in the sidebar to compare the two methods.")
    else:
        drink_wqi = float(wqi_frame(pd.DataFrame([sample]), DRINKING)[0].iloc[0])
        wcat, _ = categorize(drink_wqi)
        wqi_pot = drink_wqi <= 50
        p_ml, p_ref, phi = shapley_local(model, sample)
        ml_pot = p_ml >= 0.5
        if sentinel_mode:
            st.caption("This comparison always uses the drinking-water standards, independent of the industrial toggle.")

        c1, c2, c3 = st.columns([1, 1, 1.2])
        with c1:
            st.markdown("#### Rule-based WQI")
            st.metric("WQI", f"{drink_wqi:.1f}", delta=wcat, delta_color="normal" if wqi_pot else "inverse")
            (st.success if wqi_pot else st.error)("Potable (WQI ≤ 50)" if wqi_pot else "Not potable (WQI > 50)")
        with c2:
            st.markdown("#### Random Forest")
            st.metric("P(potable)", f"{p_ml * 100:.0f}%")
            (st.success if ml_pot else st.error)("Potable (P ≥ 50%)" if ml_pot else "Not potable (P < 50%)")
        with c3:
            gp = go.Figure(go.Indicator(mode="gauge+number", value=p_ml * 100, number={"suffix": "%", "valueformat": ".0f"},
                                        title={"text": "ML confidence the water is potable"},
                                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#34495e"},
                                               "steps": [{"range": [0, 50], "color": "#f5b7b1"},
                                                         {"range": [50, 100], "color": "#abebc6"}],
                                               "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
            gp.update_layout(height=230, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
            show_chart(gp, key="gauge_ml")

        # ---- explanation of agreement / disagreement
        bands = []
        for p, acc, perm in [("tds", 500, 2000), ("turbidity", 1, 5), ("hardness", 200, 600), ("fluoride", 1.0, 1.5)]:
            if acc < sample[p] <= perm:
                bands.append(f"{NAMES[p]} ({sample[p]:.2f} {UNITS[p]})")
        if wqi_pot == ml_pot:
            st.success("✅ **The two methods agree.**")
        else:
            st.warning("⚠️ **The methods disagree.**")
            if bands:
                st.markdown("**Likely reason:** " + ", ".join(bands) + " sits between the BIS *acceptable* limit (used by the WQI) "
                            "and the *permissible* limit (what the ML model learned from). The WQI penalises it; the ML model treats it as compliant.")
            elif wqi_pot and not ml_pot:
                st.markdown("**Likely reason:** the WQI averages across parameters, so a clean majority can mask one "
                            "parameter that is over its permissible limit. The Random Forest treats any such breach as a failure.")
            else:
                st.markdown("**Likely reason:** the WQI weights tightly regulated parameters heavily, whereas the model "
                            "also reacts to combinations of borderline values.")

        # ---- explanations
        e1, e2 = st.columns(2)
        with e1:
            imp = pack["importance"].sort_values()
            figi = go.Figure(go.Bar(x=imp.values, y=[NAMES[p] for p in imp.index], orientation="h", marker_color="#3498db"))
            figi.update_layout(title="Global Random Forest feature importance", height=330,
                               margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Mean decrease in impurity")
            show_chart(figi, key="rf_global")
        with e2:
            _, _, contrib_d, _ = wqi_frame(pd.DataFrame([sample]), DRINKING)
            wq_share = contrib_d.iloc[0].values / max(float(contrib_d.iloc[0].sum()), 1e-9) * 100
            push = np.maximum(-phi, 0.0)
            rf_share = push / push.sum() * 100 if push.sum() > 1e-9 else np.zeros(len(PARAMS))
            figs = go.Figure()
            figs.add_trace(go.Bar(name="WQI: share of score", y=[NAMES[p] for p in PARAMS], x=wq_share,
                                  orientation="h", marker_color="#e67e22"))
            figs.add_trace(go.Bar(name="Random Forest: Shapley push towards 'not potable'", y=[NAMES[p] for p in PARAMS],
                                  x=rf_share, orientation="h", marker_color="#3498db"))
            figs.update_layout(barmode="group", title="This sample: what drives each verdict (%)", height=330,
                               margin=dict(l=10, r=10, t=50, b=10), xaxis_title="% of total",
                               legend=dict(orientation="h", y=-0.25))
            show_chart(figs, key="rf_local")
        st.caption("Local bars use exact Shapley values (all 64 feature coalitions) of the Random Forest probability, measured against a "
                   "clean-water reference. Where the orange and blue bars differ most is where the two methods weigh the same water differently.")

# ---------------------------------------------------------------------
# TAB 4 - LIVE ANOMALY DETECTION
# ---------------------------------------------------------------------
with tab_live:
    st.subheader("Live sensor stream with anomaly detection")
    st.caption("A simulated 4-sensor monitoring station. **Contamination** moves several correlated sensors at once "
               "(turbidity ↑, TDS ↑, chlorine ↓, pH ↓); a **sensor fault** hits one sensor only (stuck, noisy or error code). "
               "The detector separates the two.")

    k1, k2, k3, k4 = st.columns([1, 1, 1.2, 1.2])
    with k1:
        st.toggle("▶ Live stream", value=True, key="stream_run")
        st.slider("Refresh every (s)", 0.5, 5.0, 1.0, 0.5, key="stream_every")
    with k2:
        st.radio("Detector", ["Robust z-score", "Isolation Forest"], key="det_method")
        if st.session_state.det_method == "Robust z-score":
            st.slider("z threshold", 2.5, 8.0, 4.0, 0.5, key="det_z")
        else:
            st.slider("Expected anomaly rate", 0.002, 0.05, 0.005, 0.001, key="det_cont", format="%.3f")
    with k3:
        st.button("🧪 Inject contamination spike", on_click=cb_inject_event, key="btn_event", **stretch(st.button))
        st.selectbox("Sensor", SP, format_func=lambda p: SP_LABEL[p], key="fault_param")
    with k4:
        st.selectbox("Fault type", ["Flatline (stuck sensor)", "Erratic noise burst", "Dropout (-999 error code)"],
                     key="fault_kind")
        st.button("🔧 Inject sensor fault", on_click=cb_inject_fault, key="btn_fault", **stretch(st.button))
        st.button("♻️ Reset stream", on_click=cb_reset_stream, key="btn_reset", **stretch(st.button))
    st.selectbox("Chart parameter", SP, format_func=lambda p: SP_LABEL[p], key="chart_param")

    def stream_panel():
        S = st.session_state.stream
        method = st.session_state.det_method
        if st.session_state.stream_run:
            advance_stream(S, method, st.session_state.get("det_z", 4.0), st.session_state.get("det_cont", 0.005))
        rows = S["rows"]
        recent = rows[-120:]
        p = st.session_state.chart_param
        last = rows[-1]
        recent_alerts = [r for r in rows[-12:] if r["flag"]]

        if recent_alerts:
            r = recent_alerts[-1]
            fn = st.error if r["kind"] == "Contamination event" else st.warning
            fn(f"🚨 **{r['kind']}** at t={r['t']} — {', '.join(r['params'])}")
        else:
            st.success("✅ All sensors normal")

        # FIX: 6 columns (Readings, Alerts, + 4 sensors). The old code used 5 columns, so the
        # first sensor (pH) overwrote the "Alerts (total)" metric.
        m = st.columns(6)
        m[0].metric("Readings", S["t"])
        m[1].metric("Alerts (total)", sum(r["flag"] for r in rows))
        for i, sp in enumerate(SP):
            m[i + 2].metric(SP_LABEL[sp], f"{last[sp]:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[r["t"] for r in recent], y=[r[p] for r in recent], mode="lines",
                                 name=SP_LABEL[p], line=dict(color="#3498db", width=2)))
        styles = {"Contamination event": ("#e74c3c", "x"), "Sensor fault": ("#8e44ad", "diamond"),
                  "Single-sensor anomaly": ("#f39c12", "circle-open")}
        for kind, (col, sym) in styles.items():
            pts = [r for r in recent if r["flag"] and r["kind"] == kind and SENSOR_RANGE[p][0] <= r[p] <= SENSOR_RANGE[p][1]]
            if pts:
                fig.add_trace(go.Scatter(x=[r["t"] for r in pts], y=[r[p] for r in pts], mode="markers", name=kind,
                                         marker=dict(color=col, size=11, symbol=sym, line=dict(width=2))))
        n_drop = sum(1 for r in recent if r["flag"] and not (SENSOR_RANGE[p][0] <= r[p] <= SENSOR_RANGE[p][1]))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Reading #",
                          yaxis_title=SP_LABEL[p], legend=dict(orientation="h", y=1.12),
                          title=f"{SP_LABEL[p]} — last {len(recent)} readings")
        show_chart(fig, key="stream_chart")
        if n_drop:
            st.caption(f"{n_drop} out-of-range error reading(s) are flagged but not drawn (they would flatten the chart).")

        alerts = [r for r in rows if r["flag"]][-8:][::-1]
        if alerts:
            st.markdown("**Alert log (latest first)**")
            show_table(pd.DataFrame([{"Reading #": r["t"], "Detected as": r["kind"], "Sensors": ", ".join(r["params"]),
                                      "Injected truth": r["truth"]} for r in alerts]))

    if hasattr(st, "fragment"):
        every = float(st.session_state.stream_every) if st.session_state.stream_run else None
        st.fragment(run_every=every)(stream_panel)()
    else:
        st.info("Your Streamlit version has no auto-refresh fragments; press the button for each new reading "
                "(upgrade Streamlit ≥ 1.37 for live mode).")
        if st.button("Next reading", key="manual_tick"):
            advance_stream(st.session_state.stream, st.session_state.det_method,
                           st.session_state.get("det_z", 4.0), st.session_state.get("det_cont", 0.005))
        st.session_state.stream_run = False
        stream_panel()

# ---------------------------------------------------------------------
# TAB 5 - METHODS & GUIDE
# ---------------------------------------------------------------------
with tab_guide:
    st.subheader("Methods")
    st.markdown("""
**Weighted arithmetic Water Quality Index (Brown et al. / Horton formulation)**

* Quality rating: `Qi = 100 × (Vi − Videal) / (Sn − Videal)` (Videal = 0, except pH = 7.0)
* Unit weight: `Wi = K / Sn`, where `K = 1 / Σ(1/Sn)` — weights are **inversely proportional to the standard**
* Qi is capped at 500 so that one extreme reading cannot dominate the whole index
* Index: `WQI = Σ(Wi × Qi) / ΣWi`  →  0–25 Excellent · 26–50 Good · 51–75 Poor · 76–100 Very poor · >100 Unfit

A WQI of 100 means the water sits exactly at the standard for the weighted average, so **lower is better**.
""")
    std_df = pd.DataFrame({
        "Parameter": [NAMES[p] for p in PARAMS],
        "Sn used (drinking)": ["8.5 (range 6.5–8.5)", "500 mg/L", "1 NTU", "5 mg/L", "200 mg/L", "1.0 mg/L"],
        "Source of Sn": ["BIS IS 10500:2012 / WHO operational", "BIS acceptable", "BIS acceptable (WHO: <1 NTU)",
                         "WHO health-based guideline (BIS gives only a 0.2 mg/L minimum)", "BIS acceptable", "BIS acceptable (WHO health guideline 1.5)"],
        "Weight Wi": [round(weights(DRINKING)[p], 4) for p in PARAMS],
    })
    show_table(std_df)
    st.markdown("""
**Limitations — worth stating in the presentation**
* Chemical/physical screening only; **no microbiological** parameters (coliform, E. coli) and no arsenic, nitrate, iron or heavy metals yet.
* Inverse-limit weighting gives loosely regulated parameters (TDS, hardness) very little influence, so the app also applies a permissible-limit safety guard.
* The Random Forest is trained on **simulated** labelled data; the comparison shows *method behaviour*, not field accuracy.
* The industrial limits are illustrative screening values, not a substitute for equipment or CPCB/SPCB specifications.
""")
    with st.expander("📖 What do these water parameters actually mean?"):
        st.markdown("""
* **pH:** how acidic or basic the water is. Low pH is corrosive and can leach metals from pipes; high pH promotes scale and weakens chlorine disinfection.
* **TDS:** the total of dissolved minerals, salts and metals. It tells you *how much* is dissolved, not *what* — a low TDS can still hide a toxic ion.
* **Turbidity:** cloudiness from suspended particles. Pathogens can hide on particles and survive disinfection, which is why WHO asks for < 1 NTU.
* **Free chlorine:** disinfectant residual that protects water in the pipe network. Too little means no protection; very high levels are a health and taste issue.
* **Hardness (Ca + Mg):** mostly a scaling and soap-use issue rather than a direct health risk. Computed here as `2.497·Ca + 4.118·Mg` (mg/L as CaCO3).
* **Fluoride:** beneficial in small amounts, but long-term intake above 1.5 mg/L causes dental and skeletal fluorosis; it is a common geogenic contaminant in Indian groundwater.
""")
