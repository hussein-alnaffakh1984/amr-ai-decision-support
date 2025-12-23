import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import re
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB

from templates import DATASET_TEMPLATE, BREAKPOINTS_TEMPLATE, DRUG_SAFETY_TEMPLATE

# =========================
# 0) UI
# =========================
st.set_page_config(page_title="AMR Decision Support (Research Demo)", layout="centered")
st.title("AMR Decision Support (Research Demo)")
st.caption("⚠️ أداة دعم قرار (Decision Support) بحثية/تعليمية — ليست قرارًا نهائيًا. يجب التحقق مخبريًا وقرار العلاج مسؤولية الطبيب/المختص.")

# =========================
# 1) Templates download (for users)
# =========================
with st.sidebar:
    st.subheader("قوالب جاهزة (Download)")
    st.download_button("⬇️ Dataset Template (CSV)", DATASET_TEMPLATE.encode("utf-8"),
                       file_name="lab_dataset_template.csv", mime="text/csv")
    st.download_button("⬇️ Breakpoints Template (CSV)", BREAKPOINTS_TEMPLATE.encode("utf-8"),
                       file_name="breakpoints_template.csv", mime="text/csv")
    st.download_button("⬇️ Drug Safety Template (CSV)", DRUG_SAFETY_TEMPLATE.encode("utf-8"),
                       file_name="drug_safety_template.csv", mime="text/csv")

# =========================
# 2) Helpers
# =========================
def norm_str(x):
    return str(x).strip().lower() if x is not None else ""

def parse_measurement(x):
    """Parse heterogeneous MIC formats into numeric."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none", "na"]:
        return np.nan
    s = s.replace(" ", "")
    s = s.replace("≤", "<=").replace("≥", ">=")

    # ratios like 1/0.5 or 0.125/2.4 -> average
    if re.fullmatch(r"\d+(\.\d+)?/\d+(\.\d+)?", s):
        a, b = s.split("/")
        return (float(a) + float(b)) / 2.0

    # ranges like 1-2 -> average
    if re.fullmatch(r"\d+(\.\d+)?-\d+(\.\d+)?", s):
        a, b = s.split("-")
        return (float(a) + float(b)) / 2.0

    # inequalities
    m = re.fullmatch(r"(<=|>=|<|>)(\d+(\.\d+)?)", s)
    if m:
        return float(m.group(2))

    try:
        return float(s)
    except:
        return np.nan

@st.cache_data(show_spinner=True)
def load_data_remote(release: str, limit: int) -> pd.DataFrame:
    base = f"https://ftp.ebi.ac.uk/pub/databases/amr_portal/releases/{release}"
    pheno_url = f"{base}/phenotype.parquet"
    con = duckdb.connect()
    df = con.execute(f"""
    SELECT
      species,
      genus,
      antibiotic_name,
      ast_standard,
      laboratory_typing_method,
      measurement,
      measurement_sign,
      measurement_units,
      isolation_source_category,
      geographical_region,
      collection_year,
      resistance_phenotype
    FROM read_parquet('{pheno_url}')
    WHERE resistance_phenotype IS NOT NULL
    LIMIT {limit}
    """).df()
    return df

def load_data_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def load_breakpoints_csv(uploaded_file):
    if uploaded_file is None:
        return None
    bp = pd.read_csv(uploaded_file)
    bp.columns = [c.strip().lower() for c in bp.columns]
    return bp

def load_drug_safety_csv(uploaded_file):
    if uploaded_file is None:
        return None
    ds = pd.read_csv(uploaded_file)
    ds.columns = [c.strip().lower() for c in ds.columns]
    # normalize key
    if "antibiotic_name" in ds.columns:
        ds["antibiotic_name"] = ds["antibiotic_name"].astype(str).str.strip().str.lower()
    return ds

def build_main_model(cat_cols, num_cols):
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
            ]), num_cols),
        ]
    )
    base = Pipeline(steps=[
        ("preprocess", pre),
        ("model", LogisticRegression(max_iter=2500, solver="saga"))
    ])
    return base

def build_bayes_fallback(cat_cols, num_cols):
    """
    Rare-case fallback:
    - ComplementNB on one-hot features (acts like a probabilistic/Bayesian-ish baseline)
    - Good when data is sparse/rare.
    """
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
            ]), num_cols),
        ]
    )
    nb = Pipeline(steps=[
        ("preprocess", pre),
        ("model", ComplementNB(alpha=1.0))
    ])
    return nb

def recommend_from_proba(proba, classes, threshold):
    conf = float(np.max(proba))
    pred = classes[int(np.argmax(proba))]
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    decision = "ABSTAIN" if conf < threshold else "RECOMMEND"
    return decision, pred, conf, [(k, float(v)) for k, v in ranked[:5]]

def evidence_summary(case_dict, scenario_name, model_name, extra=None):
    keys = ["genus", "species", "antibiotic_name", "collection_year"]
    out = [f"{k}={case_dict.get(k)}" for k in keys if case_dict.get(k) not in [None, "", np.nan]]
    out.append(f"Scenario={scenario_name}")
    out.append(f"Model={model_name}")
    if extra:
        out.extend(extra)
    return out[:8]

# =========================
# 3) Sidebar (controls) — covers point 3 (continuous update) + others
# =========================
with st.sidebar:
    st.header("مصدر البيانات (تحديث شهري/ربع سنوي)")

    data_source = st.radio(
        "اختر مصدر البيانات",
        ["Remote (AMR release)", "Upload CSV (مختبر/نسخة محدثة)"],
        index=0
    )
    release = st.text_input("AMR release (للـ Remote)", value="2025-11")
    limit = st.slider("حجم العينة (Remote rows)", 20000, 300000, 120000, step=20000)
    uploaded_dataset = st.file_uploader("Dataset CSV (اختياري)", type=["csv"])
    st.caption("تحديث البيانات: نزّل القالب → املأه شهريًا/ربع سنويًا → ارفعه هنا → Train/Refresh.")

    st.header("سياسة القرار (Decision Support)")
    conf_thresh = st.slider("حد الثقة لاتخاذ قرار", 0.50, 0.95, 0.60, 0.05)
    topk_alternatives = st.slider("عدد البدائل المقترحة عند الحظر/الخطر", 1, 10, 5, 1)

    st.header("توحد طريقة الاختبار (AST Method)")
    method = st.selectbox("Method", ["Broth dilution", "E-test", "Disk diffusion"], index=2)

    st.header("Breakpoints (CLSI/EUCAST) — نقطة 6/7")
    bp_file = st.file_uploader("Breakpoints CSV (اختياري)", type=["csv"])
    st.caption("اكتب version داخل الـ CSV. هذا هو التحديث المعتمد بدل الربط المباشر.")

    st.header("Drug Safety (كلى/كبد/حمل/حساسية) — نقطة 2")
    drug_safety_file = st.file_uploader("Drug Safety CSV (اختياري)", type=["csv"])

    st.header("بيانات المريض (للتصفية/التحذير) — نقطة 2 + إضافة Age (8/9)")
    age = st.number_input("Age", 0, 120, 30, 1)
    urea = st.number_input("Urea (اختياري)", value=float("nan"))
    creatinine = st.number_input("Creatinine (اختياري)", value=float("nan"))
    liver_status = st.selectbox("Liver function", ["طبيعي", "غير طبيعي"], index=0)
    pregnancy = st.selectbox("Pregnancy", ["لا", "نعم"], index=0)
    allergies = st.multiselect("Allergies", ["Penicillin", "Cephalosporins", "Other"], default=[])

    st.header("نقص البيانات (سيناريوهات) — جزء من أهداف البحث")
    scenario = st.selectbox("Scenario", ["S0_full", "S1_no_MIC", "S2_no_typing", "S3_basic_only"], index=0)

    st.header("Rare cases (Bayesian fallback) — نقطة 4")
    enable_rare_fallback = st.checkbox("Enable Bayesian/NaiveBayes fallback for rare cases", value=True)
    rare_min_count = st.slider("اعتبر النوع/الجنس نادر إذا تكراره أقل من", 5, 200, 30, 5)

    retrain_btn = st.button("Train / Refresh Model")

# =========================
# 4) Load dataset
# =========================
dataset_meta = {}
if data_source.startswith("Upload") and uploaded_dataset is not None:
    df = load_data_uploaded_csv(uploaded_dataset)
    dataset_meta = {"source": "upload_csv", "file": getattr(uploaded_dataset, "name", "uploaded.csv")}
else:
    df = load_data_remote(release=release, limit=limit)
    dataset_meta = {"source": "remote_amr", "release": release, "limit": limit}

dataset_meta["loaded_at"] = datetime.now().isoformat(timespec="seconds")
df = df.copy()

# ensure optional columns exist
for col in ["measurement","measurement_sign","measurement_units","ast_standard","laboratory_typing_method",
            "isolation_source_category","geographical_region"]:
    if col not in df.columns:
        df[col] = np.nan

# normalize + numeric
df["measurement_num"] = df["measurement"].apply(parse_measurement) if "measurement" in df.columns else np.nan
df["collection_year"] = pd.to_numeric(df.get("collection_year", np.nan), errors="coerce")

if "resistance_phenotype" not in df.columns:
    st.error("Dataset must contain resistance_phenotype.")
    st.stop()

# labels
df["y"] = df["resistance_phenotype"].astype(str).str.lower().str.strip()

# required core
for core in ["species","genus","antibiotic_name"]:
    if core not in df.columns:
        st.error(f"Dataset missing required column: {core}")
        st.stop()

# keep top-3 labels for stable demo
top3 = df["y"].value_counts().head(3).index.tolist()
df = df[df["y"].isin(top3)].reset_index(drop=True)

# Scenario masking (simulate missingness)
SCENARIOS = {
    "S0_full": [],
    "S1_no_MIC": ["measurement_num","measurement_sign","measurement_units"],
    "S2_no_typing": ["laboratory_typing_method","ast_standard"],
    "S3_basic_only": ["measurement_num","measurement_sign","measurement_units",
                     "laboratory_typing_method","ast_standard","isolation_source_category"]
}
for c in SCENARIOS.get(scenario, []):
    if c in df.columns:
        df[c] = np.nan

FEATURE_COLS = [
    "species","genus","antibiotic_name","ast_standard","laboratory_typing_method",
    "measurement_num","measurement_sign","measurement_units",
    "isolation_source_category","geographical_region","collection_year"
]
TARGET_COL = "y"

# EDA + export dataset used (supports point 3)
with st.expander("EDA + معلومات المصدر + تصدير البيانات الحالية"):
    st.write("Meta:", dataset_meta)
    st.write("Shape:", df.shape)
    st.write("Label counts:", df["y"].value_counts())
    miss = (df[FEATURE_COLS].isna().mean()*100).round(2).sort_values(ascending=False)
    st.write("Missing %:")
    st.dataframe(miss)

    # allow exporting current training dataset snapshot (for audit/versioning)
    csv_bytes = df[FEATURE_COLS + [TARGET_COL]].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export current dataset snapshot (CSV)", csv_bytes,
                       file_name=f"dataset_snapshot_{dataset_meta['loaded_at'].replace(':','-')}.csv",
                       mime="text/csv")

# =========================
# 5) Train models
# =========================
cat_cols = [
    "species","genus","antibiotic_name","ast_standard","laboratory_typing_method",
    "measurement_sign","measurement_units","isolation_source_category","geographical_region"
]
num_cols = ["measurement_num","collection_year"]

@st.cache_resource(show_spinner=True)
def train_models(df_trainable):
    X = df_trainable[FEATURE_COLS]
    y = df_trainable[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = build_main_model(cat_cols, num_cols)
    base.fit(X_train, y_train)

    calib = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calib.fit(X_train, y_train)

    # fallback NB (rare cases)
    nb = build_bayes_fallback(cat_cols, num_cols)
    nb.fit(X_train, y_train)

    # quick stats
    proba = calib.predict_proba(X_test)
    conf = proba.max(axis=1)
    stats = {
        "avg_conf_test": float(conf.mean()),
        "coverage_at_0p60": float((conf >= 0.60).mean()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return calib, nb, stats

if retrain_btn:
    st.cache_resource.clear()

calib_model, nb_model, train_stats = train_models(df)
st.success(
    f"Models ready | AvgConf(test)≈{train_stats['avg_conf_test']:.3f} | "
    f"Coverage@0.60≈{train_stats['coverage_at_0p60']:.3f} | Scenario={scenario}"
)

# =========================
# 6) Load Breakpoints + Drug Safety
# =========================
bp = load_breakpoints_csv(bp_file)
ds = load_drug_safety_csv(drug_safety_file)

# build dict for drug safety lookup
drug_info = {}
if ds is not None and "antibiotic_name" in ds.columns:
    for _, r in ds.iterrows():
        ab = norm_str(r.get("antibiotic_name"))
        if not ab:
            continue
        drug_info[ab] = {
            "drug_class": norm_str(r.get("drug_class")),
            "renal_risk": norm_str(r.get("renal_risk")),
            "hepatic_risk": norm_str(r.get("hepatic_risk")),
            "pregnancy_risk": norm_str(r.get("pregnancy_risk")),
            "avoid_if_allergy": norm_str(r.get("avoid_if_allergy")),
            "notes": str(r.get("notes")) if "notes" in ds.columns else ""
        }

# =========================
# 7) Safety rules (point 2) + “change treatment” via alternatives
# =========================
def allergy_blocks(ab_name, allergies_list):
    """Hard block only when safety table exists; otherwise we warn."""
    ab = norm_str(ab_name)
    if not ab:
        return False, None
    if ab in drug_info:
        avoid = drug_info[ab].get("avoid_if_allergy", "")
        if avoid == "penicillin" and "Penicillin" in allergies_list:
            return True, "Allergy: Penicillin class"
        if avoid == "cephalosporins" and "Cephalosporins" in allergies_list:
            return True, "Allergy: Cephalosporins class"
    return False, None

def risk_level_from_patient(urea_v, cr_v, liver_stat, preg):
    renal_flag = "normal"
    try:
        if not (isinstance(cr_v, float) and np.isnan(cr_v)) and float(cr_v) >= 1.5:
            renal_flag = "impaired"
    except:
        pass
    hepatic_flag = "impaired" if liver_stat == "غير طبيعي" else "normal"
    preg_flag = "pregnant" if preg == "نعم" else "no"
    return renal_flag, hepatic_flag, preg_flag

def safety_status_for_drug(ab_name, renal_flag, hepatic_flag, preg_flag, allergies_list):
    """
    Returns:
      status: ok / caution / avoid / blocked_allergy
      messages: list of warnings
    """
    ab = norm_str(ab_name)
    msgs = []
    if not ab:
        return "caution", ["اسم المضاد غير معروف."]

    blocked, reason = allergy_blocks(ab, allergies_list)
    if blocked:
        return "blocked_allergy", [reason]

    # If no safety table, only general warnings
    if ab not in drug_info:
        msgs.append("لا يوجد جدول سلامة لهذا المضاد (Drug Safety) — سيتم عرض تحذيرات عامة فقط.")
        return "caution", msgs

    info = drug_info[ab]
    rr = info.get("renal_risk", "")
    hr = info.get("hepatic_risk", "")
    pr = info.get("pregnancy_risk", "")
    notes = info.get("notes", "")

    status = "ok"

    # renal
    if renal_flag == "impaired":
        if rr == "high":
            status = "avoid"
            msgs.append("خطر كلوي عالي حسب جدول السلامة.")
        elif rr == "medium":
            status = "caution" if status == "ok" else status
            msgs.append("حذر: خطر كلوي متوسط حسب جدول السلامة.")

    # hepatic
    if hepatic_flag == "impaired":
        if hr == "high":
            status = "avoid"
            msgs.append("خطر كبدي عالي حسب جدول السلامة.")
        elif hr == "medium":
            status = "caution" if status == "ok" else status
            msgs.append("حذر: خطر كبدي متوسط حسب جدول السلامة.")

    # pregnancy
    if preg_flag == "pregnant":
        if pr == "high":
            status = "avoid"
            msgs.append("غير مفضل/خطر عالي بالحمل حسب جدول السلامة.")
        elif pr == "medium":
            status = "caution" if status == "ok" else status
            msgs.append("حذر بالحمل: خطر متوسط حسب جدول السلامة.")

    if notes and str(notes).strip():
        msgs.append(f"Notes: {notes}")

    return status, msgs

# =========================
# 8) Breakpoint interpretation (point 6/7) for MIC + Disk diffusion zone
# =========================
def interpret_sir(case_method, mic_value, zone_value, bp_df, genus, species, antibiotic):
    """
    Uses uploaded breakpoints CSV:
      method + antibiotic (+ genus/species optional)
    For MIC methods => uses s_bp,i_bp
    For Disk diffusion => uses s_zone,i_zone (mm)
    Returns sir, (guideline, version), debug dict
    """
    if bp_df is None:
        return None, None, {"note": "No breakpoints uploaded"}

    m = norm_str(case_method)
    ab = norm_str(antibiotic)
    g = norm_str(genus)
    sp = norm_str(species)

    q = bp_df.copy()
    # normalize columns
    for c in ["antibiotic_name","method","genus","species"]:
        if c in q.columns:
            q[c] = q[c].astype(str).str.strip().str.lower()

    if "antibiotic_name" in q.columns:
        q = q[q["antibiotic_name"] == ab]
    if "method" in q.columns:
        q = q[q["method"] == m]

    # prefer species/genus match if present
    if "species" in q.columns and q["species"].notna().any():
        q2 = q[q["species"] == sp]
        if len(q2) > 0:
            q = q2
    if "genus" in q.columns and q["genus"].notna().any():
        q2 = q[q["genus"] == g]
        if len(q2) > 0:
            q = q2

    if len(q) == 0:
        return None, None, {"note": "No matching breakpoint row"}

    row = q.iloc[0]
    guideline = row.get("guideline", "Unknown")
    version = row.get("version", "Unknown")

    if m == "disk diffusion":
        # zone interpretation: higher zone => more susceptible
        s_zone = row.get("s_zone", np.nan)
        i_zone = row.get("i_zone", np.nan)
        if zone_value is None or (isinstance(zone_value, float) and np.isnan(zone_value)):
            return None, (guideline, version), {"note": "Zone not provided for disk diffusion"}
        try:
            z = float(zone_value)
            sZ = float(s_zone)
            iZ = float(i_zone)
        except:
            return None, (guideline, version), {"note": "Zone breakpoints not numeric"}

        if z >= sZ:
            return "S", (guideline, version), {"type": "zone", "z": z, "s_zone": sZ, "i_zone": iZ}
        elif z >= iZ:
            return "I", (guideline, version), {"type": "zone", "z": z, "s_zone": sZ, "i_zone": iZ}
        else:
            return "R", (guideline, version), {"type": "zone", "z": z, "s_zone": sZ, "i_zone": iZ}
    else:
        # MIC interpretation: lower MIC => more susceptible
        s_bp = row.get("s_bp", np.nan)
        i_bp = row.get("i_bp", np.nan)
        if mic_value is None or (isinstance(mic_value, float) and np.isnan(mic_value)):
            return None, (guideline, version), {"note": "MIC not provided"}
        try:
            mic = float(mic_value)
            s = float(s_bp)
            i = float(i_bp)
        except:
            return None, (guideline, version), {"note": "MIC breakpoints not numeric"}

        if mic <= s:
            return "S", (guideline, version), {"type": "mic", "mic": mic, "s_bp": s, "i_bp": i}
        elif mic <= i:
            return "I", (guideline, version), {"type": "mic", "mic": mic, "s_bp": s, "i_bp": i}
        else:
            return "R", (guideline, version), {"type": "mic", "mic": mic, "s_bp": s, "i_bp": i}

# =========================
# 9) Single-case input (prediction + explanation + alternatives)
# =========================
st.subheader("إدخال حالة للاختبار")

c1, c2 = st.columns(2)
with c1:
    genus_in = st.selectbox("Genus", sorted(df["genus"].dropna().unique().tolist()))
    species_in = st.selectbox("Species", sorted(df["species"].dropna().unique().tolist())[:1000])
with c2:
    antibiotic_in = st.selectbox("Antibiotic", sorted(df["antibiotic_name"].dropna().unique().tolist()))
    year_in = st.number_input("Collection year", 1900, 2100, 2025)

# measurement input depends on method
mic_in = float("nan")
zone_in = float("nan")
if method in ["Broth dilution", "E-test"]:
    mic_in = st.number_input("MIC (numeric) — للـ Broth/E-test", value=float("nan"))
else:
    zone_in = st.number_input("Zone diameter (mm) — للـ Disk diffusion", value=float("nan"))

# Optional evidence fields (if present)
ast_std_in = st.selectbox("AST standard (اختياري)", sorted(df["ast_standard"].dropna().unique().tolist())[:80] + [None])
typing_in = st.selectbox("Typing method (اختياري)", sorted(df["laboratory_typing_method"].dropna().unique().tolist())[:80] + [None])
geo_in = st.selectbox("Geographical region (اختياري)", sorted(df["geographical_region"].dropna().unique().tolist())[:80] + [None])

case = {
    "species": species_in,
    "genus": genus_in,
    "antibiotic_name": antibiotic_in,
    "ast_standard": ast_std_in,
    "laboratory_typing_method": typing_in,
    "measurement_num": mic_in if not (isinstance(mic_in, float) and np.isnan(mic_in)) else np.nan,
    "measurement_sign": np.nan,
    "measurement_units": np.nan,
    "isolation_source_category": np.nan,
    "geographical_region": geo_in,
    "collection_year": year_in
}
row = pd.DataFrame([case])

# =========================
# 10) Rare-case handling (point 4)
# =========================
genus_count = int((df["genus"].astype(str) == str(genus_in)).sum())
species_count = int((df["species"].astype(str) == str(species_in)).sum())
is_rare = (genus_count < rare_min_count) or (species_count < rare_min_count)

# choose model
model_used = "CalibratedLogReg"
proba = calib_model.predict_proba(row)[0]
decision, pred, conf, top5 = recommend_from_proba(proba, calib_model.classes_, conf_thresh)

if enable_rare_fallback and is_rare:
    # if rare, use NB proba (often more stable) AND keep abstain policy
    model_used = "ComplementNB (rare-case fallback)"
    proba = nb_model.predict_proba(row)[0]
    decision, pred, conf, top5 = recommend_from_proba(proba, nb_model.classes_, conf_thresh)

# =========================
# 11) Breakpoints S/I/R (point 6/7)
# =========================
sir, gl, dbg = interpret_sir(method, case["measurement_num"], zone_in, bp, genus_in, species_in, antibiotic_in)

# =========================
# 12) Safety + “change treatment” via alternatives (point 2)
# =========================
renal_flag, hepatic_flag, preg_flag = risk_level_from_patient(urea, creatinine, liver_status, pregnancy)
safety_status, safety_msgs = safety_status_for_drug(antibiotic_in, renal_flag, hepatic_flag, preg_flag, allergies)

hard_block = (safety_status == "blocked_allergy")
avoid = (safety_status == "avoid")

# =========================
# 13) Output
# =========================
st.markdown("## النتيجة (Decision Support)")

st.write("**Decision:**", decision)
st.write("**Predicted label:**", pred)
st.write("**Confidence:**", round(conf, 3))
st.write("**Model used:**", model_used)

if is_rare:
    st.warning(f"Rare case detected: genus_count={genus_count}, species_count={species_count}. Fallback may be used.")

with st.expander("Top-5 probabilities"):
    st.write(top5)

st.markdown("### Breakpoints interpretation (S/I/R)")
if bp is None:
    st.info("لم يتم رفع Breakpoints CSV — (اختياري).")
else:
    if gl is not None:
        st.write(f"Guideline: {gl[0]} | Version: {gl[1]}")
    if sir is None:
        st.warning(f"لا يمكن استخراج S/I/R: {dbg.get('note', 'unknown')}")
    else:
        st.success(f"S/I/R = {sir}")
        st.caption("⚠️ التفسير يعتمد على ملف Breakpoints الذي رفعته — يجب اعتماد النسخة الرسمية لدى المختبر.")

st.markdown("### Patient safety layer (كلى/كبد/حمل/حساسية)")
if ds is None:
    st.warning("لم يتم رفع Drug Safety CSV. ستظهر تحذيرات عامة فقط ولن يتم فلترة دقيقة.")
else:
    st.success("Drug Safety CSV loaded — سيتم تطبيق فلترة وتحذيرات أدق.")

# general warnings
if liver_status == "غير طبيعي":
    st.warning("تنبيه: وظائف كبد غير طبيعية.")
if pregnancy == "نعم":
    st.warning("تنبيه: حمل.")
try:
    if not (isinstance(creatinine, float) and np.isnan(creatinine)) and float(creatinine) >= 1.5:
        st.warning("تنبيه: Creatinine مرتفع.")
except:
    pass
try:
    if not (isinstance(urea, float) and np.isnan(urea)) and float(urea) >= 50:
        st.warning("تنبيه: Urea مرتفعة.")
except:
    pass

# safety decision for selected antibiotic
if hard_block:
    st.error("❌ الدواء المختار محظور بسبب الحساسية.")
elif avoid:
    st.error("⚠️ الدواء المختار غير مناسب (avoid) حسب سلامة المريض/الجدول.")
elif safety_status == "caution":
    st.warning("⚠️ الدواء المختار (caution) — يحتاج حذر/مراجعة.")
else:
    st.success("✅ الدواء المختار مناسب سلاميًا (حسب البيانات المتاحة).")

if safety_msgs:
    with st.expander("تفاصيل السلامة (Reasons)"):
        for m in safety_msgs:
            st.write("•", m)

# =========================
# 14) Explainability (point 8/9) — include Age
# =========================
st.markdown("### تبرير مختصر (Evidence summary)")
extra = [f"Age={age}"]
if sir is not None:
    extra.append(f"Breakpoints_SIR={sir}")
for line in evidence_summary(case, scenario, model_used, extra=extra):
    st.write("•", line)

# =========================
# 15) Treatment “change” via alternatives (point 2) — safe decision-support
# =========================
st.markdown("## بدائل مقترحة (عند الحظر/الخطر/انخفاض الثقة)")
st.caption("هذه بدائل مرشحة بحثيًا مع مراعاة سلامة المريض، وليست وصفًا علاجيًا نهائيًا.")

def generate_alternatives(df_ref, base_case, k, model_for_pred, conf_threshold):
    """
    Build candidates = antibiotics seen with same genus/species.
    Score each candidate using chosen model.
    Apply safety filter:
      - block allergy
      - avoid high-risk if possible
    """
    genus = base_case["genus"]
    species = base_case["species"]

    cand_abs = (
        df_ref[(df_ref["genus"].astype(str) == str(genus)) &
               (df_ref["species"].astype(str) == str(species))]["antibiotic_name"]
        .dropna().astype(str).unique().tolist()
    )
    # fallback if too few
    if len(cand_abs) < 5:
        cand_abs = df_ref[df_ref["genus"].astype(str) == str(genus)]["antibiotic_name"].dropna().astype(str).unique().tolist()

    cand_abs = sorted(list(set(cand_abs)))

    rows = []
    for ab in cand_abs:
        case2 = dict(base_case)
        case2["antibiotic_name"] = ab
        # keep measurement as missing for fair comparison unless method is MIC and you want per-drug MIC
        r2 = pd.DataFrame([case2])
        p = model_for_pred.predict_proba(r2)[0]
        conf = float(np.max(p))
        pred = model_for_pred.classes_[int(np.argmax(p))]
        rows.append((ab, pred, conf))

    alt = pd.DataFrame(rows, columns=["antibiotic", "pred_label", "confidence"])
    alt = alt.sort_values("confidence", ascending=False).reset_index(drop=True)

    # apply safety
    renal_flag, hepatic_flag, preg_flag = risk_level_from_patient(urea, creatinine, liver_status, pregnancy)

    def safe_tag(ab):
        status, msgs = safety_status_for_drug(ab, renal_flag, hepatic_flag, preg_flag, allergies)
        return status, " | ".join(msgs[:2])

    alt[["safety_status", "safety_note"]] = alt["antibiotic"].apply(
        lambda x: pd.Series(safe_tag(x))
    )

    # prefer ok/caution over avoid/blocked
    priority = {"ok": 0, "caution": 1, "avoid": 2, "blocked_allergy": 3}
    alt["safety_rank"] = alt["safety_status"].map(lambda x: priority.get(x, 9))

    # Keep high-confidence first but avoid blocked; if none, still show best with warnings
    alt = alt.sort_values(["safety_rank", "confidence"], ascending=[True, False]).reset_index(drop=True)

    # add decision threshold
    alt["decision"] = alt["confidence"].apply(lambda c: "RECOMMEND" if c >= conf_threshold else "ABSTAIN")
    return alt.head(k)

need_alts = hard_block or avoid or (decision == "ABSTAIN")

if need_alts:
    # choose same model used
    model_for_alts = nb_model if ("ComplementNB" in model_used) else calib_model
    alts = generate_alternatives(df, case, topk_alternatives, model_for_alts, conf_thresh)
    st.dataframe(alts)
else:
    st.info("لا حاجة لبدائل الآن (الخيار الحالي غير محظور ولا يحمل avoid، والثقة فوق الحد).")
