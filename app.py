import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import re

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

st.set_page_config(page_title="AMR Decision Support (Research Demo)", layout="centered")

st.title("AMR Decision Support (Research Demo)")
st.caption("AI-assisted early decision under incomplete microbiological evidence (research use).")

# -----------------------------
# Utilities
# -----------------------------
def parse_measurement(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none", "na"]:
        return np.nan
    s = s.replace(" ", "")
    s = s.replace("≤", "<=").replace("≥", ">=")

    if re.fullmatch(r"\d+(\.\d+)?/\d+(\.\d+)?", s):
        a, b = s.split("/")
        return (float(a) + float(b)) / 2.0

    if re.fullmatch(r"\d+(\.\d+)?-\d+(\.\d+)?", s):
        a, b = s.split("-")
        return (float(a) + float(b)) / 2.0

    m = re.fullmatch(r"(<=|>=|<|>)(\d+(\.\d+)?)", s)
    if m:
        return float(m.group(2))

    try:
        return float(s)
    except:
        return np.nan


@st.cache_data(show_spinner=True)
def load_data(source: str, release: str = "2025-11", limit: int = 200000) -> pd.DataFrame:
    """
    source:
      - "remote_parquet": reads AMR Portal phenotype parquet from FTP
      - "local_csv": expects a CSV URL (raw GitHub / direct link)
    """
    if source == "remote_parquet":
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

    # fallback: user-provided CSV URL
    df = pd.read_csv(source)
    return df


def build_model(cat_cols, num_cols):
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
        ]
    )

    return Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=800))
    ])


def recommend(proba, classes, threshold=0.60):
    conf = float(np.max(proba))
    pred = classes[int(np.argmax(proba))]
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    if conf < threshold:
        return {"decision": "ABSTAIN", "pred": pred, "confidence": conf,
                "top": ranked[:3]}
    return {"decision": "RECOMMEND", "pred": pred, "confidence": conf,
            "top": ranked[:3]}


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Data")
    mode = st.selectbox("Data source", ["remote_parquet"], index=0)
    release = st.text_input("AMR release (if remote)", value="2025-11")
    limit = st.slider("Rows (sample)", 20000, 300000, 120000, step=20000)

    st.header("Policy")
    conf_thresh = st.slider("Confidence threshold", 0.50, 0.90, 0.60, 0.05)

    st.header("Scenario (simulate missingness)")
    scenario = st.selectbox("Scenario", ["S0_full", "S1_no_MIC", "S2_no_typing", "S3_basic_only"], index=0)

    train_btn = st.button("Train / Refresh Model")


# -----------------------------
# Load + prepare data
# -----------------------------
df = load_data("remote_parquet", release=release, limit=limit)

df = df.copy()
df["measurement_num"] = df["measurement"].apply(parse_measurement)
df["collection_year"] = pd.to_numeric(df["collection_year"], errors="coerce")
df["y"] = df["resistance_phenotype"].astype(str).str.lower().str.strip()

# keep top-3 labels for stable demo
top3 = df["y"].value_counts().head(3).index.tolist()
df = df[df["y"].isin(top3)].reset_index(drop=True)

FEATURE_COLS = [
    "species","genus","antibiotic_name","ast_standard","laboratory_typing_method",
    "measurement_num","measurement_sign","measurement_units",
    "isolation_source_category","geographical_region","collection_year"
]
TARGET_COL = "y"

# scenario missingness (drop evidence by setting NaN)
SCENARIOS = {
    "S0_full": [],
    "S1_no_MIC": ["measurement_num","measurement_sign","measurement_units"],
    "S2_no_typing": ["laboratory_typing_method","ast_standard"],
    "S3_basic_only": ["measurement_num","measurement_sign","measurement_units",
                     "laboratory_typing_method","ast_standard","isolation_source_category"]
}

if scenario in SCENARIOS and len(SCENARIOS[scenario]) > 0:
    for c in SCENARIOS[scenario]:
        if c in df.columns:
            df[c] = np.nan

# show quick EDA
with st.expander("Quick data check (EDA)"):
    st.write("Shape:", df.shape)
    st.write("Top labels:", df["y"].value_counts())
    miss = (df[FEATURE_COLS].isna().mean() * 100).round(2).sort_values(ascending=False)
    st.write("Missing % per feature:")
    st.dataframe(miss)

# -----------------------------
# Train model (cached in session)
# -----------------------------
cat_cols = [
    "species","genus","antibiotic_name","ast_standard","laboratory_typing_method",
    "measurement_sign","measurement_units","isolation_source_category","geographical_region"
]
num_cols = ["measurement_num","collection_year"]

@st.cache_resource(show_spinner=True)
def train_calibrated_model(df_trainable, conf_thresh):
    X = df_trainable[FEATURE_COLS]
    y = df_trainable[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = build_model(cat_cols, num_cols)
    base.fit(X_train, y_train)

    calib = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calib.fit(X_train, y_train)

    # simple report
    proba = calib.predict_proba(X_test)
    conf = proba.max(axis=1)
    coverage = float((conf >= conf_thresh).mean())
    avg_conf = float(conf.mean())

    return calib, coverage, avg_conf, calib.classes_

if train_btn:
    st.cache_resource.clear()

calib_model, coverage, avg_conf, classes_ = train_calibrated_model(df, conf_thresh)

st.success(f"Model ready. Coverage@{conf_thresh:.2f} = {coverage:.3f} | Avg confidence = {avg_conf:.3f}")

# -----------------------------
# User input -> prediction
# -----------------------------
st.subheader("Make a prediction (single case)")

col1, col2 = st.columns(2)
with col1:
    species = st.selectbox("species", sorted(df["species"].dropna().unique().tolist())[:500])
    genus = st.selectbox("genus", sorted(df["genus"].dropna().unique().tolist()))
    antibiotic = st.selectbox("antibiotic_name", sorted(df["antibiotic_name"].dropna().unique().tolist()))
with col2:
    geo = st.selectbox("geographical_region", sorted(df["geographical_region"].dropna().unique().tolist()))
    year = st.number_input("collection_year", min_value=1900, max_value=2100, value=2020)
    mic = st.number_input("measurement_num (MIC, if known)", value=float(np.nan), format="%.3f")

ast_std = st.selectbox("ast_standard (optional)", sorted(df["ast_standard"].dropna().unique().tolist())[:50] + [None])
typing = st.selectbox("laboratory_typing_method (optional)", sorted(df["laboratory_typing_method"].dropna().unique().tolist())[:50] + [None])

row = pd.DataFrame([{
    "species": species,
    "genus": genus,
    "antibiotic_name": antibiotic,
    "ast_standard": ast_std,
    "laboratory_typing_method": typing,
    "measurement_num": mic if not (isinstance(mic, float) and np.isnan(mic)) else np.nan,
    "measurement_sign": np.nan,
    "measurement_units": np.nan,
    "isolation_source_category": np.nan,
    "geographical_region": geo,
    "collection_year": year
}])

proba = calib_model.predict_proba(row)[0]
out = recommend(proba, classes_, threshold=conf_thresh)

st.write("Decision:", out["decision"])
st.write("Predicted label:", out["pred"])
st.write("Confidence:", round(out["confidence"], 3))
st.write("Top ranks:", [(k, float(v)) for k, v in out["top"]])
