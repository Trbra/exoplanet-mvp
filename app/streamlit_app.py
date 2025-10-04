import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import io, joblib, numpy as np, pandas as pd, plotly.express as px, streamlit as st

from src.utils import (
    RAW_DIR, PROCESSED_DIR, MODELS_DIR,
    FEATURES_INITIAL, ALL_FEATURES, LABEL_COL, LABEL_TO_INT, INT_TO_LABEL,
    PREPROCESSOR_PATH, MODEL_PATH
)
from src.preprocess import main as preprocess_main
from src.train import main as train_main
from src.evaluate import main as evaluate_main
from src.predict import predict_csv

st.set_page_config(page_title="Exoplanet Classifier MVP", layout="wide")
st.title("🔭 Exoplanet Classifier (Kepler/K2/TESS — MVP)")

def robust_read_csv(path):
    tries = [
        dict(sep=None, engine="python", encoding="utf-8-sig", comment="#"),
        dict(sep="\t", encoding="utf-8-sig", comment="#"),
        dict(sep=";", encoding="utf-8-sig", comment="#"),
    ]
    last_err = None
    for kw in tries:
        try:
            return pd.read_csv(path, **kw)
        except Exception as e:
            last_err = e
    try:
        with open(path, "rb") as f:
            if f.read(200).lstrip().lower().startswith(b"<html"):
                raise RuntimeError("The file looks like an HTML page, not a CSV. Re-download the CSV export.")
    except Exception:
        pass
    raise RuntimeError(f"Could not parse CSV {path}. Last error: {last_err}")

st.sidebar.markdown("**Navigation**\n- Explore\n- Train / Evaluate\n- Predict\n- About")
TABS = st.tabs(["Explore", "Train / Evaluate", "Predict", "About"])

with TABS[0]:
    st.header("Explore the dataset")
    default_csv = (PROCESSED_DIR / "kepler_koi_clean.csv") if (PROCESSED_DIR / "kepler_koi_clean.csv").exists() else (RAW_DIR / "kepler_koi.csv")
    if default_csv.exists():
        st.success(f"Found dataset: {default_csv}")
        try:
            df = robust_read_csv(default_csv)
            sample_n = st.slider("Sample size for preview", 200, min(2000, len(df)), 1000)
            show_cols = [c for c in [LABEL_COL] + FEATURES_INITIAL if c in df.columns]
            st.dataframe(df[show_cols].sample(sample_n, random_state=42))
            if LABEL_COL in df.columns:
                st.subheader("Class distribution")
                cd = df[LABEL_COL].astype(str).str.upper().value_counts().reset_index()
                cd.columns = ["class","count"]
                st.plotly_chart(px.bar(cd, x="class", y="count"), use_container_width=True)
            cols_in_df = [c for c in FEATURES_INITIAL if c in df.columns]
            if len(cols_in_df) >= 2:
                st.subheader("Quick scatter")
                x_col = st.selectbox("X", cols_in_df, index=0)
                y_col = st.selectbox("Y", cols_in_df, index=1)
                samp = df.sample(min(len(df), 5000), random_state=42)
                color = LABEL_COL if LABEL_COL in samp.columns else None
                st.plotly_chart(px.scatter(samp, x=x_col, y=y_col, color=color, opacity=0.6), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading dataset: {e}")
    else:
        st.warning("No dataset found. Put a CSV at data/raw/kepler_koi.csv or run the cleaner first.")

with TABS[1]:
    st.header("Train & Evaluate")
    csv_path = st.text_input("Path to training CSV (raw or cleaned)", value=str(default_csv))
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("1) Load & Preprocess"):
            with st.spinner("Preprocessing..."):
                preprocess_main(csv_path if csv_path else None)
            st.success("Preprocessing complete. Artifacts saved.")
    with c2:
        n_estimators = st.slider("n_estimators", 200, 900, 400, step=50)
        max_depth = st.slider("max_depth", 3, 7, 5)
        learning_rate = st.slider("learning_rate", 0.03, 0.15, 0.08, step=0.01)
        min_child_weight = st.slider("min_child_weight", 1, 10, 3)
        gamma = st.slider("gamma", 0.0, 2.0, 0.0, step=0.1)
        if st.button("2) Train Model"):
            with st.spinner("Training model (5-fold CV)..."):
                train_main(n_estimators, max_depth, learning_rate, min_child_weight, gamma)
            st.success("Model trained and saved.")
    with c3:
        if st.button("3) Evaluate"):
            with st.spinner("Evaluating..."):
                evaluate_main()
            cm_path = PROCESSED_DIR / "confusion_matrix.png"
            if cm_path.exists(): st.image(str(cm_path), caption="Confusion Matrix")
            else: st.info("No confusion matrix found yet.")

    if MODEL_PATH.exists():
        st.subheader("Feature importances")
        try:
            model = joblib.load(MODEL_PATH)
            imps = getattr(model, "feature_importances_", None)
            if imps is not None:
                feats = ALL_FEATURES[:len(imps)]
                df_imp = pd.DataFrame({"feature": feats, "importance": imps}).sort_values("importance", ascending=False)
                st.plotly_chart(px.bar(df_imp, x="feature", y="importance"), use_container_width=True)
            else:
                st.info("Model does not expose feature_importances_.")
        except Exception as e:
            st.info(f"(Could not plot importances: {e})")

with TABS[2]:
    st.header("Predict on new data")
    st.markdown("**Option A: Upload a CSV** (headers can be Kepler/TESS/K2 — aliases supported)")
    st.code(", ".join(FEATURES_INITIAL), language="text")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            tmp_path = PROCESSED_DIR / "_uploaded.csv"
            with open(tmp_path, "wb") as f: f.write(up.getvalue())
            st.write("Preview:")
            df_up = robust_read_csv(tmp_path); st.dataframe(df_up.head())
            if st.button("Run prediction on uploaded CSV"):
                if not (PREPROCESSOR_PATH.exists() and MODEL_PATH.exists()):
                    st.error("Please train a model first (Train / Evaluate tab).")
                else:
                    preds = predict_csv(str(tmp_path))
                    st.success("Predictions complete."); st.dataframe(preds.head())
                    buf = io.BytesIO(); preds.to_csv(buf, index=False); buf.seek(0)
                    st.download_button("Download predictions CSV", data=buf, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("**Option B: Manual entry**")
    cols = st.columns(len(FEATURES_INITIAL))
    manual = {}
    defaults = [10.0, 3.5, 500.0, 1.2, 12.0]
    for i, feat in enumerate(FEATURES_INITIAL):
        manual[feat] = cols[i].number_input(feat, value=float(defaults[i]))
    if st.button("Predict manual entry"):
        if not (PREPROCESSOR_PATH.exists() and MODEL_PATH.exists()):
            st.error("Please train a model first (Train / Evaluate tab).")
        else:
            try:
                df_m = pd.DataFrame([manual])
                df_m["duty_cycle"]  = df_m["koi_duration"] / (24.0 * df_m["koi_period"])
                df_m["depth_sqrt"]  = np.sqrt(df_m["koi_depth"])
                df_m["snr_per_hour"]= df_m["koi_snr"] / max(df_m.loc[0,"koi_duration"], 1e-3)
                df_m["log_period"]  = np.log10(df_m["koi_period"])
                df_m["log_duration"]= np.log10(df_m["koi_duration"])
                df_m["log_depth"]   = np.log10(max(df_m.loc[0,"koi_depth"], 1e-6))
                pre = joblib.load(PREPROCESSOR_PATH)
                model = joblib.load(MODEL_PATH)
                X_t = pre.transform(df_m[ALL_FEATURES])
                proba = model.predict_proba(X_t)[0]
                pred_idx = int(np.argmax(proba))
                st.success(f"Predicted: {INT_TO_LABEL[pred_idx]}")
                st.write({
                    "p_false_positive": float(proba[0]),
                    "p_candidate": float(proba[1]),
                    "p_confirmed": float(proba[2]),
                })
            except Exception as e:
                st.error(f"Error: {e}")

with TABS[3]:
    st.header("About this MVP")
    st.markdown("""
**Goal:** Predict `CONFIRMED`, `CANDIDATE`, or `FALSE POSITIVE` for transit signals using
tabular features (period, duration, depth, radius, SNR) + engineered features.

**Flow:** Clean → Preprocess → 5-fold CV train → Evaluate → Predict (CSV or manual).

**Tips for accuracy:** add more columns from your catalog (e.g., `koi_model_snr`, impact parameter, mag),
keep units consistent, and train on combined missions with a `mission` column one-hot encoded.
""")
