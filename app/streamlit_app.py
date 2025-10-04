import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# local imports (run from project root)
from src.utils import (
    RAW_DIR, PROCESSED_DIR, MODELS_DIR,
    FEATURES_INITIAL, LABEL_COL, LABEL_TO_INT, INT_TO_LABEL,
    PREPROCESSOR_PATH, MODEL_PATH
)
from src.preprocess import main as preprocess_main
from src.train import main as train_main
from src.evaluate import main as evaluate_main
from src.predict import predict_csv

st.set_page_config(page_title="Exoplanet Classifier MVP", layout="wide")
st.title(" Exoplanet Classifier (Kepler/K2/TESS — MVP)")

st.sidebar.markdown("""
**Navigation**
- Explore
- Train / Evaluate
- Predict
- About
""")

TABS = st.tabs(["Explore", "Train / Evaluate", "Predict", "About"])

# --------------------
# Explore Tab
# --------------------
with TABS[0]:
    st.header("Explore the dataset")
    default_csv = PROCESSED_DIR / "kepler_koi_clean.csv"
    if default_csv.exists():
        st.success(f"Found default dataset: {default_csv}")
        df = pd.read_csv(default_csv)
        sample_n = st.slider("Sample size for preview", 200, min(2000, len(df)), 1000)
        show_cols = [c for c in [LABEL_COL] + FEATURES_INITIAL if c in df.columns]
        st.dataframe(df[show_cols].sample(sample_n, random_state=42))
        if LABEL_COL in df.columns:
            st.subheader("Class distribution")
            cd = df[LABEL_COL].value_counts().reset_index()
            cd.columns = ["class", "count"]
            fig = px.bar(cd, x="class", y="count")
            st.plotly_chart(fig, use_container_width=True)
        # Simple scatter
        cols_in_df = [c for c in FEATURES_INITIAL if c in df.columns]
        if len(cols_in_df) >= 2:
            st.subheader("Quick scatter")
            x_col = st.selectbox("X", cols_in_df, index=0)
            y_col = st.selectbox("Y", cols_in_df, index=1)
            samp = df.sample(min(len(df), 5000), random_state=42)
            fig2 = px.scatter(
                samp, x=x_col, y=y_col,
                color=LABEL_COL if LABEL_COL in samp.columns else None,
                opacity=0.6
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No default CSV found. Please place 'kepler_koi.csv' into data/raw/.")

# --------------------
# Train / Evaluate Tab
# --------------------
with TABS[1]:
    st.header("Train & Evaluate")
    csv_path = st.text_input("Path to KOI CSV", value=str(default_csv))
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("1) Load & Preprocess"):
            with st.spinner("Preprocessing..."):
                preprocess_main(csv_path if csv_path else None)
            st.success("Preprocessing complete. Artifacts saved.")
    with c2:
        n_estimators = st.slider("n_estimators", 100, 600, 300, step=50)
        max_depth = st.slider("max_depth", 3, 6, 4)
        learning_rate = st.slider("learning_rate", 0.03, 0.2, 0.08, step=0.01)
        if st.button("2) Train Model"):
            with st.spinner("Training model..."):
                train_main(n_estimators, max_depth, learning_rate)
            st.success("Model trained and saved.")
    with c3:
        if st.button("3) Evaluate"):
            with st.spinner("Evaluating..."):
                evaluate_main()
            cm_path = PROCESSED_DIR / "confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion Matrix")
            else:
                st.info("No confusion matrix found yet.")
        # Show feature importances if available
        if MODEL_PATH.exists():
            st.subheader("Feature importances")
            model = joblib.load(MODEL_PATH)
            if hasattr(model, "feature_importances_"):
                imp = pd.DataFrame({
                    "feature": FEATURES_INITIAL,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=False)
                fig = px.bar(imp, x="feature", y="importance")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Model does not expose feature_importances_.")

# --------------------
# Predict Tab
# --------------------
with TABS[2]:
    st.header("Predict on new data")
    st.markdown("**Option A: Upload a CSV** (columns must match the feature list)")
    st.code(", ".join(FEATURES_INITIAL), language="text")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df_up = pd.read_csv(up)
            st.write("Preview:")
            st.dataframe(df_up.head())
            if st.button("Run prediction on uploaded CSV"):
                if not (PREPROCESSOR_PATH.exists() and MODEL_PATH.exists()):
                    st.error("Please train a model first (Train / Evaluate tab).")
                else:
                    preds = predict_csv(up)
                    st.success("Predictions complete.")
                    st.dataframe(preds.head())
                    # Download
                    towrite = io.BytesIO()
                    preds.to_csv(towrite, index=False)
                    towrite.seek(0)
                    st.download_button(
                        label="Download predictions CSV",
                        data=towrite,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown("---")
    st.markdown("**Option B: Manual entry**")
    cols = st.columns(len(FEATURES_INITIAL))
    manual = {}
    for i, feat in enumerate(FEATURES_INITIAL):
        manual[feat] = cols[i].number_input(feat, value=0.0)
    if st.button("Predict manual entry"):
        if not (PREPROCESSOR_PATH.exists() and MODEL_PATH.exists()):
            st.error("Please train a model first (Train / Evaluate tab).")
        else:
            df_m = pd.DataFrame([manual])
            try:
                # reuse predict helpers
                pre = joblib.load(PREPROCESSOR_PATH)
                model = joblib.load(MODEL_PATH)
                X_t = pre.transform(df_m)
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

# --------------------
# About Tab
# --------------------
with TABS[3]:
    st.header("About this MVP")
    st.markdown(
        """
        **Goal:** Predict `CONFIRMED`, `CANDIDATE`, or `FALSE POSITIVE` for transit signals using tabular features (period, duration, depth, radius, SNR). This is a fast baseline for demos and education.

        **How it works:**
        1) We impute missing numeric values (median) and train an XGBoost classifier.
        2) We evaluate with Accuracy and Macro-F1, and show a confusion matrix.
        3) Users can upload CSVs or enter values to get class probabilities.

        **Next steps (stretch):** add more features (stellar context), probability calibration, and support for TESS/TOI tables.
        """
    )