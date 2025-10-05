import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import io, joblib, numpy as np, pandas as pd, plotly.express as px, streamlit as st
import subprocess

from src.utils import (
    RAW_DIR, PROCESSED_DIR, MODELS_DIR,
    FEATURES_INITIAL, ALL_FEATURES, LABEL_COL, LABEL_TO_INT, INT_TO_LABEL,
    PREPROCESSOR_PATH, MODEL_PATH
)
from src.preprocess import main as preprocess_main
from src.train import main as train_main
from src.evaluate import main as evaluate_main
from src.predict import predict_csv

st.set_page_config(page_title="🔭 Exoplanet Classifier | NASA Kepler Mission", layout="wide", page_icon="🔭")
st.title("🔭 Exoplanet Classifier (Kepler/K2/TESS — MVP)")
st.markdown("*Discover distant worlds using machine learning on NASA's mission data*")

def run_command(cmd_parts, description="Running command"):
    """Execute a command and show progress in Streamlit"""
    with st.spinner(f"{description}..."):
        try:
            result = subprocess.run(
                cmd_parts,
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                shell=True
            )
            if result.returncode == 0:
                st.success(f"✅ {description} completed successfully!")
                if result.stdout.strip():
                    output = result.stdout.strip()
                    
                    # Special handling for evaluation output to highlight accuracy
                    if "Accuracy:" in output and "Macro-F1:" in output:
                        lines = output.split('\n')
                        for line in lines:
                            if "Accuracy:" in line:
                                accuracy = line.split("Accuracy:")[1].strip()
                                st.metric("🎯 Model Accuracy", accuracy)
                            elif "Macro-F1:" in line:
                                f1_score = line.split("Macro-F1:")[1].strip()
                                st.metric("📊 Macro F1-Score", f1_score)
                    
                    # Show full output in expandable section
                    with st.expander("📋 View detailed output", expanded=False):
                        st.code(output)
                return True
            else:
                st.error(f"❌ {description} failed!")
                if result.stderr.strip():
                    st.code(result.stderr.strip(), language="text")
                return False
        except Exception as e:
            st.error(f"❌ Error running command: {e}")
            return False

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

# Space-themed custom CSS
st.markdown("""
<style>
    /* Import space font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Root variables - Space theme colors */
    :root {
        --space-bg: #0a0e1a;
        --space-card: #151b2e;
        --space-purple: #9333ea;
        --space-cyan: #06b6d4;
        --space-pink: #ec4899;
        --space-text: #e2e8f0;
        --space-muted: #94a3b8;
    }
    
    /* Main app background with starfield */
    .stApp {
        background: linear-gradient(135deg, #050811, #0a0e1a, #1a1530);
        background-attachment: fixed;
        font-family: 'Space Grotesk', sans-serif;
        color: var(--space-text);
    }
    
    /* Starfield effect */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20% 30%, white, transparent),
            radial-gradient(2px 2px at 60% 70%, white, transparent),
            radial-gradient(1px 1px at 50% 50%, white, transparent),
            radial-gradient(1px 1px at 80% 10%, white, transparent),
            radial-gradient(2px 2px at 90% 60%, white, transparent),
            radial-gradient(1px 1px at 33% 80%, white, transparent),
            radial-gradient(1px 1px at 15% 15%, white, transparent);
        background-size: 200% 200%;
        opacity: 0.4;
        pointer-events: none;
        z-index: 0;
        animation: twinkle 200s ease-in-out infinite;
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.5; }
    }
    
    /* Title styling */
    h1 {
        font-family: 'Orbitron', sans-serif;
        color: var(--space-cyan);
        text-shadow: 0 0 20px rgba(147, 51, 234, 0.5);
        font-weight: 900;
        letter-spacing: 2px;
    }
    
    h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: var(--space-purple);
        text-shadow: 0 0 10px rgba(147, 51, 234, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e1a, #151b2e);
        border-right: 1px solid rgba(147, 51, 234, 0.3);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--space-text);
    }
    
    /* Card/container styling */
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(21, 27, 46, 0.6);
        border-radius: 12px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(147, 51, 234, 0.2);
        box-shadow: 0 0 30px rgba(147, 51, 234, 0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(21, 27, 46, 0.4);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 1px solid rgba(147, 51, 234, 0.3);
        color: var(--space-muted);
        border-radius: 8px;
        font-family: 'Orbitron', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(147, 51, 234, 0.3), rgba(236, 72, 153, 0.3));
        border-color: var(--space-purple);
        color: var(--space-cyan);
        box-shadow: 0 0 20px rgba(147, 51, 234, 0.4);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #9333ea, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-family: 'Orbitron', sans-serif;
        font-weight: 600;
        box-shadow: 0 0 20px rgba(147, 51, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c3aed, #6d28d9);
        box-shadow: 0 0 30px rgba(147, 51, 234, 0.5);
        transform: translateY(-2px);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stSlider > div > div {
        background: rgba(21, 27, 46, 0.8);
        border: 1px solid rgba(147, 51, 234, 0.3);
        border-radius: 8px;
        color: var(--space-text);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(21, 27, 46, 0.6);
        border-radius: 8px;
        border: 1px solid rgba(147, 51, 234, 0.2);
    }
    
    /* Success/warning/error messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(6, 182, 212, 0.1));
        border-left: 4px solid var(--space-cyan);
        color: var(--space-cyan);
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.2), rgba(236, 72, 153, 0.1));
        border-left: 4px solid var(--space-pink);
        color: var(--space-pink);
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
        border-left: 4px solid #ef4444;
        color: #ef4444;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(21, 27, 46, 0.6);
        border: 2px dashed rgba(147, 51, 234, 0.4);
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        color: var(--space-cyan);
        text-shadow: 0 0 10px rgba(6, 182, 212, 0.5);
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: rgba(10, 14, 26, 0.9);
        border: 1px solid rgba(147, 51, 234, 0.3);
        border-radius: 8px;
    }
    
    /* Plotly charts - dark theme compatibility */
    .js-plotly-plot {
        border-radius: 8px;
        background: rgba(21, 27, 46, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🌌 Navigation")
st.sidebar.markdown("- 🔍 Explore\n- 🚀 Train / Evaluate\n- 🎯 Predict\n- ℹ️ About")
TABS = st.tabs(["🔍 Explore", "🚀 Train / Evaluate", "🎯 Predict", "ℹ️ About"])

with TABS[0]:
    st.header("🔍 Explore the dataset")
    default_csv = (PROCESSED_DIR / "kepler_koi_clean.csv") if (PROCESSED_DIR / "kepler_koi_clean.csv").exists() else (RAW_DIR / "kepler_koi.csv")
    if default_csv.exists():
        st.success(f"✨ Found dataset: {default_csv}")
        try:
            df = robust_read_csv(default_csv)
            sample_n = st.slider("Sample size for preview", 200, min(2000, len(df)), 1000)
            show_cols = [c for c in [LABEL_COL] + FEATURES_INITIAL if c in df.columns]
            st.dataframe(df[show_cols].sample(sample_n, random_state=42))
            if LABEL_COL in df.columns:
                st.subheader("📊 Class distribution")
                cd = df[LABEL_COL].astype(str).str.upper().value_counts().reset_index()
                cd.columns = ["class","count"]
                fig = px.bar(cd, x="class", y="count", 
                           color="class",
                           color_discrete_sequence=["#9333ea", "#06b6d4", "#ec4899"])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0'
                )
                st.plotly_chart(fig, use_container_width=True)
            cols_in_df = [c for c in FEATURES_INITIAL if c in df.columns]
            if len(cols_in_df) >= 2:
                st.subheader("✨ Quick scatter")
                x_col = st.selectbox("X-axis", cols_in_df, index=0)
                y_col = st.selectbox("Y-axis", cols_in_df, index=1)
                samp = df.sample(min(len(df), 5000), random_state=42)
                color = LABEL_COL if LABEL_COL in samp.columns else None
                fig = px.scatter(samp, x=x_col, y=y_col, color=color, opacity=0.6,
                               color_discrete_sequence=["#9333ea", "#06b6d4", "#ec4899"])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0'
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error reading dataset: {e}")
    else:
        st.warning("⚠️ No dataset found. Put a CSV at data/raw/kepler_koi.csv or run the cleaner first.")

with TABS[1]:
    st.header("🚀 Train & Evaluate")
    
    # Data Source Selection
    st.subheader("📁 Step 0: Select Data Source")
    data_source = st.radio(
        "Choose your data source:",
        ["🔭 Use Kepler KOI dataset (default)", "📤 Upload your own CSV"],
        index=0
    )
    
    if data_source == "📤 Upload your own CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV file (should contain exoplanet features)", 
            type=["csv"],
            help="CSV should contain columns like koi_period, koi_duration, koi_depth, koi_prad, koi_snr, and koi_disposition"
        )
        if uploaded_file is not None:
            # Save uploaded file
            upload_path = RAW_DIR / "uploaded_data.csv"
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.success(f"✅ File uploaded successfully! Saved to {upload_path}")
            raw_csv_path_default = str(upload_path)
        else:
            st.info("Please upload a CSV file to proceed")
            raw_csv_path_default = "data/raw/kepler_koi.csv"
    else:
        raw_csv_path_default = "data/raw/kepler_koi.csv"
        if Path(raw_csv_path_default).exists():
            st.success(f"✅ Using default Kepler KOI dataset: {raw_csv_path_default}")
        else:
            st.warning(f"⚠️ Default dataset not found at {raw_csv_path_default}. Please upload your own CSV.")
    
    # Step 1: Data Cleaning
    st.subheader("🧹 Step 1: Data Cleaning")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        raw_csv_path = st.text_input("Raw CSV path", value=raw_csv_path_default)
        clean_csv_path = st.text_input("Output clean CSV path", value="data/processed/kepler_koi_clean.csv")
    
    with col2:
        drop_outliers = st.checkbox("Drop outliers", value=True)
        iqr_k = st.number_input("IQR factor", value=3.0, min_value=1.0, max_value=5.0, step=0.5)
    
    if st.button("🧹 Clean CSV"):
        python_exe = ROOT / "exoplanet-mvp" / "Scripts" / "python.exe"
        cmd = f'"{python_exe}" -m src.clean_csv --in "{raw_csv_path}" --out "{clean_csv_path}"'
        if drop_outliers:
            cmd += f" --drop-outliers --iqr-k {iqr_k}"
        
        run_command(cmd, "Cleaning CSV")
    
    st.markdown("---")
    
    # Use cleaned CSV for subsequent steps
    csv_path = st.text_input("Path to training CSV (cleaned)", value=clean_csv_path)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("🔄 Step 1: Preprocessing")
        if st.button("1️⃣ Load & Preprocess"):
            python_exe = ROOT / "exoplanet-mvp" / "Scripts" / "python.exe"
            cmd = f'"{python_exe}" -m src.preprocess --csv "{csv_path}"'
            run_command(cmd, "Preprocessing data")

    with c2:
        st.subheader("🎛️ Step 2: Model Parameters")
        st.markdown("*Optimized hyperparameters for high accuracy*")
        
        # Enhanced parameters based on your requirements
        n_estimators      = st.slider("n_estimators",      200, 3000, 1200, step=50)
        learning_rate     = st.slider("learning_rate",     0.005, 0.20, 0.01, step=0.005)
        max_depth         = st.slider("max_depth",         3,   10,   5)
        min_child_weight  = st.slider("min_child_weight",  1,   10,   5)
        gamma             = st.slider("gamma",             0.0, 2.0,  0.3,  step=0.1)
        subsample         = st.slider("subsample",         0.5, 1.0,  0.8,  step=0.05)
        colsample_bytree  = st.slider("colsample_bytree",  0.5, 1.0,  0.9,  step=0.05)
        reg_alpha         = st.slider("reg_alpha (L1)",    0.0, 1.0,  0.0,  step=0.1)
        reg_lambda        = st.slider("reg_lambda (L2)",   0.0, 3.0,  1.2,  step=0.1)

        # Show current selection for transparency
        with st.expander("🔧 Current hyperparameters"):
            st.json({
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "gamma": gamma,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
            })

        if st.button("2️⃣ Train Model"):
            python_exe = ROOT / "exoplanet-mvp" / "Scripts" / "python.exe"
            cmd = (f'"{python_exe}" -m src.train --csv "{csv_path}" '
                   f'--n_estimators {n_estimators} --learning_rate {learning_rate} '
                   f'--max_depth {max_depth} --min_child_weight {min_child_weight} '
                   f'--gamma {gamma} --subsample {subsample} '
                   f'--colsample_bytree {colsample_bytree} --reg_alpha {reg_alpha} '
                   f'--reg_lambda {reg_lambda}')
            
            run_command(cmd, "Training model with 5-fold CV")

    with c3:
        st.subheader("📊 Step 3: Evaluation")
        if st.button("3️⃣ Evaluate Model"):
            python_exe = ROOT / "exoplanet-mvp" / "Scripts" / "python.exe"
            cmd = f'"{python_exe}" -m src.evaluate --csv "{csv_path}"'
            run_command(cmd, "Evaluating model performance")
        
        # Show confusion matrix if available
        cm_path = PROCESSED_DIR / "confusion_matrix.png"
        if cm_path.exists(): 
            st.image(str(cm_path), caption="📈 Confusion Matrix")
        else: 
            st.info("ℹ️ No confusion matrix found yet.")

    # Feature importance visualization
    if MODEL_PATH.exists():
        st.subheader("📈 Feature Importances")
        try:
            model = joblib.load(MODEL_PATH)
            imps = getattr(model, "feature_importances_", None)
            if imps is not None:
                feats = ALL_FEATURES[:len(imps)]
                df_imp = pd.DataFrame({"feature": feats, "importance": imps}).sort_values("importance", ascending=False)
                fig = px.bar(df_imp.head(15), x="feature", y="importance",
                           color="importance",
                           color_continuous_scale=["#9333ea", "#06b6d4"],
                           title="Top 15 Most Important Features")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0',
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ Model does not expose feature_importances_.")
        except Exception as e:
            st.info(f"ℹ️ (Could not plot importances: {e})")
    
    # Command pipeline summary
    st.markdown("---")
    st.subheader("🔗 Complete Pipeline Commands")
    st.markdown("*For reference or batch processing:*")
    python_exe = ROOT / "exoplanet-mvp" / "Scripts" / "python.exe"
    pipeline_commands = f"""
```bash
# 1. Clean the data
"{python_exe}" -m src.clean_csv --in data/raw/kepler_koi.csv --out data/processed/kepler_koi_clean.csv --drop-outliers --iqr-k 3.0

# 2. Preprocess features  
"{python_exe}" -m src.preprocess --csv data/processed/kepler_koi_clean.csv

# 3. Train with optimized parameters
"{python_exe}" -m src.train --csv data/processed/kepler_koi_clean.csv --n_estimators 1200 --learning_rate 0.01 --max_depth 5 --min_child_weight 5 --gamma 0.3 --subsample 0.8 --colsample_bytree 0.9 --reg_alpha 0.0 --reg_lambda 1.2

# 4. Evaluate model
"{python_exe}" -m src.evaluate --csv data/processed/kepler_koi_clean.csv

# 5. Generate predictions
"{python_exe}" -m src.predict --csv data/processed/kepler_koi_clean.csv --out predictions.csv
```
"""
    st.markdown(pipeline_commands)

with TABS[2]:
    st.header("🎯 Predict on new data")
    
    # Option A: Predict on datasets
    st.markdown("**Option A: Predict on datasets**")
    predict_option = st.radio(
        "Choose data source for prediction:",
        ["🔭 Use Kepler KOI dataset (test set)", "📤 Upload your own CSV"],
        index=0
    )
    
    if predict_option == "📤 Upload your own CSV":
        st.markdown("*Headers can be Kepler/TESS/K2 — aliases supported*")
        st.code(", ".join(FEATURES_INITIAL), language="text")
        up = st.file_uploader("📤 Upload CSV", type=["csv"])
        if up is not None:
            try:
                tmp_path = PROCESSED_DIR / "_uploaded.csv"
                with open(tmp_path, "wb") as f: f.write(up.getvalue())
                st.write("👁️ Preview:")
                df_up = robust_read_csv(tmp_path); st.dataframe(df_up.head())
                if st.button("🔮 Run prediction on uploaded CSV"):
                    if not (PREPROCESSOR_PATH.exists() and MODEL_PATH.exists()):
                        st.error("❌ Please train a model first (Train / Evaluate tab).")
                    else:
                        preds = predict_csv(str(tmp_path))
                        st.success("✨ Predictions complete."); st.dataframe(preds.head())
                        buf = io.BytesIO(); preds.to_csv(buf, index=False); buf.seek(0)
                        st.download_button("⬇️ Download predictions CSV", data=buf, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        # Use Kepler dataset
        kepler_path = "data/processed/kepler_koi_clean.csv"
        if Path(kepler_path).exists():
            st.success(f"✅ Using Kepler KOI dataset: {kepler_path}")
            if st.button("🔮 Generate predictions on Kepler dataset"):
                if not (PREPROCESSOR_PATH.exists() and MODEL_PATH.exists()):
                    st.error("❌ Please train a model first (Train / Evaluate tab).")
                else:
                    python_exe = ROOT / "exoplanet-mvp" / "Scripts" / "python.exe"
                    cmd = f'"{python_exe}" -m src.predict --csv "{kepler_path}" --out predictions.csv'
                    run_command(cmd, "Generating predictions on Kepler dataset")
        else:
            st.warning(f"⚠️ Kepler dataset not found at {kepler_path}. Please clean the data first or upload your own CSV.")

    st.markdown("---")
    st.markdown("**Option B: Manual entry**")
    cols = st.columns(len(FEATURES_INITIAL))
    manual = {}
    defaults = [10.0, 3.5, 500.0, 1.2, 12.0]
    for i, feat in enumerate(FEATURES_INITIAL):
        manual[feat] = cols[i].number_input(
            feat, 
            value=float(defaults[i]), 
            format="%.4f",
            step=0.0001
        )
    if st.button("🔮 Predict manual entry"):
        if not (PREPROCESSOR_PATH.exists() and MODEL_PATH.exists()):
            st.error("Please train a model first (Train / Evaluate tab).")
        else:
            try:
                # Create dataframe with only the 5 basic features that the model was trained on
                df_m = pd.DataFrame([manual])
                # Extract only the features the scaler expects (5 basic features)
                X_manual = df_m[FEATURES_INITIAL].values
                
                # Load scaler and model
                scaler = joblib.load(PREPROCESSOR_PATH)
                model = joblib.load(MODEL_PATH)
                
                # Scale the input using the same 5 features the model was trained on
                X_scaled = scaler.transform(X_manual)
                
                # Make prediction
                proba = model.predict_proba(X_scaled)[0]
                pred_idx = int(np.argmax(proba))
                
                st.success(f"✨ **Predicted: {INT_TO_LABEL[pred_idx]}**")
                st.write({
                    "🔴 p_false_positive": f"{float(proba[0]):.3f}",
                    "🟡 p_candidate": f"{float(proba[1]):.3f}",
                    "🟢 p_confirmed": f"{float(proba[2]):.3f}",
                })
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.write("**Debug info:**")
                st.write(f"- Input values: {manual}")
                st.write(f"- Expected features: {FEATURES_INITIAL}")
                st.write(f"- Model path exists: {MODEL_PATH.exists()}")
                st.write(f"- Scaler path exists: {PREPROCESSOR_PATH.exists()}")

with TABS[3]:
    st.header("ℹ️ About this MVP")
    st.markdown("""
### 🌟 Goal
Predict `CONFIRMED`, `CANDIDATE`, or `FALSE POSITIVE` for transit signals using
tabular features (period, duration, depth, radius, SNR) + engineered features.

### 🔄 Workflow
**Clean → Preprocess → 5-fold CV train → Evaluate → Predict** (CSV or manual).

### 💡 Tips for accuracy
- Add more columns from your catalog (e.g., `koi_model_snr`, impact parameter, mag)
- Keep units consistent
- Train on combined missions with a `mission` column one-hot encoded

### 🚀 Technologies
- **Machine Learning**: XGBoost with cross-validation
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly for interactive charts
- **Data Source**: NASA Kepler, K2, and TESS missions
""")
