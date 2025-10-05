# 🔭 Exoplanet Classifier (Kepler/K2/TESS — MVP)

Predict the likelihood that a transit signal is a **CONFIRMED planet**, **CANDIDATE**, or **FALSE POSITIVE** using tabular features from NASA's Kepler, K2, and TESS missions.

---

## 🌟 Project Goal
Automatically classify exoplanet transit signals using machine learning, including:

- Period, duration, depth, radius, SNR
- Engineered features:
  - Duty cycle
  - Depth square root
  - SNR per hour
  - Log-transformed period, duration, and depth

---

## 🔄 Workflow
1. **Clean**: Remove outliers, handle missing values.
2. **Preprocess**: Compute engineered features, standardize inputs.
3. **Train**: 5-fold cross-validation using XGBoost.
4. **Evaluate**: Confusion matrix, metrics, visualizations.
5. **Predict**: CSV batch or manual input.

---

## 🚀 Features

### Data Exploration
- Interactive dataset preview
- Class distribution bar charts
- Quick scatter plots of features

### Train & Evaluate
- Clean, preprocess, train, and evaluate directly in the app
- Adjustable hyperparameters for XGBoost
- Full pipeline button: Clean → Preprocess → Train → Evaluate → Predict
- Confusion matrix visualization

### Prediction
- Predict manually by entering feature values
- Probability distribution for each class

### Visualization
- Dark, space-themed interface
- Interactive Plotly charts
- Styled tables and metrics

---

## 💡 Tips for Accuracy
- Include additional features such as `koi_model_snr`, impact parameter, stellar magnitude
- Keep consistent units
- One-hot encode mission column when combining data from multiple missions
- Increase training dataset for better generalization

---

## 📦 Requirements
- Python 3.10+
- Packages:
  - streamlit
  - pandas
  - numpy
  - plotly
  - scikit-learn
  - xgboost
  - joblib

---

## ⚡ Run the App
```bash
git clone 
cd exoplanet-classifier
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
