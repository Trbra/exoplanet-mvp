import pandas as pd, numpy as np, joblib
from pathlib import Path

FEATURES = ["koi_period","koi_duration","koi_depth","koi_prad","koi_snr"]

# Map numeric class indices to text labels
CLASS_MAP = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}

def load_model_and_scaler():
    """Load model and scaler, with error handling"""
    try:
        clf = joblib.load("models/model.joblib")
        scaler = joblib.load("models/scaler.joblib")
        return clf, scaler
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model files not found. Please train a model first. Error: {e}")

def predict_csv(csv_path: str, out_file: str = "predictions.csv"):
    clf, scaler = load_model_and_scaler()
    
    df = pd.read_csv(csv_path)
    df_features = df[FEATURES].copy()
    df_features = df_features.apply(pd.to_numeric, errors='coerce')
    df_features = df_features.fillna(df_features.mean())
    X_scaled = scaler.transform(df_features.values)
    preds = clf.predict(X_scaled)
    
    # Convert numeric predictions to text labels
    df["predicted_class"] = [CLASS_MAP[p] for p in preds]
    df.to_csv(out_file, index=False)
    print(f"Saved predictions -> {out_file}")

    # If actual labels exist, compute match percentage
    if "koi_disposition" in df.columns:
        matches = (df["predicted_class"] == df["koi_disposition"]).sum()
        total = len(df)
        percent_match = matches / total * 100
        print(f"Predicted matches actual koi_disposition: {percent_match:.2f}%")
    
    return df  # Return the dataframe with predictions

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()
    
    predict_csv(a.csv, a.out)
    predict_csv(a.csv, a.out)
