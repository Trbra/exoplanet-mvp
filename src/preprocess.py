import pandas as pd
import joblib
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

label_map = {"FALSE POSITIVE":0, "CANDIDATE":1, "CONFIRMED":2}

def preprocess_csv(path="data/processed/kepler_koi_clean.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["koi_period","koi_duration","koi_depth","koi_prad","koi_snr","koi_disposition"])
    df["koi_disposition"] = df["koi_disposition"].map(label_map)
    
    X = df[["koi_period","koi_duration","koi_depth","koi_prad","koi_snr"]].values
    y = df["koi_disposition"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Ensure models directory exists
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    joblib.dump((X_train,X_test,y_train,y_test), "data/processed/preprocessed_arrays.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print("[OK] Preprocessed arrays and scaler saved.")

def main():
    parser = argparse.ArgumentParser(description="Preprocess exoplanet data.")
    parser.add_argument("--csv", required=True, help="Path to cleaned CSV")
    args = parser.parse_args()
    
    preprocess_csv(args.csv)

if __name__=="__main__":
    main()
