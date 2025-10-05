from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model with best parameters.")
    parser.add_argument("--csv", required=True, help="Path to cleaned CSV")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-class-weights", action="store_true")
    # Best parameters
    parser.add_argument("--n_estimators", type=int, default=1200)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--min_child_weight", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--reg_lambda", type=float, default=1.2)
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    X = df[["koi_period","koi_duration","koi_depth","koi_prad","koi_snr"]].values
    # Map string labels to integers
    label_map = {"FALSE POSITIVE":0, "CANDIDATE":1, "CONFIRMED":2}
    y = df["koi_disposition"].map(label_map).values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Compute sample weights if needed
    sample_weights = None
    if args.use_class_weights:
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # Initialize XGBoost
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        objective="multi:softprob",
        num_class=3,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=args.random_state
    )

    # Fit model
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Save model and scaler
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print("✅ Saved model → models/model.joblib")
    print("✅ Saved scaler → models/scaler.joblib")

if __name__ == "__main__":
    main()
