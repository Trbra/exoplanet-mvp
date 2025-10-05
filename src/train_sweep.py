from pathlib import Path
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def load_data(csv_path: str, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    FEATURES = ["koi_period","koi_duration","koi_depth","koi_prad","koi_snr"]
    X = df[FEATURES].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())
    y = df["koi_disposition"].astype(str).str.strip().str.upper()
    label_map = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
    y = y.map(label_map)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data(args.csv, args.test_size, args.random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Parameter grid around your current best
    param_grid = {
        "n_estimators": [1100, 1200, 1300],
        "learning_rate": [0.008, 0.01, 0.012],
        "max_depth": [5, 6],
        "min_child_weight": [4, 5, 6],
        "gamma": [0.2, 0.3, 0.4],
        "subsample": [0.75, 0.8, 0.85],
        "colsample_bytree": [0.85, 0.9, 0.95],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [1.0, 1.2]
    }

    best_acc = 0
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        clf = XGBClassifier(
            **params,
            objective="multi:softmax",
            num_class=3,
            random_state=args.random_state,
            eval_metric="mlogloss"
        )
        scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring="accuracy")
        mean_acc = scores.mean()
        print(f"Tested params: {params} -> CV Accuracy: {mean_acc:.4f}")

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_model = clf
            best_params = params

    print(f"Best CV accuracy: {best_acc:.4f} with params: {best_params}")
    best_model.fit(X_train_scaled, y_train)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, Path(args.output_dir) / "model.joblib")
    joblib.dump(scaler, Path(args.output_dir) / "scaler.joblib")
    print(f"Saved model and scaler to {args.output_dir}")

    # Final evaluation on test set
    y_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
