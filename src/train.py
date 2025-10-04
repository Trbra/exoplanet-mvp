from __future__ import annotations
import argparse, joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from .utils import PROCESSED_DIR, MODEL_PATH, Ensemble

def compute_class_weights(y: np.ndarray) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    return {cls: total / (len(classes) * cnt) for cls, cnt in zip(classes, counts)}

def main(n_estimators: int, max_depth: int, learning_rate: float, min_child_weight: int, gamma: float):
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models, losses = [], []

    for fold,(tr,va) in enumerate(skf.split(X_train, y_train),1):
        m = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9, colsample_bytree=0.9,
            min_child_weight=min_child_weight, gamma=gamma,
            random_state=42+fold, n_jobs=0,
        )
        cw = compute_class_weights(y_train[tr])
        sw = np.array([cw[y] for y in y_train[tr]])
        m.fit(X_train.iloc[tr], y_train[tr],
              sample_weight=sw,
              eval_set=[(X_train.iloc[va], y_train[va])],
              verbose=False)
        proba = m.predict_proba(X_train.iloc[va])
        losses.append(log_loss(y_train[va], proba))
        models.append(m)

    print(f"CV mlogloss: mean={np.mean(losses):.4f} ± {np.std(losses):.4f}")
    ens = Ensemble(models)
    joblib.dump(ens, MODEL_PATH)
    print("Saved model →", MODEL_PATH)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--learning_rate", type=float, default=0.08)
    ap.add_argument("--min_child_weight", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.0)
    a = ap.parse_args()
    main(a.n_estimators, a.max_depth, a.learning_rate, a.min_child_weight, a.gamma)
