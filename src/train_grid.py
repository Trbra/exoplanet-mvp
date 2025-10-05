import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier


def load_data():
    """Load preprocessed training arrays"""
    X = joblib.load("data/processed/X.joblib")
    y = joblib.load("data/processed/y.joblib")
    return X, y


def log_loss_scorer(estimator, X, y):
    """Custom scorer for negative log loss (higher = better)"""
    return -log_loss(y, estimator.predict_proba(X))


def main(n_iter, cv):
    X, y = load_data()

    base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(np.unique(y)),
        n_jobs=-1,
        random_state=42
    )

    # Parameter space to explore
    param_dist = {
        "n_estimators": [800, 1200, 1600, 2000],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1, 2, 3, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0.0, 0.1, 0.2, 0.3],
        "reg_lambda": [0.8, 1.0, 1.2, 1.5],
        "reg_alpha": [0.0, 0.1, 0.2],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=log_loss_scorer,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    print(">>> Running hyperparameter search...")
    search.fit(X, y)

    print(f"\nBest CV logloss: {search.best_score_:.4f}")
    print("Best parameters:", search.best_params_)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(search.best_estimator_, "models/model.joblib")
    print("✅ Saved best model → models/model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=30, help="Number of parameter settings sampled")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()
    main(args.n_iter, args.cv)
