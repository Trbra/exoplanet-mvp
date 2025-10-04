"""
Train an XGBoost multi-class model and save it to models/.
Run:
	python -m src.train --n_estimators 300 --max_depth 4 --learning_rate 0.08
"""

from __future__ import annotations
import argparse
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from .utils import (
	PROCESSED_DIR, PREPROCESSOR_PATH, MODEL_PATH,
	FEATURES_INITIAL
)


def compute_class_weights(y: np.ndarray) -> dict:
	# Inverse frequency weighting
	classes, counts = np.unique(y, return_counts=True)
	total = counts.sum()
	weights = {cls: total / (len(classes) * cnt) for cls, cnt in zip(classes, counts)}
	return weights


def main(n_estimators: int, max_depth: int, learning_rate: float):
	# Load processed data
	X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
	X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
	y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel()
	y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()
	preprocessor = joblib.load(PREPROCESSOR_PATH)

	# Build model
	model = XGBClassifier(
		objective="multi:softprob",
		eval_metric="mlogloss",
		n_estimators=n_estimators,
		max_depth=max_depth,
		learning_rate=learning_rate,
		subsample=0.9,
		colsample_bytree=0.9,
		random_state=42,
		n_jobs=0,
	)

	# Compute sample weights for imbalance
	cw = compute_class_weights(y_train)
	sample_weight = np.array([cw[y] for y in y_train])
	model.fit(X_train, y_train, sample_weight=sample_weight)
	joblib.dump(model, MODEL_PATH)
	print("Model trained and saved →", MODEL_PATH)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--n_estimators", type=int, default=300)
	parser.add_argument("--max_depth", type=int, default=4)
	parser.add_argument("--learning_rate", type=float, default=0.08)
	args = parser.parse_args()
	main(args.n_estimators, args.max_depth, args.learning_rate)
