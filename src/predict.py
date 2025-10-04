"""
Batch prediction helper for CSVs matching the training feature schema.
Usage:
	python -m src.predict --csv my_new_objects.csv --out predictions.csv

The input CSV must contain the feature columns listed in utils.FEATURES_INITIAL.
"""

from __future__ import annotations
import argparse
import joblib
import pandas as pd
import numpy as np
from .utils import (
	PREPROCESSOR_PATH, MODEL_PATH, FEATURES_INITIAL, INT_TO_LABEL
)


def predict_csv(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	missing = [c for c in FEATURES_INITIAL if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")
	X = df[FEATURES_INITIAL].copy()
	pre = joblib.load(PREPROCESSOR_PATH)
	model = joblib.load(MODEL_PATH)
	X_t = pre.transform(X)
	proba = model.predict_proba(X_t)
	pred_int = np.argmax(proba, axis=1)
	pred_label = [INT_TO_LABEL[i] for i in pred_int]
	out = df.copy()
	out["predicted_class"] = pred_label
	out["p_false_positive"] = proba[:, 0]
	out["p_candidate"] = proba[:, 1]
	out["p_confirmed"] = proba[:, 2]
	return out


def main(csv: str, out: str | None):
	preds = predict_csv(csv)
	if out:
		preds.to_csv(out, index=False)
		print("Saved predictions →", out)
	else:
		print(preds.head())


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--csv", type=str, required=True)
	parser.add_argument("--out", type=str, default=None)
	args = parser.parse_args()
	main(args.csv, args.out)