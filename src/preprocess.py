"""
Preprocess the raw KOI CSV into X/y splits and save a fitted preprocessor.
Run:
	python -m src.preprocess --csv data/raw/kepler_koi.csv
"""

from __future__ import annotations
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from .utils import (
	RAW_DIR, PROCESSED_DIR, PREPROCESSOR_PATH,
	FEATURES_INITIAL, LABEL_COL, LABEL_TO_INT
)


def load_raw(csv_path: str | None) -> pd.DataFrame:
	if csv_path is None:
		csv_path = RAW_DIR / "kepler_koi.csv"
	df = pd.read_csv(csv_path)
	return df


def build_preprocessor():
	# For numeric-only features, a simple median imputer is enough for trees
	return Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
	])


def main(csv: str | None):
	df = load_raw(csv)

	# Filter to rows that have our columns + label
	cols_needed = FEATURES_INITIAL + [LABEL_COL]
	df = df[[c for c in cols_needed if c in df.columns]].copy()

	# Drop rows without label
	df = df.dropna(subset=[LABEL_COL])

	# Encode label
	df["label_int"] = df[LABEL_COL].map(LABEL_TO_INT)
	df = df.dropna(subset=["label_int"])  # remove unknown labels if any
	df["label_int"] = df["label_int"].astype(int)

	# X / y
	X = df[FEATURES_INITIAL].copy()
	y = df["label_int"].copy()

	# Train/val split (stratified)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, stratify=y, random_state=42
	)

	# Fit preprocessor on training only
	preprocessor = build_preprocessor()
	preprocessor.fit(X_train)

	# Transform and save processed splits (optional)
	X_train_t = preprocessor.transform(X_train)
	X_test_t = preprocessor.transform(X_test)

	# Persist artifacts
	joblib.dump(preprocessor, PREPROCESSOR_PATH)

	# Save processed arrays as parquet for debugging (optional)
	pd.DataFrame(X_train_t, columns=FEATURES_INITIAL).to_parquet(
		PROCESSED_DIR / "X_train.parquet", index=False
	)
	pd.DataFrame(X_test_t, columns=FEATURES_INITIAL).to_parquet(
		PROCESSED_DIR / "X_test.parquet", index=False
	)
	pd.Series(y_train).to_csv(PROCESSED_DIR / "y_train.csv", index=False)
	pd.Series(y_test).to_csv(PROCESSED_DIR / "y_test.csv", index=False)

	print("Preprocessing complete.")
	print(f"Saved preprocessor  {PREPROCESSOR_PATH}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--csv", type=str, default=None, help="Path to KOI CSV")
	args = parser.parse_args()
	main(args.csv)