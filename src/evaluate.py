"""
Evaluate the saved model on the test split and write metrics/plots.
Run:
	python -m src.evaluate
"""

from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
	accuracy_score, f1_score, classification_report, confusion_matrix
)
from .utils import (
	PROCESSED_DIR, PREPROCESSOR_PATH, MODEL_PATH,
	INT_TO_LABEL, FEATURES_INITIAL
)

CM_PATH = PROCESSED_DIR / "confusion_matrix.png"


def main():
	X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
	y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()
	model = joblib.load(MODEL_PATH)
	y_pred = model.predict(X_test)

	acc = accuracy_score(y_test, y_pred)
	f1_macro = f1_score(y_test, y_pred, average="macro")
	print(f"Accuracy: {acc:.3f}")
	print(f"Macro-F1: {f1_macro:.3f}")
	print("\nPer-class report:")
	target_names = [INT_TO_LABEL[i] for i in sorted(INT_TO_LABEL)]
	print(classification_report(y_test, y_pred, target_names=target_names))

	# Confusion matrix
	cm = confusion_matrix(y_test, y_pred)
	fig, ax = plt.subplots(figsize=(5, 4))
	im = ax.imshow(cm, interpolation='nearest')
	ax.set_title('Confusion Matrix')
	ax.set_xlabel('Predicted')
	ax.set_ylabel('True')
	ax.set_xticks(range(len(target_names)))
	ax.set_yticks(range(len(target_names)))
	ax.set_xticklabels(target_names, rotation=45, ha='right')
	ax.set_yticklabels(target_names)

	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, cm[i, j], ha='center', va='center')

	fig.tight_layout()
	fig.savefig(CM_PATH, dpi=150)
	print("Saved confusion matrix →", CM_PATH)


if __name__ == "__main__":
	main()