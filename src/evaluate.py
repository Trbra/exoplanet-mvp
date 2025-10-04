from __future__ import annotations
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from .utils import (
    PROCESSED_DIR, MODEL_PATH, INT_TO_LABEL, ALL_FEATURES, plot_feature_importances
)

CM_PATH = PROCESSED_DIR / "confusion_matrix.png"

def main():
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()
    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro-F1: {f1m:.3f}")

    names = [INT_TO_LABEL[i] for i in sorted(INT_TO_LABEL)]
    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, target_names=names))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm, interpolation='nearest'); ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right'); ax.set_yticklabels(names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    fig.tight_layout(); fig.savefig(CM_PATH, dpi=150); plt.close(fig)
    print("Saved confusion matrix →", CM_PATH)

    try:
        path = plot_feature_importances(model, feature_names=ALL_FEATURES)
        print("Saved feature importances →", path)
    except Exception as e:
        print(f"(Skipping importances: {e})")

if __name__ == "__main__":
    main()
