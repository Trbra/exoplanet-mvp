import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to cleaned CSV")
    parser.add_argument("--model", default="models/model.joblib", help="Path to trained model")
    parser.add_argument("--scaler", default="models/scaler.joblib", help="Path to scaler")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)
    X = df[["koi_period","koi_duration","koi_depth","koi_prad","koi_snr"]].values
    label_map = {"FALSE POSITIVE":0, "CANDIDATE":1, "CONFIRMED":2}
    y = df["koi_disposition"].map(label_map).values

    # Load scaler and model
    scaler = joblib.load(args.scaler)
    model = joblib.load(args.model)

    # Scale features
    X = scaler.transform(X)

    # Predict
    y_pred = model.predict(X)

    # Metrics
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro-F1: {f1:.3f}")
    print("\nPer-class report:")
    print(classification_report(y, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    labels = ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]
    
    # Create a better confusion matrix plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')
    
    # Add labels
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("data/processed/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved confusion matrix -> data/processed/confusion_matrix.png")

if __name__ == "__main__":
    main()
