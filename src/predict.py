from __future__ import annotations
import argparse, joblib, pandas as pd, numpy as np
from .utils import PREPROCESSOR_PATH, MODEL_PATH, FEATURES_INITIAL, ALL_FEATURES, INT_TO_LABEL, ALIASES

def resolve_feature_aliases(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for canon, opts in ALIASES.items():
        for o in opts:
            if o.lower().strip() in lower:
                rename[lower[o.lower().strip()]] = canon; break
    return df.rename(columns=rename)

def predict_csv(csv_path: str) -> pd.DataFrame:
    # robust read
    tries = [
        dict(sep=None, engine="python", encoding="utf-8-sig", comment="#"),
        dict(sep="\t", encoding="utf-8-sig", comment="#"),
        dict(sep=";", encoding="utf-8-sig", comment="#"),
    ]
    last=None
    for kw in tries:
        try:
            df = pd.read_csv(csv_path, **kw); break
        except Exception as e: last=e
    else:
        raise RuntimeError(f"Could not parse {csv_path}. Last error: {last}")

    df = resolve_feature_aliases(df)

    missing = [c for c in FEATURES_INITIAL if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # compute engineered if needed
    if "duty_cycle" not in df.columns:
        df["duty_cycle"]  = df["koi_duration"] / (24.0 * df["koi_period"].clip(lower=1e-6))
    if "depth_sqrt" not in df.columns:
        df["depth_sqrt"]  = np.sqrt(df["koi_depth"].clip(lower=1e-6))
    if "snr_per_hour" not in df.columns:
        df["snr_per_hour"]= df["koi_snr"] / df["koi_duration"].clip(lower=1e-3)
    if "log_period" not in df.columns:
        df["log_period"]  = np.log10(df["koi_period"].clip(lower=1e-6))
    if "log_duration" not in df.columns:
        df["log_duration"]= np.log10(df["koi_duration"].clip(lower=1e-6))
    if "log_depth" not in df.columns:
        df["log_depth"]   = np.log10(df["koi_depth"].clip(lower=1e-6))

    X = df[ALL_FEATURES].copy()

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
        preds.to_csv(out, index=False); print("Saved predictions →", out)
    else:
        print(preds.head())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    main(a.csv, a.out)
