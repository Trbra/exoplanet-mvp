from __future__ import annotations
import argparse, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .utils import (
    RAW_DIR, PROCESSED_DIR, PREPROCESSOR_PATH,
    FEATURES_INITIAL, ENGINEERED_FEATURES, ALL_FEATURES,
    LABEL_COL, LABEL_TO_INT, ALIASES, LABEL_ALIASES
)

def robust_read(path):
    tries = [
        dict(sep=None, engine="python", encoding="utf-8-sig", comment="#"),
        dict(sep="\t", encoding="utf-8-sig", comment="#"),
        dict(sep=";", encoding="utf-8-sig", comment="#"),
    ]
    last = None
    for kw in tries:
        try: return pd.read_csv(path, **kw)
        except Exception as e: last = e
    raise RuntimeError(f"Could not parse {path}. Last error: {last}")

def resolve_aliases(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for canon, opts in ALIASES.items():
        for o in opts:
            if o.lower().strip() in lower:
                rename[lower[o.lower().strip()]] = canon; break
    lab = None
    for o in LABEL_ALIASES:
        if o.lower().strip() in lower: lab = lower[o.lower().strip()]; break
    if lab and lab != LABEL_COL: rename[lab] = LABEL_COL
    df2 = df.rename(columns=rename)
    missing = [c for c in FEATURES_INITIAL if c not in df2.columns]
    if missing: raise ValueError(f"Missing required columns after alias resolution: {missing}")
    return df2

def build_preprocessor():
    return Pipeline([("imputer", SimpleImputer(strategy="median"))])

def main(csv: str | None):
    csv_path = csv or (RAW_DIR / "kepler_koi.csv")
    df = robust_read(csv_path)
    df = resolve_aliases(df)

    if LABEL_COL in df.columns:
        df["label_int"] = df[LABEL_COL].map(LABEL_TO_INT)
        df = df.dropna(subset=["label_int"])
        df["label_int"] = df["label_int"].astype(int)

    # coerce numeric for raw features
    for c in FEATURES_INITIAL:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # engineered features
    df["duty_cycle"]   = df["koi_duration"] / (24.0 * df["koi_period"].clip(lower=1e-6))
    df["depth_sqrt"]   = np.sqrt(df["koi_depth"].clip(lower=1e-6))
    df["snr_per_hour"] = df["koi_snr"] / df["koi_duration"].clip(lower=1e-3)
    df["log_period"]   = np.log10(df["koi_period"].clip(lower=1e-6))
    df["log_duration"] = np.log10(df["koi_duration"].clip(lower=1e-6))
    df["log_depth"]    = np.log10(df["koi_depth"].clip(lower=1e-6))

    use_cols = [c for c in ALL_FEATURES if c in df.columns]

    # keep rows with enough info (loose gate)
    df = df[df[use_cols].notna().sum(axis=1) >= max(4, int(0.7*len(use_cols)))]

    X = df[use_cols].copy()
    y = df["label_int"] if "label_int" in df.columns else None

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        pre = build_preprocessor()
        pre.fit(X_train)
        X_train_t, X_test_t = pre.transform(X_train), pre.transform(X_test)

        joblib.dump(pre, PREPROCESSOR_PATH)
        pd.DataFrame(X_train_t, columns=use_cols).to_parquet(PROCESSED_DIR / "X_train.parquet", index=False)
        pd.DataFrame(X_test_t,  columns=use_cols).to_parquet(PROCESSED_DIR / "X_test.parquet",  index=False)
        pd.Series(y_train).to_csv(PROCESSED_DIR / "y_train.csv", index=False)
        pd.Series(y_test).to_csv(PROCESSED_DIR / "y_test.csv", index=False)
        print("Preprocessing complete. Saved arrays and preprocessor.")
    else:
        pre = build_preprocessor()
        pre.fit(X)
        joblib.dump(pre, PREPROCESSOR_PATH)
        pd.DataFrame(pre.transform(X), columns=use_cols).to_parquet(PROCESSED_DIR / "X_all.parquet", index=False)
        print("Preprocessor fitted on all rows (no labels). Saved.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None)
    a = ap.parse_args()
    main(a.csv)
