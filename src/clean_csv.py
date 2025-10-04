"""
Clean exoplanet CSVs (Kepler/TESS/K2) into a consistent, model-ready file.

Usage (from project root):
    python -m src.clean_csv --in data/raw/kepler_koi.csv --out data/processed/kepler_koi_clean.csv --drop-outliers

If you haven't created data/processed yet:
    mkdir data\processed  (Windows)  or  mkdir -p data/processed (Mac/Linux)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---- Canonical feature & label names used by the pipeline
CANON_FEATURES = [
    "koi_period",    # days
    "koi_duration",  # hours
    "koi_depth",     # ppm
    "koi_prad",      # Earth radii
    "koi_snr",       # SNR/MES
]
CANON_LABEL = "koi_disposition"
VALID_LABELS = {"FALSE POSITIVE", "CANDIDATE", "CONFIRMED"}

# ---- Common header aliases across missions (edit/extend as needed)
ALIASES: Dict[str, List[str]] = {
    "koi_period":   ["koi_period", "period", "orbital_period", "pl_orbper", "per"],
    "koi_duration": ["koi_duration", "duration", "dur_hr", "transit_duration_hr", "dur", "duration_hours"],
    "koi_depth":    ["koi_depth", "depth_ppm", "depth", "tran_depth_ppm", "transit_depth_ppm"],
    "koi_prad":     ["koi_prad", "planet_radius_re", "pl_rade", "prad_re", "radius_re"],
    "koi_snr":      ["koi_snr", "snr", "model_snr", "mes", "koi_model_snr"],
}
LABEL_ALIASES = [
    "koi_disposition", "disposition", "tfopwg_disp", "k2_disposition", "disp"
]


def _robust_read(path: Path) -> pd.DataFrame:
    """
    Read CSV with auto delimiter detection and common quirks handled.
    Tries: auto (python engine), tab, semicolon. Skips '#' comments and BOM.
    """
    tries = [
        dict(sep=None, engine="python", encoding="utf-8-sig", comment="#"),
        dict(sep="\t", encoding="utf-8-sig", comment="#"),
        dict(sep=";", encoding="utf-8-sig", comment="#"),
    ]
    last_err = None
    for kw in tries:
        try:
            return pd.read_csv(path, **kw)
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Could not parse file {path}. Last error: {last_err}\n"
        "Check the delimiter (comma/tab/semicolon) or if you saved an HTML page instead of CSV."
    )


def _resolve_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename mission-specific headers to canonical names the pipeline expects.
    """
    cols_lower = {c: c for c in df.columns}
    # normalize trim & lower comparison but keep original case in df
    lower_map = {c.lower().strip(): c for c in df.columns}

    rename_map = {}

    # features
    for canonical, options in ALIASES.items():
        found = None
        for opt in options:
            key = opt.lower().strip()
            if key in lower_map:
                found = lower_map[key]
                break
        if found:
            rename_map[found] = canonical

    # label
    found_label = None
    for opt in LABEL_ALIASES:
        key = opt.lower().strip()
        if key in lower_map:
            found_label = lower_map[key]
            break
    if found_label and found_label != CANON_LABEL:
        rename_map[found_label] = CANON_LABEL

    df2 = df.rename(columns=rename_map)

    # sanity check
    missing = [c for c in CANON_FEATURES if c not in df2.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after alias resolution: {missing}\n"
            f"Present columns: {list(df.columns)[:25]}..."
        )
    return df2


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Coerce given columns to numeric; invalid values -> NaN. Also strip whitespace on all columns.
    """
    # strip whitespace in object columns
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _drop_impossible_and_clip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with physically impossible or clearly invalid values.
    """
    # period: must be >0
    df = df[df["koi_period"] > 0]
    # duration: must be >0 and not longer than period*24h (very loose upper bound)
    df = df[(df["koi_duration"] > 0) & (df["koi_duration"] < df["koi_period"] * 24)]
    # depth: must be >0
    df = df[df["koi_depth"] > 0]
    # radius: >0
    df = df[df["koi_prad"] > 0]
    # snr: >= 0
    df = df[df["koi_snr"] >= 0]
    return df


def _drop_too_missing(df: pd.DataFrame, required: List[str], min_valid: int) -> pd.DataFrame:
    """
    Drop rows that have fewer than min_valid non-null among required columns.
    """
    valid_counts = df[required].notna().sum(axis=1)
    return df[valid_counts >= min_valid]


def _iqr_outlier_mask(s: pd.Series, k: float = 3.0) -> pd.Series:
    """
    Return a boolean mask where values are within [Q1 - k*IQR, Q3 + k*IQR].
    """
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return s.between(low, high)


def _remove_outliers_iqr(df: pd.DataFrame, cols: List[str], k: float = 3.0) -> pd.DataFrame:
    """
    Remove rows that are outliers in any of the specified columns using IQR rule.
    Uses a fairly loose k=3.0 by default.
    """
    mask = pd.Series(True, index=df.index)
    for c in cols:
        # only consider if we have at least a few non-nulls
        if df[c].notna().sum() > 20:
            mask &= _iqr_outlier_mask(df[c].astype(float), k=k).fillna(False)
    return df[mask]


def _class_balance(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False)
    total = counts.sum()
    parts = [f"{idx}: {cnt} ({cnt/total:.1%})" for idx, cnt in counts.items()]
    return " | ".join(parts)


def clean_csv(
    in_path: Path,
    out_path: Path,
    min_valid: int = 4,
    drop_outliers: bool = False,
    iqr_k: float = 3.0,
) -> Tuple[int, int]:
    """
    Main cleaning pipeline. Returns (n_before, n_after).
    """
    df = _robust_read(in_path)
    n_before = len(df)

    # Resolve aliases
    df = _resolve_aliases(df)

    # Keep only canonical features + label (if present)
    keep_cols = [c for c in CANON_FEATURES if c in df.columns]
    has_label = CANON_LABEL in df.columns
    if has_label:
        keep_cols += [CANON_LABEL]
    df = df[keep_cols].copy()

    # Coerce to numeric for features
    df = _coerce_numeric(df, CANON_FEATURES)

    # Standardize label strings (if present)
    if has_label:
        df[CANON_LABEL] = df[CANON_LABEL].astype(str).str.strip().str.upper()
        # Keep only the three target classes (drop others/unknown)
        df = df[df[CANON_LABEL].isin(VALID_LABELS)]

    # Drop impossible values
    df = _drop_impossible_and_clip(df)

    # Drop rows missing too many required feature values
    df = _drop_too_missing(df, required=CANON_FEATURES, min_valid=min_valid)

    # Optional: remove outliers (loose filter)
    if drop_outliers:
        df = _remove_outliers_iqr(df, cols=CANON_FEATURES, k=iqr_k)

    # Deduplicate identical rows
    df = df.drop_duplicates()

    # Reset index and save
    df = df.reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    return n_before, len(df)


def main():
    p = argparse.ArgumentParser(description="Clean exoplanet CSV (Kepler/TESS/K2) into a consistent format.")
    p.add_argument("--in", dest="inp", required=True, help="Input CSV path (raw file).")
    p.add_argument("--out", dest="out", default="data/processed/kepler_koi_clean.csv", help="Output cleaned CSV path.")
    p.add_argument("--min-valid", type=int, default=4, help="Min non-null required among the 5 features.")
    p.add_argument("--drop-outliers", action="store_true", help="Apply IQR-based outlier removal (k=3).")
    p.add_argument("--iqr-k", type=float, default=3.0, help="IQR multiplier for outlier removal.")
    args = p.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)

    try:
        before, after = clean_csv(
            in_path=in_path,
            out_path=out_path,
            min_valid=args.min_valid,
            drop_outliers=args.drop_outliers,
            iqr_k=args.iqr_k,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] Wrote cleaned CSV → {out_path}")
    print(f"Rows: {before} → {after}  (removed {before - after})")

    # Optional: show class balance if label present
    try:
        df_clean = pd.read_csv(out_path)
        if CANON_LABEL in df_clean.columns:
            print("Class balance (cleaned):", _class_balance(df_clean[CANON_LABEL]))
    except Exception:
        pass


if __name__ == "__main__":
    main()