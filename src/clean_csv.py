from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Dict, List
import numpy as np, pandas as pd

CANON_FEATURES = ["koi_period","koi_duration","koi_depth","koi_prad","koi_snr"]
CANON_LABEL = "koi_disposition"
VALID_LABELS = {"FALSE POSITIVE","CANDIDATE","CONFIRMED"}

ALIASES: Dict[str, List[str]] = {
    "koi_period":   ["koi_period","period","orbital_period","pl_orbper","per"],
    "koi_duration": ["koi_duration","duration","dur_hr","transit_duration_hr","dur","duration_hours"],
    "koi_depth":    ["koi_depth","depth_ppm","depth","tran_depth_ppm","transit_depth_ppm"],
    "koi_prad":     ["koi_prad","planet_radius_re","pl_rade","prad_re","radius_re"],
    "koi_snr":      ["koi_snr","snr","model_snr","mes","koi_model_snr"],
}
LABEL_ALIASES = ["koi_disposition","disposition","tfopwg_disp","k2_disposition","disp"]

def robust_read(path: Path) -> pd.DataFrame:
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
    if lab and lab != CANON_LABEL: rename[lab] = CANON_LABEL
    df2 = df.rename(columns=rename)
    missing = [c for c in CANON_FEATURES if c not in df2.columns]
    if missing: raise ValueError(f"Missing required columns after alias resolution: {missing}")
    return df2

def drop_impossible(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["koi_period"] > 0]
    df = df[(df["koi_duration"] > 0) & (df["koi_duration"] < df["koi_period"]*24)]
    df = df[df["koi_depth"] > 0]
    df = df[df["koi_prad"] > 0]
    df = df[df["koi_snr"] >= 0]
    return df

def drop_too_missing(df: pd.DataFrame, min_valid=4) -> pd.DataFrame:
    return df[df[CANON_FEATURES].notna().sum(axis=1) >= min_valid]

def remove_outliers_iqr(df: pd.DataFrame, cols: List[str], k: float=3.0) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 20: continue
        q1, q3 = s.quantile([0.25,0.75]); iqr = q3-q1; low, high = q1-k*iqr, q3+k*iqr
        mask &= df[c].between(low, high)
    return df[mask]

def main():
    ap = argparse.ArgumentParser(description="Clean raw exoplanet CSV.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", default="data/processed/kepler_koi_clean.csv")
    ap.add_argument("--min-valid", type=int, default=4)
    ap.add_argument("--drop-outliers", action="store_true")
    ap.add_argument("--iqr-k", type=float, default=3.0)
    a = ap.parse_args()

    inp, out = Path(a.inp), Path(a.out)
    try:
        df = robust_read(inp)
        df = resolve_aliases(df)
        keep = [c for c in CANON_FEATURES if c in df.columns]
        if CANON_LABEL in df.columns: keep += [CANON_LABEL]
        df = df[keep].copy()
        for c in CANON_FEATURES: df[c] = pd.to_numeric(df[c], errors="coerce")
        if CANON_LABEL in df.columns:
            df[CANON_LABEL] = df[CANON_LABEL].astype(str).str.strip().str.upper()
            df = df[df[CANON_LABEL].isin(VALID_LABELS)]
        df = drop_impossible(df)
        df = drop_too_missing(df, a.min_valid)
        if a.drop_outliers: df = remove_outliers_iqr(df, CANON_FEATURES, a.iqr_k)
        df = df.drop_duplicates().reset_index(drop=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"[OK] Wrote -> {out} | Rows: {len(df)}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
