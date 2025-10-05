from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

CANON_FEATURES = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_snr"]
CANON_LABEL = "koi_disposition"
VALID_LABELS = {"FALSE POSITIVE", "CANDIDATE", "CONFIRMED"}

# TESS-specific disposition mapping
TESS_DISPOSITION_MAP = {
    "PC": "CONFIRMED",     # Planet Confirmed
    "CP": "CONFIRMED",     # Confirmed Planet  
    "FP": "FALSE POSITIVE", # False Positive
    "KP": "CANDIDATE",     # Known Planet (treat as candidate for training)
    "APC": "CANDIDATE"     # Awaiting Planet Confirmation
}

ALIASES: Dict[str, List[str]] = {
    "koi_period":   ["koi_period", "period", "orbital_period", "pl_orbper", "per", "toi_period", "epic_period"],
    "koi_duration": ["koi_duration", "duration", "dur_hr", "transit_duration_hr", "dur", "duration_hours", "toi_duration", "epic_duration", "pl_trandurh", "pl_trandur"],
    "koi_depth":    ["koi_depth", "depth_ppm", "depth", "tran_depth_ppm", "transit_depth_ppm", "toi_depth", "epic_depth", "pl_trandep", "pl_transdep"],
    "koi_prad":     ["koi_prad", "planet_radius_re", "pl_rade", "prad_re", "radius_re", "toi_prad", "epic_prad"],
    "koi_snr":      ["koi_snr", "snr", "model_snr", "mes", "koi_model_snr", "toi_snr", "epic_snr", "pl_snr"],
}

LABEL_ALIASES = ["koi_disposition", "disposition", "tfopwg_disp", "k2_disposition", "disp", "toi_disposition", "epic_disposition"]

# Mission support
MISSION_ALIASES = ["mission", "facility", "pl_facility", "discoverymethod"]
MISSION_MAPPING = {
    "kepler": "Kepler",
    "k2": "K2", 
    "tess": "TESS",
    "transit": "Transit",
    "radial velocity": "RV",
    "direct imaging": "Direct"
}

def detect_mission(df: pd.DataFrame) -> str:
    """Detect mission type from column names or data content"""
    cols = [c.lower() for c in df.columns]
    
    # Check for mission-specific column prefixes or patterns
    if any(c.startswith('toi') or 'tfopwg' in c for c in cols):
        return "TESS"
    elif any(c.startswith('epic_') for c in cols):
        return "K2"
    elif any(c.startswith('koi_') for c in cols):
        return "Kepler"
    elif 'discoverymethod' in cols:
        # Check the discovery method to infer mission
        method_col = [c for c in df.columns if c.lower() == 'discoverymethod'][0]
        if len(df) > 0:
            common_methods = df[method_col].value_counts()
            if len(common_methods) > 0:
                top_method = common_methods.index[0].lower()
                if 'tess' in top_method:
                    return "TESS"
                elif 'k2' in top_method or 'kepler' in top_method:
                    return "Kepler/K2"
    
    # Check for explicit mission column
    for alias in MISSION_ALIASES:
        if alias.lower() in cols:
            mission_col = [c for c in df.columns if c.lower() == alias.lower()][0]
            if len(df) > 0:
                common_mission = df[mission_col].mode().iloc[0] if len(df[mission_col].mode()) > 0 else "Unknown"
                return MISSION_MAPPING.get(common_mission.lower(), common_mission)
    
    return "Unknown"

def robust_read(path: Path) -> pd.DataFrame:
    tries = [
        dict(sep=None, engine="python", encoding="utf-8-sig", comment="#"),
        dict(sep="\t", encoding="utf-8-sig", comment="#"),
        dict(sep=";", encoding="utf-8-sig", comment="#"),
    ]
    last = None
    for kw in tries:
        try: 
            return pd.read_csv(path, **kw)
        except Exception as e: 
            last = e
    raise RuntimeError(f"Could not parse {path}. Last error: {last}")

def resolve_aliases(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for canon, opts in ALIASES.items():
        for o in opts:
            if o.lower().strip() in lower:
                rename[lower[o.lower().strip()]] = canon
                break
    
    lab = None
    for o in LABEL_ALIASES:
        if o.lower().strip() in lower: 
            lab = lower[o.lower().strip()]
            break
    
    if lab and lab != CANON_LABEL: 
        rename[lab] = CANON_LABEL
    
    df2 = df.rename(columns=rename)
    
    # Add mission column if not present
    if "mission" not in df2.columns:
        detected_mission = detect_mission(df)
        df2["mission"] = detected_mission
        print(f"Detected mission: {detected_mission}")
    
    # Handle TESS-specific dispositions
    if CANON_LABEL in df2.columns and len(df2) > 0 and "TESS" in str(df2["mission"].iloc[0]):
        print("Converting TESS dispositions to standard format")
        df2[CANON_LABEL] = df2[CANON_LABEL].map(TESS_DISPOSITION_MAP).fillna(df2[CANON_LABEL])
    
    # Check which required features are available
    missing = [c for c in CANON_FEATURES if c not in df2.columns]
    available = [c for c in CANON_FEATURES if c in df2.columns]
    
    if len(available) < 2:  # Need at least 2 basic features (period + radius minimum)
        raise ValueError(f"Insufficient required columns. Available: {available}, Missing: {missing}")
    
    if missing:
        print(f"Warning: Missing optional columns: {missing}")
        # Create placeholder values for missing features
        if "koi_duration" in missing:
            print("Creating placeholder duration values (2.5% of period)")
            df2["koi_duration"] = df2["koi_period"] * 0.025 * 24  # 2.5% of period in hours
        if "koi_depth" in missing:
            print("Creating placeholder depth values")
            df2["koi_depth"] = 1000.0  # Default depth in ppm
        if "koi_snr" in missing:
            print("Creating placeholder SNR values")
            df2["koi_snr"] = 10.0  # Default SNR value
    
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
        if len(s) < 20: 
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - k*iqr, q3 + k*iqr
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
        if CANON_LABEL in df.columns: 
            keep += [CANON_LABEL]
        if "mission" in df.columns: 
            keep += ["mission"]  # Preserve mission info
            
        df = df[keep].copy()
        
        for c in CANON_FEATURES: 
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                
        if CANON_LABEL in df.columns:
            df[CANON_LABEL] = df[CANON_LABEL].astype(str).str.strip().str.upper()
            # Map REFUTED to FALSE POSITIVE for consistency
            df[CANON_LABEL] = df[CANON_LABEL].replace("REFUTED", "FALSE POSITIVE")
            df = df[df[CANON_LABEL].isin(VALID_LABELS)]
            
        df = drop_impossible(df)
        df = drop_too_missing(df, a.min_valid)
        
        if a.drop_outliers: 
            df = remove_outliers_iqr(df, [c for c in CANON_FEATURES if c in df.columns], a.iqr_k)
            
        df = df.drop_duplicates().reset_index(drop=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"[OK] Wrote -> {out} | Rows: {len(df)}")
        
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
