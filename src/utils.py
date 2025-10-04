from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- features ----------
FEATURES_INITIAL = [
    "koi_period",    # days
    "koi_duration",  # hours
    "koi_depth",     # ppm
    "koi_prad",      # Earth radii
    "koi_snr",       # SNR/MES
]

ENGINEERED_FEATURES = [
    "duty_cycle",
    "depth_sqrt",
    "snr_per_hour",
    "log_period",
    "log_duration",
    "log_depth",
]

ALL_FEATURES = FEATURES_INITIAL + [f for f in ENGINEERED_FEATURES if f not in FEATURES_INITIAL]

# ---------- labels ----------
LABEL_COL = "koi_disposition"
LABEL_TO_INT = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

# alias support for Kepler/K2/TESS CSVs
ALIASES = {
    "koi_period":   ["koi_period", "period", "orbital_period", "pl_orbper", "per"],
    "koi_duration": ["koi_duration", "duration", "dur_hr", "transit_duration_hr", "dur", "duration_hours"],
    "koi_depth":    ["koi_depth", "depth_ppm", "depth", "tran_depth_ppm", "transit_depth_ppm"],
    "koi_prad":     ["koi_prad", "planet_radius_re", "pl_rade", "prad_re", "radius_re"],
    "koi_snr":      ["koi_snr", "snr", "model_snr", "mes", "koi_model_snr"],
}
LABEL_ALIASES = ["koi_disposition", "disposition", "tfopwg_disp", "k2_disposition", "disp"]

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_PATH = MODELS_DIR / "model.joblib"

# ---------- simple ensemble class (used by train/evaluate) ----------
class Ensemble:
    """
    A lightweight probability-averaging ensemble wrapper.
    Stores a list of trained models with predict_proba.
    """
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        ps = [m.predict_proba(X) for m in self.models]
        return np.mean(ps, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    @property
    def feature_importances_(self):
        imps = [m.feature_importances_ for m in self.models if hasattr(m, "feature_importances_")]
        return np.mean(np.vstack(imps), axis=0)

# ---------- plotting helpers ----------
def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def plot_feature_importances(model, feature_names=ALL_FEATURES, out_path: Path = None, top_n: int = 20):
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model has no attribute `feature_importances_`.")
    importances = np.array(model.feature_importances_, dtype=float)
    names = np.array(feature_names)
    if len(importances) != len(names):
        n = min(len(importances), len(names))
        importances, names = importances[:n], names[:n]
    order = np.argsort(importances)[::-1]
    names, vals = names[order][:top_n], importances[order][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4*len(names))))
    ax.barh(range(len(names)), vals)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("Importance"); ax.set_title("Feature Importances")
    fig.tight_layout()
    out_path = out_path or (PROCESSED_DIR / "feature_importances.png")
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
