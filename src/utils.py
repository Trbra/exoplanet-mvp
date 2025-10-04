
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURES_INITIAL = [
	"koi_period",      # days
	"koi_duration",   # hours
	"koi_depth",      # ppm
	"koi_prad",       # Earth radii
	"koi_snr",        # signal-to-noise ratio
]

LABEL_COL = "koi_disposition"
LABEL_TO_INT = {
	"FALSE POSITIVE": 0,
	"CANDIDATE": 1,
	"CONFIRMED": 2
}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_PATH = MODELS_DIR / "model.joblib"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
