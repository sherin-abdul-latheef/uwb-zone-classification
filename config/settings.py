"""
Central configuration: every script reads its file paths from here.

By default the data files are expected in the `data/` folder at the project
root (see data/README.md). To keep the data elsewhere, set the environment
variable UWB_DATA_DIR to that folder instead of editing this file.
"""

import os
from pathlib import Path

# ── Project folders ────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = Path(os.environ.get("UWB_DATA_DIR", ROOT / "data"))
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures" / "output"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Fingerprint database (one file per preprocessing stage) ────────────────
RAW_PATH          = DATA_DIR / "raw_data.xlsx"      # raw anchor distances
ZQCM_PATH         = DATA_DIR / "zqcm_data.xlsx"     # after zero-value correction
GQCM_PATH         = DATA_DIR / "gqcm_data.xlsx"     # gross errors flagged as NaN
GQCM_Hermite_path = DATA_DIR / "gqcm_hermite.xlsx"  # GQCM gaps refilled (Hermite)
DATA_PATH         = DATA_DIR / "total_nqcm.xlsx"    # final NQCM features + labels

# ── Live validation trajectory (separate 80 s recording) ───────────────────
LIVE_TRAJECTORY_PATH = DATA_DIR / "live_zone1_nqcm.xlsx"

# ── Files shipped with the repository ──────────────────────────────────────
FOLD_INDICES_PATH = ROOT / "config" / "fold_indices.pkl"          # frozen CV folds
PREDICTIONS_PATH  = RESULTS_DIR / "predictions_live_zone1.npy"    # classifier output on the live trajectory

# ── K values evaluated in cross-validation ─────────────────────────────────
k_values = [3, 5, 7, 11, 21, 31, 37, 41, 47, 51, 57, 61, 67]
