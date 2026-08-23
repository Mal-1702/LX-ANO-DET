"""
Linux Authentication Log Anomaly Classification Pipeline
=========================================================
Production-grade ML pipeline:
  Phase 1  - Data loading & audit
  Phase 2  - EDA with visualizations
  Phase 3  - Feature engineering (chronological, leakage-safe)
  Phase 4  - Chronological Train / Val / Test split
  Phase 5  - Model training (DT, RF, GB, XGBoost if installed) with class weighting & tuning
  Phase 6  - Evaluation: accuracy, macro/weighted F1, per-class report, confusion matrix, ROC-AUC
  Phase 7  - Model selection
  Phase 8  - Feature importance & robustness check on aux datasets
  Phase 9  - Artifact serialization + standalone inference API
  Phase 10 - Verification of deployment interfaces

Requirements: scikit-learn, pandas, numpy, matplotlib, seaborn, joblib
Optional:     xgboost
"""

import os
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MAIN_DATA       = PROJECT_ROOT / "linux_auth_logs_labeled.csv"
BALANCED_DATA   = PROJECT_ROOT / "linux_auth_logs_full(balanced).csv"
UNBALANCED_DATA = PROJECT_ROOT / "linux_auth_logs_full(new_unbalanced).csv"
INFERENCE_DATA  = PROJECT_ROOT / "linux_auth_logs_multiple_anomalies.csv"
TARGET = "anomaly_label"


# ==============================================================================
# PHASE 1 - DATA LOADING & AUDIT
# ==============================================================================

def load_and_audit(path, label: str = "dataset") -> pd.DataFrame:
    log.info(f"Loading {label} from {path}")
    df = pd.read_csv(path)
    return df