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

    sep = "=" * 60
    print(f"\n{sep}\nAUDIT: {label.upper()}\n{sep}")
    print(f"Shape      : {df.shape}")
    print(f"Columns    : {df.columns.tolist()}")
    print(f"Duplicates : {df.duplicated().sum()}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    if TARGET in df.columns:
        vc  = df[TARGET].value_counts()
        pct = (df[TARGET].value_counts(normalize=True) * 100).round(2)
        print(f"\nClass distribution:\n{vc}")
        print(f"\nClass %:\n{pct}")

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        print(f"\nTimestamp range : {ts.min()} -> {ts.max()}")
        print(f"Timestamp nulls : {ts.isna().sum()}")

    print("\nUnique counts per column:")
    for c in df.columns:
        print(f"  {c:25s}: {df[c].nunique():>8,}")

    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    str_cols  = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "string"]
    low_card  = [c for c in str_cols if df[c].nunique() <= 20]
    high_card = [c for c in str_cols if df[c].nunique() > 20]
    print(f"\nNumeric columns        : {num_cols}")
    print(f"Low-cardinality (<=20) : {low_card}")
    print(f"High-cardinality (>20) : {high_card}")

    return df.drop_duplicates().reset_index(drop=True)