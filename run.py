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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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


# ==============================================================================
# PHASE 2 - EDA
# ==============================================================================

def run_eda(df: pd.DataFrame):
    log.info("Phase 2: Running EDA ...")
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["_hour"]  = ts.dt.hour
    df["_dow"]   = ts.dt.day_of_week
    df["_month"] = ts.dt.month

    palette = sns.color_palette("tab10", df[TARGET].nunique())
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("EDA - Linux Auth Logs Anomaly Detection", fontsize=14, fontweight="bold")

    # 1. Class distribution
    ax = axes[0, 0]
    counts = df[TARGET].value_counts()
    bars = ax.bar(counts.index, counts.values, color=palette)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 500,
                f"{v:,}", ha="center", fontsize=8)

    # 2. Attempts distribution per class
    ax = axes[0, 1]
    attempts_clean = pd.to_numeric(df["attempts"], errors="coerce").fillna(1)
    for cls, color in zip(df[TARGET].unique(), palette):
        ax.hist(attempts_clean[df[TARGET] == cls],
                bins=30, alpha=0.55, label=cls, density=True, color=color)
    ax.set_title("Attempts Distribution by Class")
    ax.set_xlabel("attempts")
    ax.legend(fontsize=7)

    # 3. Top 20 ports
    ax = axes[0, 2]
    port_clean = pd.to_numeric(df["port"], errors="coerce").fillna(0).astype(int)
    pc = port_clean.value_counts().head(20)
    ax.bar(pc.index.astype(str), pc.values, color="steelblue")
    ax.set_title("Top 20 Ports")
    ax.tick_params(axis="x", rotation=90)

    # 4. Anomaly count by hour
    ax = axes[1, 0]
    piv_h = df.groupby(["_hour", TARGET]).size().unstack(fill_value=0)
    for col, color in zip(piv_h.columns, palette):
        ax.plot(piv_h.index, piv_h[col], label=col, marker=".", color=color)
    ax.set_title("Anomaly Count by Hour of Day")
    ax.set_xlabel("Hour")
    ax.legend(fontsize=7)

    # 5. Anomaly count by day-of-week
    ax = axes[1, 1]
    piv_d = df.groupby(["_dow", TARGET]).size().unstack(fill_value=0)
    piv_d.plot(kind="bar", ax=ax, colormap="tab10", legend=True)
    ax.set_title("Anomaly Count by Day of Week (0=Mon)")
    ax.set_xlabel("Day")
    ax.legend(fontsize=7)

    # 6. Anomaly count by service
    ax = axes[1, 2]
    piv_s = df.groupby(["service", TARGET]).size().unstack(fill_value=0)
    piv_s.plot(kind="bar", ax=ax, colormap="tab10", stacked=True)
    ax.set_title("Anomaly Count by Service")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=7)

    # 7. Anomaly by status
    ax = axes[2, 0]
    piv_st = df.groupby(["status", TARGET]).size().unstack(fill_value=0)
    piv_st.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_title("Anomaly Count by Status")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=7)

    # 8. Protocol missingness by class
    ax = axes[2, 1]
    df["_proto_miss"] = df["protocol"].isna().astype(int)
    piv_pm = df.groupby([TARGET, "_proto_miss"]).size().unstack(fill_value=0)
    piv_pm.columns = ["Not Missing", "Missing"]
    piv_pm.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("Protocol Missingness by Class")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)

    # 9. Attempts vs log(port) scatter
    ax = axes[2, 2]
    sample = df.sample(min(5000, len(df)), random_state=SEED)
    sample_attempts = pd.to_numeric(sample["attempts"], errors="coerce").fillna(1)
    sample_port = pd.to_numeric(sample["port"], errors="coerce").fillna(0).clip(lower=0)
    color_map = dict(zip(df[TARGET].unique(), palette))
    for cls, grp_idx in sample.groupby(TARGET).groups.items():
        ax.scatter(sample_attempts.loc[grp_idx], np.log1p(sample_port.loc[grp_idx]),
                   alpha=0.3, s=5, c=[color_map[cls]], label=cls)
    ax.set_title("Attempts vs log1p(port) [sampled 5k]")
    ax.set_xlabel("attempts")
    ax.set_ylabel("log1p(port)")
    ax.legend(fontsize=7, markerscale=3)

    plt.tight_layout()
    eda_path = OUTPUT_DIR / "eda_overview.png"
    plt.savefig(eda_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"EDA overview saved -> {eda_path}")

    # Correlation heatmap
    corr_df = pd.DataFrame({
        "attempts": attempts_clean,
        "port": port_clean,
        "_hour": df["_hour"],
        "_dow": df["_dow"],
        "_month": df["_month"],
    })
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlations")
    plt.tight_layout()
    corr_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(corr_path, dpi=120)
    plt.close()
    log.info(f"Correlation heatmap saved -> {corr_path}")


# ==============================================================================
# PHASE 3 - FEATURE ENGINEERING
# ==============================================================================

def transform_features(df: pd.DataFrame, maps: dict = None) -> pd.DataFrame:
    """
    Apply feature transformations safely and deterministically.
    Handles missing columns, non-numeric values, and unseen categories without crashing.
    """
    df = df.copy()

    # 1. Attempts: numeric coercion with robust bounds [1, inf)
    attempts_raw = df["attempts"] if "attempts" in df.columns else 1
    attempts_num = pd.to_numeric(attempts_raw, errors="coerce").fillna(1.0)
    df["attempts"] = np.maximum(attempts_num.values, 1.0)

    # 2. Port: numeric coercion with bounds [0, 65535] and log1p transform
    port_raw = df["port"] if "port" in df.columns else 0
    port_num = pd.to_numeric(port_raw, errors="coerce").fillna(0.0)
    df["port"] = np.clip(port_num.values, 0.0, 65535.0)
    df["log_port"] = np.log1p(df["port"].values)

    # 3. Temporal features
    ts_raw = df["timestamp"] if "timestamp" in df.columns else pd.NaT
    ts = pd.to_datetime(ts_raw, errors="coerce")
    df["hour"]       = ts.dt.hour.fillna(0).astype(int)
    df["dow"]        = ts.dt.day_of_week.fillna(0).astype(int)
    df["month"]      = ts.dt.month.fillna(1).astype(int)
    df["day"]        = ts.dt.day.fillna(1).astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_night"]   = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    return df


def get_X(df_transformed: pd.DataFrame) -> np.ndarray:
    """
    Extract strictly ordered feature matrix X and ensure no NaN/Inf reaches models.
    """
    cols = [c for c in ["attempts", "log_port", "hour", "dow", "month", "day", "is_weekend", "is_night"] if c in df_transformed.columns]
    X = df_transformed[cols].values.astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ==============================================================================
# PHASE 4 - CHRONOLOGICAL SPLIT
# ==============================================================================

def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """
    Temporal split: 70% Train, 15% Validation, 15% Test.
    Prevents lookahead leakage across authentication sequences.
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n  = len(df_sorted)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    train = df_sorted.iloc[:t1].copy()
    val   = df_sorted.iloc[t1:t2].copy()
    test  = df_sorted.iloc[t2:].copy()
    log.info(f"Chronological split -> train={len(train):,}  val={len(val):,}  test={len(test):,}")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"\n{name} class distribution:\n{split[TARGET].value_counts()}")
    return train, val, test