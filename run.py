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

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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

PROTO_MAP = {"SSH": 0, "SSH2": 0, "RDP": 1, "TELNET": 2, "unknown": 3}

FEATURE_COLS = [
    "attempts", "log_port",
    "hour", "dow", "month", "day", "is_weekend", "is_night",
    "protocol_enc",
    "server_enc", "service_enc", "status_enc",
    "source_ip_freq", "username_freq",
    "comment_failed", "comment_accepted",
]


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

def fit_feature_maps(df_train: pd.DataFrame) -> dict:
    """
    Fit all encoding maps strictly on train split to prevent data leakage.
    Unknown/unseen categories at inference time map to default indices (-1 / 0.0).
    """
    maps = {}
    for col in ["server", "service", "status"]:
        series = df_train[col].fillna("unknown").astype(str)
        le = LabelEncoder()
        le.fit(series)
        maps[f"{col}_le"] = {cls: int(i) for i, cls in enumerate(le.classes_)}
    for col in ["source_ip", "username"]:
        series = df_train[col].fillna("unknown").astype(str)
        maps[f"{col}_freq"] = series.value_counts(normalize=True).to_dict()
    return maps


def transform_features(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
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

    # 4. Protocol encoding
    proto_series = df["protocol"].fillna("unknown").astype(str) if "protocol" in df.columns else pd.Series(["unknown"] * len(df))
    df["protocol_enc"] = proto_series.map(PROTO_MAP).fillna(PROTO_MAP["unknown"]).astype(int)

    # 5. Categorical label encodings (unseen -> -1)
    for col in ["server", "service", "status"]:
        if col in df.columns:
            s = df[col].fillna("unknown").astype(str)
        else:
            s = pd.Series(["unknown"] * len(df))
        encoder_dict = maps.get(f"{col}_le", {})
        df[f"{col}_enc"] = s.map(encoder_dict).fillna(-1).astype(int)

    # 6. Frequency encodings (unseen/cold-start -> 0.0)
    for col in ["source_ip", "username"]:
        if col in df.columns:
            s = df[col].fillna("unknown").astype(str)
        else:
            s = pd.Series(["unknown"] * len(df))
        freq_dict = maps.get(f"{col}_freq", {})
        df[f"{col}_freq"] = s.map(freq_dict).fillna(0.0).astype(float)

    # 7. Comment keyword indicators
    comment_series = df["comment"].fillna("").astype(str) if "comment" in df.columns else pd.Series([""] * len(df))
    df["comment_failed"]   = comment_series.str.contains("failed|Failed", na=False).astype(int)
    df["comment_accepted"] = comment_series.str.contains("accepted|Accepted", na=False).astype(int)

    return df


def get_X(df_transformed: pd.DataFrame) -> np.ndarray:
    """
    Extract strictly ordered feature matrix X and ensure no NaN/Inf reaches models.
    """
    for c in FEATURE_COLS:
        if c not in df_transformed.columns:
            df_transformed[c] = 0.0
    X = df_transformed[FEATURE_COLS].values.astype(np.float32)
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


# ==============================================================================
# PHASE 5 - MODEL TRAINING
# ==============================================================================

def build_class_weight_dicts(y_train: np.ndarray, classes):
    """Build class weight dictionaries matching encoded integer targets."""
    numeric_classes = np.arange(len(classes))
    weights = compute_class_weight("balanced", classes=numeric_classes, y=y_train)
    cw_int = dict(enumerate(weights))
    cw_str = {classes[i]: float(w) for i, w in cw_int.items()}
    return cw_int, cw_str


def quick_param_search(estimator, param_grid: dict, X_sub, y_sub, n_iter=5) -> dict:
    """Perform 3-fold Stratified CV hyperparameter search scored on Macro-F1."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        estimator, param_grid, n_iter=n_iter, cv=cv,
        scoring="f1_macro", random_state=SEED, n_jobs=1, verbose=0, refit=False,
    )
    search.fit(X_sub, y_sub)
    log.info(f"  Best CV macro-F1={search.best_score_:.4f}  params={search.best_params_}")
    return search.best_params_


def make_sub_sample(X_train, y_train, size=20_000):
    """Generate reproducible stratified sub-sample for fast parameter tuning."""
    rng = np.random.default_rng(SEED)
    classes, counts = np.unique(y_train, return_counts=True)
    idx = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y_train == cls)[0]
        take = max(1, int(size * cnt / len(y_train)))
        idx.extend(rng.choice(cls_idx, min(take, len(cls_idx)), replace=False).tolist())
    idx = np.array(sorted(idx))
    return X_train[idx], y_train[idx]


def train_decision_tree(X_train, y_train, X_sub, y_sub, cw_int):
    log.info("Training DecisionTree ...")
    param_grid = {
        "max_depth": [10, 15, 20, 25],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    }
    best = quick_param_search(
        DecisionTreeClassifier(random_state=SEED, class_weight=cw_int),
        param_grid, X_sub, y_sub, n_iter=15
    )
    model = DecisionTreeClassifier(**best, random_state=SEED, class_weight=cw_int)
    model.fit(X_train, y_train)
    log.info("  DecisionTree done.")
    return model


def train_random_forest(X_train, y_train, X_sub, y_sub, cw_int):
    log.info("Training RandomForest ...")
    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [12, 16, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }
    best = quick_param_search(
        RandomForestClassifier(random_state=SEED, class_weight=cw_int, n_jobs=1),
        param_grid, X_sub, y_sub, n_iter=6
    )
    model = RandomForestClassifier(**best, random_state=SEED, class_weight=cw_int, n_jobs=-1)
    model.fit(X_train, y_train)
    log.info("  RandomForest done.")
    return model


def train_gradient_boosting(X_train, y_train, X_sub, y_sub, cw_str, le):
    log.info("Training GradientBoosting on 100k subsample with sample_weight ...")
    param_grid = {
        "n_estimators": [80, 120],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    best = quick_param_search(
        GradientBoostingClassifier(random_state=SEED),
        param_grid, X_sub, y_sub, n_iter=6
    )
    n_gb = min(100_000, len(X_train))
    rng_gb = np.random.default_rng(SEED)
    idx  = sorted(rng_gb.choice(len(X_train), n_gb, replace=False))
    X_gb, y_gb = X_train[idx], y_train[idx]
    sw_gb = np.array([cw_str[le.classes_[c]] for c in y_gb])
    model = GradientBoostingClassifier(**best, random_state=SEED)
    model.fit(X_gb, y_gb, sample_weight=sw_gb)
    log.info(f"  GradientBoosting done (n={n_gb:,}).")
    return model


def train_xgboost_model(X_train, y_train, X_sub, y_sub, cw_str, le):
    log.info("Training XGBoost with exact sample weights ...")
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    best = quick_param_search(
        XGBClassifier(random_state=SEED, eval_metric="mlogloss",
                      n_jobs=1, tree_method="hist", verbosity=0),
        param_grid, X_sub, y_sub, n_iter=6
    )
    # Ensure sample_weight matches y_train length exactly
    sw_train = np.array([cw_str[le.classes_[c]] for c in y_train], dtype=np.float32)
    model = XGBClassifier(
        **best, random_state=SEED, eval_metric="mlogloss",
        n_jobs=-1, tree_method="hist", verbosity=0
    )
    model.fit(X_train, y_train, sample_weight=sw_train)
    log.info("  XGBoost done.")
    return model


def train_all_models(X_train, y_train, le, cw_int, cw_str):
    X_sub, y_sub = make_sub_sample(X_train, y_train, size=25_000)
    trained = {}
    trained["DecisionTree"] = train_decision_tree(X_train, y_train, X_sub, y_sub, cw_int)
    trained["RandomForest"] = train_random_forest(X_train, y_train, X_sub, y_sub, cw_int)
    trained["GradientBoosting"] = train_gradient_boosting(X_train, y_train, X_sub, y_sub, cw_str, le)
    if XGBOOST_AVAILABLE:
        trained["XGBoost"] = train_xgboost_model(X_train, y_train, X_sub, y_sub, cw_str, le)
    return trained


# ==============================================================================
# PHASE 6 - EVALUATION
# ==============================================================================

def evaluate_model(name: str, model, X, y_enc, le, split_name="val") -> dict:
    y_pred_enc = model.predict(X)
    y_true_lbl = le.inverse_transform(y_enc)
    y_pred_lbl = le.inverse_transform(y_pred_enc)

    unique_labels = sorted(set(y_true_lbl) | set(y_pred_lbl))

    acc = accuracy_score(y_enc, y_pred_enc)
    mf1 = f1_score(y_enc, y_pred_enc, average="macro")
    wf1 = f1_score(y_enc, y_pred_enc, average="weighted")

    print(f"\n{'-'*60}")
    print(f"MODEL: {name}  |  Split: {split_name}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Macro F1     : {mf1:.4f}")
    print(f"  Weighted F1  : {wf1:.4f}")
    print(classification_report(y_true_lbl, y_pred_lbl, target_names=unique_labels, digits=4))

    roc = None
    if hasattr(model, "predict_proba"):
        try:
            if len(np.unique(y_enc)) >= 2:
                y_prob = model.predict_proba(X)
                roc = roc_auc_score(y_enc, y_prob, multi_class="ovr", average="macro")
                print(f"  ROC-AUC (macro OVR): {roc:.4f}")
        except Exception as e:
            log.warning(f"ROC-AUC calculation failed for {name} ({split_name}): {e}")

    cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=unique_labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(f"Confusion Matrix - {name} ({split_name})")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"cm_{name}_{split_name}.png", dpi=110, bbox_inches="tight")
    plt.close()

    return {"name": name, "split": split_name,
            "accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1, "roc_auc": roc}


def evaluate_all(trained_models, X_val, y_val, X_test, y_test, le) -> list:
    results = []
    for name, model in trained_models.items():
        results.append(evaluate_model(name, model, X_val,  y_val,  le, "val"))
        results.append(evaluate_model(name, model, X_test, y_test, le, "test"))
    return results


# ==============================================================================
# PHASE 7 - MODEL SELECTION
# ==============================================================================

def select_best_model(results: list, trained_models: dict):
    test_res = [r for r in results if r["split"] == "test"]
    if not test_res:
        raise ValueError("No test results found in evaluation results.")

    df_res = pd.DataFrame(test_res).sort_values("macro_f1", ascending=False)

    print(f"\n{'='*60}")
    print("MODEL SELECTION - Test Set Summary")
    print("="*60)
    print(df_res[["name", "accuracy", "macro_f1", "weighted_f1", "roc_auc"]].to_string(index=False))

    best_name = df_res.iloc[0]["name"]
    best_f1   = df_res.iloc[0]["macro_f1"]
    print(f"\n[BEST] {best_name}  (macro F1 = {best_f1:.4f})")

    if len(df_res) > 1:
        second = df_res.iloc[1]
        gap = best_f1 - second["macro_f1"]
        print(f"[2nd]  {second['name']}  (macro F1 = {second['macro_f1']:.4f}, gap={gap:.4f})")

    return best_name, trained_models[best_name]


# ==============================================================================
# PHASE 8 - FEATURE IMPORTANCE & ROBUSTNESS
# ==============================================================================

def show_feature_importance(model, model_name: str):
    if not hasattr(model, "feature_importances_"):
        return
    if len(model.feature_importances_) != len(FEATURE_COLS):
        log.warning(f"Feature importances length ({len(model.feature_importances_)}) != FEATURE_COLS length ({len(FEATURE_COLS)}) for {model_name}; skipping.")
        return
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\nTop 10 features ({model_name}):")
    print(fi.head(10).to_string())
    plt.figure(figsize=(9, 5))
    fi.head(16).plot(kind="bar", color="steelblue")
    plt.title(f"Feature Importances - {model_name}")
    plt.ylabel("Importance")
    plt.tight_layout()
    path = OUTPUT_DIR / f"feature_importance_{model_name}.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    log.info(f"Feature importance chart saved -> {path}")


def robustness_check(best_model, maps: dict, le: LabelEncoder):
    log.info("Phase 8: Robustness check on auxiliary datasets ...")
    known = set(le.classes_)
    for label, path in [("balanced", BALANCED_DATA), ("unbalanced", UNBALANCED_DATA)]:
        if not path.exists():
            log.warning(f"Auxiliary dataset not found: {path}")
            continue
        df_aux = pd.read_csv(path)
        if TARGET not in df_aux.columns:
            log.warning(f"No label column in {label} dataset; skipping.")
            continue
        df_aux = df_aux[df_aux[TARGET].isin(known)].copy()
        if df_aux.empty:
            log.warning(f"No known-class rows in {label}; skipping.")
            continue
        X_aux = get_X(transform_features(df_aux, maps))
        y_aux = le.transform(df_aux[TARGET].astype(str))
        r = evaluate_model(best_model.__class__.__name__, best_model, X_aux, y_aux, le, label)
        print(f"  [{label}] macro F1 = {r['macro_f1']:.4f}")


# ==============================================================================
# PHASE 9 - ARTIFACT SAVING & STANDALONE PREDICTION API
# ==============================================================================

def save_artifacts(best_model, maps: dict, le: LabelEncoder, best_name: str) -> dict:
    joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")
    joblib.dump(maps,       OUTPUT_DIR / "feature_maps.joblib")
    joblib.dump(le,         OUTPUT_DIR / "label_encoder.joblib")
    meta = {
        "model_name":     best_name,
        "feature_names":  FEATURE_COLS,
        "classes":        le.classes_.tolist(),
        "class_to_index": {c: int(i) for i, c in enumerate(le.classes_)},
    }
    with open(OUTPUT_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Artifacts saved to {OUTPUT_DIR.resolve()}")
    return meta


def load_artifacts(output_dir: Path = None):
    """
    Load trained model and metadata independently of current working directory.
    Raises FileNotFoundError if artifacts are missing.
    """
    dir_path = Path(output_dir) if output_dir else OUTPUT_DIR
    model_path = dir_path / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Artifacts not found in {dir_path.resolve()}. "
            "Run main() in run.py first to train and serialize the model."
        )
    model = joblib.load(dir_path / "best_model.joblib")
    maps  = joblib.load(dir_path / "feature_maps.joblib")
    le    = joblib.load(dir_path / "label_encoder.joblib")
    with open(dir_path / "model_meta.json") as f:
        meta = json.load(f)
    return model, maps, le, meta


def predict_single(row: dict, model=None, maps=None, le=None, meta=None) -> dict:
    """
    Predict anomaly class for a single log record (dict) safely.
    Returns {"label": str, "probability": float, "all_probs": {class: prob}}
    """
    if model is None:
        model, maps, le, meta = load_artifacts()
    df_single = pd.DataFrame([row])
    df_t = transform_features(df_single, maps)
    X    = get_X(df_t)

    pred_enc   = int(model.predict(X)[0])
    pred_label = le.inverse_transform([pred_enc])[0]

    if hasattr(model, "predict_proba"):
        probs     = model.predict_proba(X)[0]
        all_probs = {cls: round(float(p), 4) for cls, p in zip(le.classes_, probs)}
        pred_prob = round(float(probs[pred_enc]), 4) if pred_enc < len(probs) else None
    else:
        all_probs = {}
        pred_prob = None

    return {
        "label": pred_label,
        "probability": pred_prob,
        "is_anomaly": pred_label != "normal",
        "all_probs": all_probs,
    }


def predict_batch(csv_or_df, output_path: str = None,
                  model=None, maps=None, le=None, meta=None) -> pd.DataFrame:
    """
    Predict anomaly class for a batch DataFrame or CSV path.
    Adds predicted_label, is_anomaly, and prob_{class} columns.
    """
    if model is None:
        model, maps, le, meta = load_artifacts()

    if isinstance(csv_or_df, (str, Path)):
        df = pd.read_csv(csv_or_df)
    elif isinstance(csv_or_df, pd.DataFrame):
        df = csv_or_df.copy()
    else:
        raise ValueError("csv_or_df must be a filepath or pandas DataFrame")

    if df.empty:
        df["predicted_label"] = []
        df["is_anomaly"] = []
        return df

    X = get_X(transform_features(df, maps))
    preds_enc = model.predict(X)
    df["predicted_label"] = le.inverse_transform(preds_enc)
    df["is_anomaly"] = df["predicted_label"] != "normal"

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        for i, cls in enumerate(le.classes_):
            df[f"prob_{cls}"] = probs[:, i].round(4)

    if output_path:
        df.to_csv(output_path, index=False)
        log.info(f"Batch predictions saved -> {output_path}")

    return df


def get_feature_contributions(row: dict, model=None, maps=None, le=None, meta=None) -> dict:
    """Explainability: top feature importances and values for a single record."""
    if model is None:
        model, maps, le, meta = load_artifacts()
    if not hasattr(model, "feature_importances_"):
        return {}
    X = get_X(transform_features(pd.DataFrame([row]), maps))
    contribs = {
        feat: {"importance": round(float(imp), 5), "value": round(float(val), 4)}
        for feat, imp, val in zip(FEATURE_COLS, model.feature_importances_, X[0])
    }
    return dict(sorted(contribs.items(), key=lambda x: -x[1]["importance"]))


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================

def main():
    print("\n" + "#" * 60)
    print("  LINUX AUTH LOG ANOMALY CLASSIFICATION PIPELINE")
    print("  XGBoost available:", XGBOOST_AVAILABLE)
    print("  Outputs directory :", OUTPUT_DIR.resolve())
    print("#" * 60)

    # Phase 1
    log.info("=== PHASE 1: Data loading & audit ===")
    df = load_and_audit(MAIN_DATA, "main training dataset")

    # Phase 2
    log.info("=== PHASE 2: EDA ===")
    run_eda(df)

    # Phase 4 (Chronological split before feature engineering to prevent leakage)
    log.info("=== PHASE 4: Chronological split ===")
    df_train_raw, df_val_raw, df_test_raw = chronological_split(df)

    # Phase 3
    log.info("=== PHASE 3: Feature engineering ===")
    maps    = fit_feature_maps(df_train_raw)
    X_train = get_X(transform_features(df_train_raw, maps))
    X_val   = get_X(transform_features(df_val_raw,   maps))
    X_test  = get_X(transform_features(df_test_raw,  maps))

    le      = LabelEncoder()
    y_train = le.fit_transform(df_train_raw[TARGET].astype(str))
    y_val   = le.transform(df_val_raw[TARGET].astype(str))
    y_test  = le.transform(df_test_raw[TARGET].astype(str))

    print(f"\nFeatures ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Classes : {le.classes_.tolist()}")
    print(f"Shapes  -> train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    cw_int, cw_str = build_class_weight_dicts(y_train, le.classes_)
    print(f"\nClass weights (int-keyed): {cw_int}")

    # Phase 5
    log.info("=== PHASE 5: Model training ===")
    trained_models = train_all_models(X_train, y_train, le, cw_int, cw_str)

    # Phase 6
    log.info("=== PHASE 6: Evaluation ===")
    results = evaluate_all(trained_models, X_val, y_val, X_test, y_test, le)

    # Phase 7
    log.info("=== PHASE 7: Model selection ===")
    best_name, best_model = select_best_model(results, trained_models)

    # Phase 8
    log.info("=== PHASE 8: Feature importance & robustness ===")
    for name, model in trained_models.items():
        show_feature_importance(model, name)
    robustness_check(best_model, maps, le)

    # Phase 9
    log.info("=== PHASE 9: Saving artifacts ===")
    meta = save_artifacts(best_model, maps, le, best_name)

    # Demo inference on unlabeled evaluation dataset
    if INFERENCE_DATA.exists():
        log.info("DEMO: Batch inference on unlabeled dataset ...")
        demo_out = str(OUTPUT_DIR / "inference_predictions.csv")
        df_infer = predict_batch(
            INFERENCE_DATA, output_path=demo_out,
            model=best_model, maps=maps, le=le, meta=meta
        )
        print("\nInference summary (unlabeled dataset):")
        print(df_infer["predicted_label"].value_counts())

    print("\n" + "#" * 60)
    print(f"  PIPELINE COMPLETE  ->  artifacts saved to: {OUTPUT_DIR.resolve()}")
    print("#" * 60)


if __name__ == "__main__":
    main()