"""
Streamlit ML Research Dashboard — Linux Auth Log Anomaly Detection
===================================================================
Professional 10-tab research dashboard for validated ML system.
Loads pre-trained artifacts only. Does NOT retrain models.

Run with:
    streamlit run outputs/streamlit_app.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# PATH SETUP — resolve project root from this file's location
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run import (
    load_artifacts,
    predict_single,
    predict_batch,
    get_feature_contributions,
    transform_features,
    get_X,
    FEATURE_COLS,
    TARGET,
)

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Linux Auth Anomaly Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CACHED RESOURCE LOADING — loads model ONCE across all reruns
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    """Load trained model, feature maps, label encoder, and metadata."""
    try:
        model, maps, le, meta = load_artifacts()
        return model, maps, le, meta, None
    except Exception as e:
        return None, None, None, {}, str(e)


@st.cache_data
def load_evaluation_results():
    """Load pre-computed evaluation results JSON."""
    path = OUTPUT_DIR / "evaluation_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_dataset_sample(path_str, nrows=None):
    """Load a dataset with optional row limit."""
    try:
        p = Path(path_str)
        if not p.exists():
            return None
        return pd.read_csv(p, nrows=nrows)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------------------------
model, maps, le, meta, load_error = load_model_artifacts()
eval_data = load_evaluation_results()

if model is None:
    st.error(f"⚠️ Failed to load model artifacts: {load_error}")
    st.info("Run `python run.py` first to train and serialize the model.")
    st.stop()

classes = meta.get("classes", le.classes_.tolist() if le is not None else [])
model_name = meta.get("model_name", "Unknown")

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/linux--v1.png", width=64)
    st.title("🛡️ Anomaly Detector")
    st.caption("Linux Authentication Log Analysis System")
    st.divider()
    st.markdown(f"**Active Model:** `{model_name}`")
    st.markdown(f"**Features:** {len(FEATURE_COLS)}")
    st.markdown(f"**Classes:** {len(classes)}")
    for cls in classes:
        icon = "🟢" if cls == "normal" else "🔴"
        st.markdown(f"  {icon} `{cls}`")
    st.divider()
    st.caption("Built with scikit-learn, XGBoost, and Streamlit")

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def get_test_metrics(model_name_filter=None):
    """Extract test-split metrics from evaluation results."""
    if eval_data is None:
        return None
    results = eval_data.get("evaluation_results", [])
    test_results = [r for r in results if r["split"] == "test"]
    if model_name_filter:
        test_results = [r for r in test_results if r["model"] == model_name_filter]
    return test_results


def safe_metric(value, fmt=".4f"):
    """Format a metric safely, handling None."""
    if value is None:
        return "N/A"
    return f"{value:{fmt}}"


def make_comparison_df():
    """Build a model comparison DataFrame from evaluation results."""
    if eval_data is None:
        return None
    rows = []
    for r in eval_data.get("evaluation_results", []):
        if r["split"] == "test":
            rows.append({
                "Model": r["model"],
                "Accuracy": round(r["accuracy"], 4),
                "Macro Precision": round(r["macro_precision"], 4),
                "Macro Recall": round(r["macro_recall"], 4),
                "Macro F1": round(r["macro_f1"], 4),
                "Weighted F1": round(r["weighted_f1"], 4),
                "ROC-AUC": round(r["roc_auc"], 4) if r["roc_auc"] else None,
            })
    if not rows:
        return None
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "📊 Overview",
    "🗂️ Dataset Explorer",
    "🏆 Model Comparison",
    "📉 Confusion Matrices",
    "🎯 Feature Importance",
    "🔬 Robustness",
    "📁 Batch Inference",
    "🔍 Single Prediction",
    "💡 Explanation",
    "⚙️ System Info",
])

# ========================================================================
# TAB 1: OVERVIEW
# ========================================================================
with tabs[0]:
    st.header("Linux Authentication Log Anomaly Detector")
    st.markdown(
        "Enterprise-grade ML pipeline for detecting **brute force attacks**, "
        "**port scans**, **geographic anomalies**, and **privilege escalation** "
        "in Linux authentication logs (`/var/log/auth.log`)."
    )

    # KPI cards from evaluation results
    test_metrics = get_test_metrics(model_name)
    if test_metrics:
        m = test_metrics[0]
        ds_info = eval_data.get("dataset_info", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Training Records", f"{ds_info.get('train_size', 0):,}")
        col2.metric("Anomaly Classes", str(len(classes)))
        col3.metric("Best Model", model_name)
        col4.metric("Test Macro F1", safe_metric(m["macro_f1"]))

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Test Accuracy", safe_metric(m["accuracy"]))
        col6.metric("Test Macro Recall", safe_metric(m["macro_recall"]))
        col7.metric("Test Macro Precision", safe_metric(m["macro_precision"]))
        col8.metric("Validation Size", f"{ds_info.get('val_size', 0):,}")
    else:
        st.warning("No evaluation results found. Run the evaluation script to generate metrics.")

    st.divider()
    st.subheader("Pipeline Architecture")
    st.code(
        "RAW AUTH LOGS → Data Audit → Chronological Split → Feature Engineering\n"
        "→ Label Encoding → Class Weighting → Model Training (DT/RF/GB/XGB)\n"
        "→ Evaluation → Model Selection → Artifact Serialization → Inference API",
        language=None,
    )

    # Deployment readiness notice
    overfitting = eval_data.get("overfitting_analysis", {}) if eval_data else {}
    robustness = eval_data.get("robustness_results", []) if eval_data else []
    if robustness:
        worst_rob = min(r["macro_f1"] for r in robustness if r["model"] == model_name)
        if worst_rob < 0.50:
            st.error(
                f"⚠️ **DEPLOYMENT STATUS: DEMO / MVP ONLY** — "
                f"Auxiliary dataset robustness Macro F1 drops to {worst_rob:.4f}. "
                f"Significant generalization gap detected. See Robustness tab for details."
            )
        elif worst_rob < 0.70:
            st.warning(
                f"⚠️ Auxiliary dataset Macro F1 = {worst_rob:.4f}. "
                f"Moderate generalization concerns. See Robustness tab."
            )
