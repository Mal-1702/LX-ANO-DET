"""
Streamlit Dashboard for Linux Auth Log Anomaly Detection
=========================================================
Run with:
  streamlit run outputs/streamlit_app.py
or from outputs directory:
  streamlit run streamlit_app.py
"""

import os
import sys
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd

# Robust module path resolution: ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run import load_artifacts, predict_batch, get_feature_contributions

st.set_page_config(page_title="Linux Auth Anomaly Detector", layout="wide", page_icon="🛡️")

st.title("🛡️ Linux Authentication Log Anomaly Detector")
st.markdown("Automated detection of **Brute Force**, **Port Scans**, **Geo Anomalies**, and **Privilege Escalations** in Linux security logs.")

@st.cache_resource
def get_model():
    try:
        return load_artifacts()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        return None, None, None, {}

model, maps, le, meta = get_model()

if model is None:
    st.warning("⚠️ Model artifacts are not loaded. Please run `python run.py` first to train and serialize the model.")
    st.stop()

# Sidebar Information
with st.sidebar:
    st.header("⚙️ Model Info")
    st.info(f"**Model Name:** {meta.get('model_name', 'RandomForest')}")
    st.markdown(f"**Classes Recognized:**")
    for cls in meta.get("classes", []):
        st.write(f"- `{cls}`")
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.caption("Upload a `.csv` file containing authentication logs with fields such as `timestamp`, `source_ip`, `server`, `username`, `service`, `attempts`, `status`, `port`, `protocol`, `comment`.")

uploaded = st.file_uploader("📂 Upload Linux Auth Log CSV", type=["csv"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        with st.spinner("🔍 Analyzing log patterns and detecting anomalies..."):
            df_pred = predict_batch(tmp_path, model=model, maps=maps, le=le, meta=meta)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    total_records = len(df_pred)
    anomalies = df_pred[df_pred["predicted_label"] != "normal"]
    anomaly_count = len(anomalies)
    normal_count = total_records - anomaly_count
    anomaly_rate = (anomaly_count / total_records * 100) if total_records > 0 else 0

    # Summary KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Normal Events", f"{normal_count:,}")
    col3.metric("Anomalies Detected", f"{anomaly_count:,}", delta=f"{anomaly_rate:.2f}% rate", delta_color="inverse")
    col4.metric("Model Selected", meta.get("model_name", "RandomForest"))

    st.markdown("---")

    # Class Breakdown Chart & Table
    st.subheader("📊 Breakdown by Predicted Classification")
    summary = df_pred["predicted_label"].value_counts().rename_axis("Class").reset_index(name="Count")
    summary["Percentage"] = (summary["Count"] / total_records * 100).round(2)

    chart_col, table_col = st.columns([3, 2])
    with chart_col:
        st.bar_chart(summary.set_index("Class")["Count"])
    with table_col:
        st.dataframe(summary, use_container_width=True)

    # Detailed Table View
    st.subheader("🚨 Inspection View (Anomalies Highlighted)")
    
    filter_choice = st.selectbox("Filter Display:", ["All Records", "Anomalies Only"] + meta.get("classes", []))
    
    if filter_choice == "Anomalies Only":
        display_df = anomalies
    elif filter_choice != "All Records":
        display_df = df_pred[df_pred["predicted_label"] == filter_choice]
    else:
        display_df = df_pred

    def highlight_anomalies(row):
        color = "background-color: #ffcccc; color: #900C3F;" if row["predicted_label"] != "normal" else ""
        return [color] * len(row)

    show_cols = [c for c in display_df.columns if not c.startswith("prob_")]
    st.dataframe(display_df[show_cols].head(500).style.apply(highlight_anomalies, axis=1), use_container_width=True)

    # Download Button
    st.download_button(
        label="📥 Download Full Predictions CSV",
        data=df_pred.to_csv(index=False),
        file_name="auth_anomaly_predictions.csv",
        mime="text/csv"
    )

    # Explainability Section for Top Anomaly
    if not anomalies.empty:
        st.markdown("---")
        st.subheader("🔍 Explainable AI: Feature Contributions for Anomaly Record")
        sample_idx = st.slider("Select Anomaly Row Index for Explanation", 0, min(100, len(anomalies) - 1), 0)
        selected_row = anomalies.iloc[sample_idx].to_dict()

        st.write(f"**Target Class:** `{selected_row.get('predicted_label')}` | **User:** `{selected_row.get('username')}` | **IP:** `{selected_row.get('source_ip')}`")
        contribs = get_feature_contributions(selected_row, model=model, maps=maps, le=le, meta=meta)
        if contribs:
            top_features = {k: v["importance"] for k, v in list(contribs.items())[:10]}
            st.bar_chart(top_features)
