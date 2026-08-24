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


# ========================================================================
# TAB 2: DATASET EXPLORATION
# ========================================================================
with tabs[1]:
    st.header("Dataset Exploration")

    if eval_data and "dataset_info" in eval_data:
        ds_info = eval_data["dataset_info"]

        # Class distribution
        st.subheader("Training Class Distribution")
        train_dist = ds_info.get("train_class_dist", {})
        if train_dist:
            dist_df = pd.DataFrame([
                {"Class": cls, "Count": cnt, "Percentage": f"{cnt / sum(train_dist.values()) * 100:.2f}%"}
                for cls, cnt in sorted(train_dist.items(), key=lambda x: -x[1])
            ])
            col_chart, col_table = st.columns([3, 2])
            with col_chart:
                import plotly.express as px
                fig = px.bar(
                    dist_df, x="Class", y="Count", color="Class",
                    title="Class Distribution in Training Set",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            with col_table:
                st.dataframe(dist_df, use_container_width=True, hide_index=True)

            # Normal vs anomaly
            normal_n = train_dist.get("normal", 0)
            anomaly_n = sum(v for k, v in train_dist.items() if k != "normal")
            st.subheader("Normal vs Anomaly Split")
            na_col1, na_col2 = st.columns(2)
            na_col1.metric("Normal Events", f"{normal_n:,}")
            na_col2.metric("Anomaly Events", f"{anomaly_n:,}", delta=f"{anomaly_n/(normal_n+anomaly_n)*100:.2f}%")

        # Dataset sample viewer
        st.subheader("Dataset Sample Viewer")
        ds_choice = st.selectbox("Select Dataset", [
            "Main (labeled)",
            "Balanced (auxiliary)",
            "Unbalanced (auxiliary)",
            "Inference (unlabeled)",
        ])
        ds_map = {
            "Main (labeled)": PROJECT_ROOT / "linux_auth_logs_labeled.csv",
            "Balanced (auxiliary)": PROJECT_ROOT / "linux_auth_logs_full(balanced).csv",
            "Unbalanced (auxiliary)": PROJECT_ROOT / "linux_auth_logs_full(new_unbalanced).csv",
            "Inference (unlabeled)": PROJECT_ROOT / "linux_auth_logs_multiple_anomalies.csv",
        }
        sample_df = load_dataset_sample(str(ds_map[ds_choice]), nrows=500)
        if sample_df is not None:
            st.caption(f"Showing first 500 rows of {ds_choice}")
            st.dataframe(sample_df, use_container_width=True, height=300)
        else:
            st.warning(f"Dataset not found: {ds_map[ds_choice]}")

    else:
        st.warning("Evaluation results not found. Run evaluation script first.")


# ========================================================================
# TAB 3: MODEL COMPARISON
# ========================================================================
with tabs[2]:
    st.header("Model Comparison — Test Set Performance")

    comp_df = make_comparison_df()
    if comp_df is not None:
        # Highlight the selected model
        def highlight_best(row):
            if row["Model"] == model_name:
                return ["background-color: #d4edda; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            comp_df.style.apply(highlight_best, axis=1).format({
                "Accuracy": "{:.4f}", "Macro Precision": "{:.4f}",
                "Macro Recall": "{:.4f}", "Macro F1": "{:.4f}",
                "Weighted F1": "{:.4f}", "ROC-AUC": "{:.4f}",
            }),
            use_container_width=True, hide_index=True,
        )

        st.subheader("Metric Comparison Chart")
        import plotly.graph_objects as go
        metrics_to_plot = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
        fig = go.Figure()
        for _, row in comp_df.iterrows():
            fig.add_trace(go.Bar(
                name=row["Model"],
                x=metrics_to_plot,
                y=[row[m] for m in metrics_to_plot],
            ))
        fig.update_layout(barmode="group", height=450, yaxis_range=[0, 1.05],
                         title="Model Performance Comparison (Test Set)")
        st.plotly_chart(fig, use_container_width=True)

        # Per-class breakdown
        st.subheader("Per-Class Performance (Test Set)")
        model_choice = st.selectbox("Select model for per-class view", comp_df["Model"].tolist(), key="pc_model")
        test_r = next((r for r in eval_data["evaluation_results"]
                       if r["model"] == model_choice and r["split"] == "test"), None)
        if test_r and "per_class" in test_r:
            pc_rows = []
            for cls, metrics in test_r["per_class"].items():
                pc_rows.append({
                    "Class": cls,
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-Score": metrics["f1"],
                    "Support": metrics["support"],
                    "False Positives": metrics["fp"],
                    "False Negatives": metrics["fn"],
                })
            st.dataframe(pd.DataFrame(pc_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No evaluation results found.")


# ========================================================================
# TAB 4: CONFUSION MATRICES
# ========================================================================
with tabs[3]:
    st.header("Confusion Matrices")

    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        cm_model = st.selectbox("Model", list(set(
            r["model"] for r in eval_data["evaluation_results"]
        )) if eval_data else [], key="cm_model")
    with cm_col2:
        cm_split = st.selectbox("Split", ["val", "test"], key="cm_split")

    if eval_data and cm_model:
        # Try loading the existing PNG first
        png_path = OUTPUT_DIR / f"cm_{cm_model}_{cm_split}.png"
        if png_path.exists():
            st.image(str(png_path), caption=f"Confusion Matrix — {cm_model} ({cm_split})", width=700)
        else:
            st.info(f"No pre-generated image found at {png_path.name}")

        # Also show dynamic table from evaluation results
        r = next((r for r in eval_data["evaluation_results"]
                  if r["model"] == cm_model and r["split"] == cm_split), None)
        if r and "confusion_matrix" in r:
            st.subheader("Confusion Matrix (Numeric)")
            cm_array = np.array(r["confusion_matrix"])
            cm_labels = r.get("cm_labels", classes)
            cm_df = pd.DataFrame(cm_array, index=cm_labels, columns=cm_labels)
            cm_df.index.name = "True \\ Predicted"
            st.dataframe(cm_df, use_container_width=True)

            # Plotly heatmap
            import plotly.figure_factory as ff
            fig = ff.create_annotated_heatmap(
                z=cm_array, x=cm_labels, y=cm_labels,
                colorscale="Blues", showscale=True,
            )
            fig.update_layout(
                title=f"Confusion Matrix — {cm_model} ({cm_split})",
                xaxis_title="Predicted", yaxis_title="True",
                height=500,
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)


# ========================================================================
# TAB 5: FEATURE IMPORTANCE
# ========================================================================
with tabs[4]:
    st.header("Feature Importance Analysis")
    st.caption("⚠️ Feature importance indicates model contribution, NOT causation.")

    fi_models = []
    if eval_data:
        fi_models = list(set(r["model"] for r in eval_data["evaluation_results"]))
    fi_model = st.selectbox("Select Model", fi_models, key="fi_model")

    if fi_model:
        # Try loading existing PNG
        fi_png = OUTPUT_DIR / f"feature_importance_{fi_model}.png"
        if fi_png.exists():
            st.image(str(fi_png), caption=f"Feature Importance — {fi_model}", width=800)

        # Also try to compute from loaded model if it's the active model
        if fi_model == model_name and hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
            import plotly.express as px
            fig = px.bar(
                x=fi.values, y=fi.index, orientation="h",
                title=f"Feature Importances — {fi_model} (Interactive)",
                labels={"x": "Importance", "y": "Feature"},
                color=fi.values, color_continuous_scale="Viridis",
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        elif fi_model != model_name:
            st.info(f"Interactive chart only available for the active model ({model_name}). Showing saved image above.")


# ========================================================================
# TAB 6: ROBUSTNESS / GENERALIZATION
# ========================================================================
with tabs[5]:
    st.header("Robustness & Generalization Analysis")
    st.markdown(
        "Compares model performance on the **main test set** versus "
        "**auxiliary datasets** with different class distributions."
    )

    if eval_data and eval_data.get("robustness_results"):
        rob_results = eval_data["robustness_results"]
        test_results = [r for r in eval_data["evaluation_results"] if r["split"] == "test"]

        # Build comparison table
        rob_rows = []
        for r in test_results:
            rob_rows.append({
                "Model": r["model"], "Dataset": "Main Test",
                "Accuracy": round(r["accuracy"], 4),
                "Macro Precision": round(r["macro_precision"], 4),
                "Macro Recall": round(r["macro_recall"], 4),
                "Macro F1": round(r["macro_f1"], 4),
            })
        for r in rob_results:
            rob_rows.append({
                "Model": r["model"], "Dataset": r["split"].title(),
                "Accuracy": round(r["accuracy"], 4),
                "Macro Precision": round(r["macro_precision"], 4),
                "Macro Recall": round(r["macro_recall"], 4),
                "Macro F1": round(r["macro_f1"], 4),
            })

        rob_df = pd.DataFrame(rob_rows)
        st.dataframe(rob_df, use_container_width=True, hide_index=True)

        # Chart: Macro F1 comparison across datasets
        st.subheader("Macro F1 Across Datasets")
        import plotly.express as px
        fig = px.bar(
            rob_df, x="Model", y="Macro F1", color="Dataset", barmode="group",
            title="Macro F1: Main Test vs Auxiliary Datasets",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig.update_layout(height=450, yaxis_range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

        # Performance drop warnings
        st.subheader("Generalization Warnings")
        for m_name in set(r["model"] for r in rob_results):
            test_f1 = next((r["macro_f1"] for r in test_results if r["model"] == m_name), None)
            if test_f1 is None:
                continue
            for r in rob_results:
                if r["model"] == m_name:
                    drop = test_f1 - r["macro_f1"]
                    if drop > 0.15:
                        st.error(f"🔴 **{m_name}** drops by {drop:.4f} on {r['split']} dataset (F1: {test_f1:.4f} → {r['macro_f1']:.4f})")
                    elif drop > 0.05:
                        st.warning(f"🟡 **{m_name}** drops by {drop:.4f} on {r['split']} dataset (F1: {test_f1:.4f} → {r['macro_f1']:.4f})")
                    else:
                        st.success(f"🟢 **{m_name}** stable on {r['split']} dataset (F1: {test_f1:.4f} → {r['macro_f1']:.4f})")

        # Overfitting analysis
        st.subheader("Overfitting Analysis (Train → Val → Test)")
        of_data = eval_data.get("overfitting_analysis", {})
        if of_data:
            of_rows = []
            for m, info in of_data.items():
                of_rows.append({
                    "Model": m,
                    "Train F1": info["train_f1"],
                    "Val F1": info["val_f1"],
                    "Test F1": info["test_f1"],
                    "Gap (Train→Test)": info["gap_train_test"],
                    "Verdict": info["verdict"],
                })
            of_df = pd.DataFrame(of_rows)

            def color_verdict(val):
                if val == "OVERFITTING":
                    return "background-color: #f8d7da; color: #721c24;"
                elif val == "ACCEPTABLE":
                    return "background-color: #fff3cd; color: #856404;"
                else:
                    return "background-color: #d4edda; color: #155724;"

            st.dataframe(
                of_df.style.applymap(color_verdict, subset=["Verdict"]),
                use_container_width=True, hide_index=True,
            )

        # Schema mismatch warning
        st.subheader("Known Data Quality Issues")
        st.warning(
            "**BALANCED dataset** is missing the `privilege_escalation` class entirely. "
            "Only 4 of 5 classes are present, and 'port_scan' dominates at 40.92%."
        )
        st.warning(
            "**UNBALANCED dataset** has 12,500 null `port` values, which default to 0.0 "
            "during preprocessing."
        )
    else:
        st.info("No robustness results found. Run the evaluation script to generate results.")


# ========================================================================
# TAB 7: BATCH INFERENCE
# ========================================================================
with tabs[6]:
    st.header("Batch Inference — Upload CSV")
    st.markdown("Upload a CSV of authentication logs for batch anomaly classification.")
    st.caption(
        "Expected columns: `timestamp`, `source_ip`, `server`, `username`, `service`, "
        "`attempts`, `status`, `port`, `protocol`, `comment`. "
        "Missing columns are handled gracefully with safe defaults."
    )

    uploaded = st.file_uploader("📂 Upload Authentication Log CSV", type=["csv"], key="batch_upload")

    if uploaded is not None:
        try:
            # Validate file
            content = uploaded.read()
            if not content.strip():
                st.error("Uploaded CSV file is empty.")
                st.stop()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            with st.spinner("🔍 Analyzing log patterns and detecting anomalies..."):
                df_pred = predict_batch(tmp_path, model=model, maps=maps, le=le, meta=meta)

            os.unlink(tmp_path)

            total = len(df_pred)
            if total == 0:
                st.warning("CSV contained no data rows.")
            else:
                anomalies = df_pred[df_pred["predicted_label"] != "normal"]
                anomaly_count = len(anomalies)
                normal_count = total - anomaly_count
                anomaly_rate = anomaly_count / total * 100

                # KPIs
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Records", f"{total:,}")
                c2.metric("Normal Events", f"{normal_count:,}")
                c3.metric("Anomalies Detected", f"{anomaly_count:,}",
                         delta=f"{anomaly_rate:.2f}% rate", delta_color="inverse")
                c4.metric("Model", model_name)

                st.divider()

                # Class breakdown
                st.subheader("Classification Breakdown")
                summary = df_pred["predicted_label"].value_counts().reset_index()
                summary.columns = ["Class", "Count"]
                summary["Percentage"] = (summary["Count"] / total * 100).round(2)

                ch_col, tb_col = st.columns([3, 2])
                with ch_col:
                    import plotly.express as px
                    fig = px.pie(summary, values="Count", names="Class",
                                title="Prediction Distribution",
                                color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(fig, use_container_width=True)
                with tb_col:
                    st.dataframe(summary, use_container_width=True, hide_index=True)

                # Table view
                st.subheader("Prediction Results")
                filter_opt = st.selectbox("Filter:", ["All Records", "Anomalies Only"] + classes, key="batch_filter")
                if filter_opt == "Anomalies Only":
                    show_df = anomalies
                elif filter_opt != "All Records":
                    show_df = df_pred[df_pred["predicted_label"] == filter_opt]
                else:
                    show_df = df_pred

                show_cols = [c for c in show_df.columns if not c.startswith("prob_")]

                def highlight_anomalies(row):
                    if row.get("predicted_label", "normal") != "normal":
                        return ["background-color: #ffcccc; color: #900C3F;"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    show_df[show_cols].head(500).style.apply(highlight_anomalies, axis=1),
                    use_container_width=True, height=400,
                )

                # Download
                st.download_button(
                    "📥 Download Full Predictions CSV",
                    data=df_pred.to_csv(index=False),
                    file_name="auth_anomaly_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


# ========================================================================
# TAB 8: SINGLE EVENT PREDICTION
# ========================================================================
with tabs[7]:
    st.header("Single Event Prediction")
    st.markdown("Enter a single authentication log event for real-time classification.")

    with st.form("single_pred_form"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            inp_timestamp = st.text_input("Timestamp", value="2024-03-28T19:34:14", help="ISO-8601 format")
            inp_source_ip = st.text_input("Source IP", value="195.241.151.7")
            inp_server = st.text_input("Server", value="srv-tok-03")
            inp_username = st.text_input("Username", value="admin")
        with fc2:
            inp_service = st.selectbox("Service", ["ssh", "sshd", "sudo", "su", "login", "cron"])
            inp_status = st.selectbox("Status", ["Failed", "Success"])
            inp_protocol = st.selectbox("Protocol", ["SSH", "SSH2", "RDP", "TELNET", "unknown"])
        with fc3:
            inp_attempts = st.number_input("Attempts", min_value=1, max_value=10000, value=1)
            inp_port = st.number_input("Port", min_value=0, max_value=65535, value=22)
            inp_comment = st.text_area("Comment", value="", height=68)

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        row = {
            "timestamp": inp_timestamp,
            "source_ip": inp_source_ip,
            "server": inp_server,
            "username": inp_username,
            "service": inp_service,
            "attempts": inp_attempts,
            "status": inp_status,
            "port": inp_port,
            "protocol": inp_protocol,
            "comment": inp_comment,
        }
        try:
            result = predict_single(row, model=model, maps=maps, le=le, meta=meta)

            # Display result
            is_anomaly = result["is_anomaly"]
            label = result["label"]
            prob = result.get("probability")

            if is_anomaly:
                st.error(f"🚨 **ANOMALY DETECTED: `{label}`**")
            else:
                st.success(f"✅ **Normal activity: `{label}`**")

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Predicted Class", label)
            rc2.metric("Confidence", safe_metric(prob) if prob else "N/A")
            rc3.metric("Anomaly", "YES" if is_anomaly else "NO")

            # Probability breakdown
            if result.get("all_probs"):
                st.subheader("Class Probability Distribution")
                prob_df = pd.DataFrame([
                    {"Class": cls, "Probability": p}
                    for cls, p in sorted(result["all_probs"].items(), key=lambda x: -x[1])
                ])
                import plotly.express as px
                fig = px.bar(prob_df, x="Class", y="Probability",
                            color="Probability", color_continuous_scale="RdYlGn_r",
                            title="Prediction Probabilities")
                fig.update_layout(height=350, yaxis_range=[0, 1.05])
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


# ========================================================================
# TAB 9: ANOMALY EXPLANATION
# ========================================================================
with tabs[8]:
    st.header("Anomaly Explanation — Feature Contributions")
    st.caption(
        "⚠️ These are **model-level feature importances**, NOT causal explanations. "
        "They indicate which features the model relied on most for this specific prediction."
    )

    if not hasattr(model, "feature_importances_"):
        st.warning(
            f"The current model ({model_name}) does not support `feature_importances_`. "
            "Explanation is not available for this model type."
        )
    else:
        with st.form("explain_form"):
            ec1, ec2 = st.columns(2)
            with ec1:
                ex_source_ip = st.text_input("Source IP", value="195.241.151.7", key="ex_ip")
                ex_username = st.text_input("Username", value="root", key="ex_user")
                ex_service = st.selectbox("Service", ["ssh", "sshd", "sudo", "su", "login", "cron"], key="ex_svc")
                ex_status = st.selectbox("Status", ["Failed", "Success"], key="ex_status")
            with ec2:
                ex_port = st.number_input("Port", min_value=0, max_value=65535, value=22, key="ex_port")
                ex_attempts = st.number_input("Attempts", min_value=1, max_value=10000, value=5, key="ex_att")
                ex_timestamp = st.text_input("Timestamp", value="2024-03-28T03:14:00", key="ex_ts")
                ex_protocol = st.selectbox("Protocol", ["SSH", "SSH2", "RDP", "TELNET", "unknown"], key="ex_proto")
            ex_submitted = st.form_submit_button("🔍 Explain", use_container_width=True)

        if ex_submitted:
            row = {
                "timestamp": ex_timestamp, "source_ip": ex_source_ip,
                "username": ex_username, "service": ex_service,
                "status": ex_status, "port": ex_port,
                "attempts": ex_attempts, "protocol": ex_protocol,
            }
            try:
                # Get prediction first
                pred = predict_single(row, model=model, maps=maps, le=le, meta=meta)
                label = pred["label"]
                is_anom = pred["is_anomaly"]

                if is_anom:
                    st.error(f"🚨 Predicted: **`{label}`** (Anomaly)")
                else:
                    st.success(f"✅ Predicted: **`{label}`** (Normal)")

                contribs = get_feature_contributions(row, model=model, maps=maps, le=le, meta=meta)
                if contribs:
                    contrib_df = pd.DataFrame([
                        {"Feature": feat, "Importance": info["importance"], "Value": info["value"]}
                        for feat, info in contribs.items()
                    ])

                    st.subheader("Top Feature Contributions")
                    import plotly.express as px
                    top_n = contrib_df.head(10)
                    fig = px.bar(
                        top_n, x="Importance", y="Feature", orientation="h",
                        title=f"Top 10 Feature Contributions for '{label}' Prediction",
                        color="Importance", color_continuous_scale="Viridis",
                    )
                    fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Full Feature Breakdown")
                    st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No feature contributions available for this model.")

            except Exception as e:
                st.error(f"Explanation failed: {str(e)}")
