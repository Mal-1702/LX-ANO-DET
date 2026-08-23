# Linux Authentication Log Anomaly Detection System

A production-grade machine learning system designed to detect and classify security anomalies in Linux authentication logs (`/var/log/auth.log`). The system identifies threats including **Brute Force Attacks**, **Port Scans**, **Geographic Anomalies**, and **Privilege Escalations**.

---

## 🚀 Key Features

- **Multi-Class Threat Detection**: Identifies `normal`, `brute_force`, `port_scan`, `geo_anomaly`, and `privilege_escalation` events.
- **Leakage-Safe Feature Engineering**: Chronological splitting and strictly train-fitted frequency and categorical encoding to prevent lookahead bias.
- **Ensemble & Tree Models**: Decision Trees, Random Forests, Gradient Boosting, and XGBoost with automated hyperparameter tuning and class weighting.
- **Unified Evaluation Suite**: Macro/weighted F1, ROC-AUC (one-vs-rest), classification reports, and confusion matrix visualizations.
- **Explainable AI**: Per-prediction feature contribution analysis.
- **Production APIs & UI**:
  - **FastAPI REST API**: Endpoints for single event prediction, batch CSV processing, explainability, and health checks.
  - **Streamlit Web Dashboard**: Interactive analytics, KPI metrics, anomaly inspection, and explainability visualizations.
- **Automated Test Suite**: Pytest test suite validating 14 edge cases, data sanitization, and API endpoints.

---

## 📦 Project Structure

```text
├── .gitignore
├── requirements.txt
├── run.py                 # Core ML training, evaluation, and inference pipeline
├── tests/
│   └── test_pipeline.py   # Comprehensive production test suite
└── outputs/
    ├── fastapi_app.py     # FastAPI REST API service
    ├── streamlit_app.py   # Streamlit web dashboard
    ├── model_meta.json    # Model metadata and class mappings
    └── *.png              # EDA, confusion matrices, and feature importance charts
```

---

## 🛠️ Installation & Setup

1. **Clone repository and set up virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Pipeline**:
   ```bash
   python run.py
   ```

3. **Run Tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Launch FastAPI Service**:
   ```bash
   uvicorn outputs.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run outputs/streamlit_app.py
   ```
