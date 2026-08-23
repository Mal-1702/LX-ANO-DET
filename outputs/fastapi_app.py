"""
FastAPI Production Service for Linux Auth Log Anomaly Detection
===============================================================
Run with:
  uvicorn outputs.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
or from outputs directory:
  uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd

# Robust module path resolution: ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run import (
    load_artifacts,
    predict_single,
    predict_batch,
    get_feature_contributions,
    FEATURE_COLS,
)

app = FastAPI(
    title="Linux Auth Log Anomaly Detection API",
    description="Production-ready REST API for detecting brute-force, port-scans, geo-anomalies, and privilege escalation in Linux auth logs.",
    version="1.0.0",
)

# Enable CORS for secure dashboard/frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML artifacts safely on startup
try:
    model, maps, le, meta = load_artifacts()
    MODEL_LOADED = True
except Exception as e:
    model, maps, le, meta = None, None, None, {}
    MODEL_LOADED = False
    print(f"Warning: Artifacts could not be loaded on startup: {e}")


# Pydantic Schemas for Request/Response Validation
class LogRecord(BaseModel):
    timestamp: Optional[str] = Field(None, description="ISO-8601 or standard datetime string", example="2024-03-28T19:34:14")
    source_ip: Optional[str] = Field(None, description="Source IPv4 address", example="195.241.151.7")
    server: Optional[str] = Field(None, description="Target server hostname", example="srv-tok-03")
    username: Optional[str] = Field(None, description="Authentication username", example="admin")
    service: Optional[str] = Field(None, description="Auth service (ssh, sudo, su, login, cron)", example="sshd")
    attempts: Optional[float] = Field(1.0, ge=1.0, description="Number of consecutive login attempts", example=3)
    status: Optional[str] = Field("Failed", description="Auth status (Failed, Success)", example="Failed")
    port: Optional[float] = Field(22.0, ge=0.0, le=65535.0, description="Network port number", example=22)
    protocol: Optional[str] = Field("SSH", description="Protocol (SSH, SSH2, RDP, TELNET, unknown)", example="SSH")
    comment: Optional[str] = Field("", description="Auth log message content", example="Failed password for invalid user admin")


class PredictionResponse(BaseModel):
    label: str
    is_anomaly: bool
    probability: Optional[float]
    all_probs: Dict[str, float]


@app.get("/health", summary="Service Health Check")
async def health_check():
    """Health and model metadata endpoint."""
    return {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "model_name": meta.get("model_name", "unknown"),
        "classes": meta.get("classes", []),
        "features": FEATURE_COLS,
    }


@app.get("/classes", summary="Get Target Classes")
async def get_classes():
    """Returns the list of anomaly classes recognized by the model."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return {"classes": le.classes_.tolist()}


@app.post("/predict/single", response_model=PredictionResponse, summary="Predict Single Log Event")
async def api_predict_single(record: LogRecord):
    """Predicts whether a single authentication log event is normal or an anomaly."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    try:
        row_dict = record.model_dump()
        result = predict_single(row_dict, model=model, maps=maps, le=le, meta=meta)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", summary="Predict Batch CSV File")
async def api_predict_batch(file: UploadFile = File(...)):
    """Uploads a CSV of auth logs and returns prediction summary + sample results."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV format.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            if not content.strip():
                raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")
            tmp.write(content)
            tmp_path = tmp.name

        df_pred = predict_batch(tmp_path, model=model, maps=maps, le=le, meta=meta)
        os.unlink(tmp_path)

        total = len(df_pred)
        anomaly_count = int((df_pred["predicted_label"] != "normal").sum())
        summary = df_pred["predicted_label"].value_counts().to_dict()

        return {
            "total_records": total,
            "anomaly_records": anomaly_count,
            "anomaly_rate_percent": round((anomaly_count / total * 100) if total > 0 else 0, 2),
            "summary_by_class": summary,
            "sample_predictions": df_pred.head(100).to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch processing failed: {str(e)}")


@app.post("/explain", summary="Explain Single Record Features")
async def api_explain(record: LogRecord):
    """Returns top feature contributions for explainable AI on an anomaly record."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    try:
        row_dict = record.model_dump()
        contribs = get_feature_contributions(row_dict, model=model, maps=maps, le=le, meta=meta)
        return {
            "feature_contributions": contribs,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explainability calculation error: {str(e)}")
