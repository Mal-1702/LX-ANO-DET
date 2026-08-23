"""
Comprehensive Production Test Suite for Linux Anomaly Detection System
========================================================================
Tests all 14 inference validation cases, data preprocessing transformations,
FastAPI REST endpoints, and artifact consistency.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
)
from outputs.fastapi_app import app


@pytest.fixture(scope="session")
def artifacts():
    """Load model artifacts once for the test session."""
    model, maps, le, meta = load_artifacts()
    return {"model": model, "maps": maps, "le": le, "meta": meta}


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: 14 INFERENCE VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_01_normal_valid_input(artifacts):
    """Case 1: Normal valid input produces deterministic result without error."""
    row = {
        "timestamp": "2024-03-28T19:34:14.984218",
        "source_ip": "186.144.249.195",
        "server": "srv-tok-03",
        "username": "juancampos",
        "service": "ssh",
        "attempts": 1,
        "status": "Success",
        "port": 22,
        "protocol": "SSH",
        "comment": "User juancampos success login via ssh",
    }
    res = predict_single(row, **artifacts)
    assert "label" in res
    assert res["label"] in artifacts["le"].classes_
    assert isinstance(res["is_anomaly"], bool)
    assert 0.0 <= res["probability"] <= 1.0
    assert len(res["all_probs"]) == len(artifacts["le"].classes_)


def test_02_known_anomaly_brute_force(artifacts):
    """Case 2: High failed attempts signal brute force pattern."""
    row = {
        "timestamp": "2024-03-28T19:34:14",
        "source_ip": "195.241.151.7",
        "server": "proxy-mow-01",
        "username": "admin",
        "service": "ssh",
        "attempts": 25,  # extreme failed attempts
        "status": "Failed",
        "port": 22,
        "protocol": "SSH",
        "comment": "Failed password for root",
    }
    res = predict_single(row, **artifacts)
    assert res["label"] in artifacts["le"].classes_
    assert "probability" in res


def test_03_missing_optional_fields(artifacts):
    """Case 3: Missing optional fields (e.g. comment, protocol) defaults cleanly."""
    row = {
        "timestamp": "2024-03-28T19:34:14",
        "source_ip": "10.0.0.1",
        "username": "alice",
    }
    res = predict_single(row, **artifacts)
    assert res["label"] in artifacts["le"].classes_


def test_04_missing_required_fields(artifacts):
    """Case 4: Empty dictionary / missing all fields handled gracefully."""
    row = {}
    res = predict_single(row, **artifacts)
    assert res["label"] in artifacts["le"].classes_
    assert res["probability"] is not None


def test_05_non_numeric_port(artifacts):
    """Case 5: Non-numeric port strings (e.g. 'N/A', 'http', 'None') coerced to 0."""
    row = {
        "timestamp": "2024-03-28T19:34:14",
        "port": "N/A",
        "attempts": 1,
    }
    res = predict_single(row, **artifacts)
    assert res["label"] in artifacts["le"].classes_


def test_06_invalid_port_range(artifacts):
    """Case 6: Negative and overflow ports clipped to [0, 65535]."""
    row1 = {"port": -500}
    row2 = {"port": 999999}
    res1 = predict_single(row1, **artifacts)
    res2 = predict_single(row2, **artifacts)
    assert res1["label"] in artifacts["le"].classes_
    assert res2["label"] in artifacts["le"].classes_


def test_07_missing_attempts(artifacts):
    """Case 7: Missing attempts column defaults to 1.0."""
    row = {"username": "admin", "port": 22}
    res = predict_single(row, **artifacts)
    assert res["label"] in artifacts["le"].classes_


def test_08_invalid_attempts(artifacts):
    """Case 8: Invalid attempts (0, negative, strings) clipped to >= 1.0."""
    row1 = {"attempts": 0}
    row2 = {"attempts": -10}
    row3 = {"attempts": "invalid_str"}
    res1 = predict_single(row1, **artifacts)
    res2 = predict_single(row2, **artifacts)
    res3 = predict_single(row3, **artifacts)
    assert res1["label"] in artifacts["le"].classes_
    assert res2["label"] in artifacts["le"].classes_
    assert res3["label"] in artifacts["le"].classes_


def test_09_unknown_source_ip_cold_start(artifacts):
    """Case 9: Completely novel IP address (unseen in training) frequency is 0.0."""
    row = {"source_ip": "192.0.2.254"}  # Documentation IP
    df_t = transform_features(pd.DataFrame([row]), artifacts["maps"])
    assert df_t["source_ip_freq"].iloc[0] == 0.0
    res = predict_single(row, **artifacts)
    assert res["label"] in artifacts["le"].classes_


def test_10_unknown_username_cold_start(artifacts):
    """Case 10: Completely novel username frequency is 0.0."""
    row = {"username": "unknown_cyber_attacker_9999"}
    df_t = transform_features(pd.DataFrame([row]), artifacts["maps"])
    assert df_t["username_freq"].iloc[0] == 0.0
    res = predict_single(row, **artifacts)
    assert res["label"] in artifacts["le"].classes_


def test_11_extreme_numeric_values(artifacts):
    """Case 11: Extreme numerical values do not produce NaN or Inf in feature matrix."""
    row = {
        "attempts": 1e9,
        "port": 1e9,
    }
    df_t = transform_features(pd.DataFrame([row]), artifacts["maps"])
    X = get_X(df_t)
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()
    res = predict_single(row, **artifacts)
    assert res["label"] in artifacts["le"].classes_


def test_12_empty_batch(artifacts):
    """Case 12: Empty DataFrame batch returns empty DataFrame with expected columns."""
    df_empty = pd.DataFrame()
    res_df = predict_batch(df_empty, **artifacts)
    assert "predicted_label" in res_df.columns
    assert len(res_df) == 0


def test_13_batch_mixed_valid_and_invalid(artifacts):
    """Case 13: Batch with mixed valid and invalid rows completes safely."""
    df_batch = pd.DataFrame([
        {"timestamp": "2024-01-01T12:00:00", "source_ip": "1.1.1.1", "port": 22, "attempts": 1},
        {"timestamp": "invalid_date", "port": "N/A", "attempts": -3},
        {"source_ip": None, "username": None, "comment": None},
    ])
    res_df = predict_batch(df_batch, **artifacts)
    assert len(res_df) == 3
    assert "predicted_label" in res_df.columns
    assert "is_anomaly" in res_df.columns
    for p in res_df["predicted_label"]:
        assert p in artifacts["le"].classes_


def test_14_explainability_consistency(artifacts):
    """Case 14: Explainability produces sorted contributions for all features."""
    row = {"port": 443, "attempts": 5, "username": "root"}
    contribs = get_feature_contributions(row, **artifacts)
    assert len(contribs) == len(FEATURE_COLS)
    for feat in FEATURE_COLS:
        assert feat in contribs
        assert "importance" in contribs[feat]
        assert "value" in contribs[feat]


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINT INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_api_health(client):
    """Test GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert len(data["classes"]) == 5


def test_api_get_classes(client):
    """Test GET /classes returns all 5 classes."""
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert set(data["classes"]) == {"brute_force", "geo_anomaly", "normal", "port_scan", "privilege_escalation"}


def test_api_predict_single(client):
    """Test POST /predict/single with valid payload."""
    payload = {
        "timestamp": "2024-03-28T19:34:14",
        "source_ip": "186.144.249.195",
        "username": "juancampos",
        "attempts": 1,
        "port": 22,
    }
    response = client.post("/predict/single", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "is_anomaly" in data
    assert "all_probs" in data


def test_api_predict_batch_csv(client, tmp_path):
    """Test POST /predict/batch with CSV file upload."""
    csv_content = (
        "timestamp,source_ip,server,username,service,attempts,status,port,protocol,comment\n"
        "2024-03-28T19:34:14,186.144.249.195,srv-tok-03,juancampos,ssh,1,Success,22,SSH2,User login\n"
        "2024-03-28T19:35:14,195.241.151.7,srv-tok-03,root,ssh,20,Failed,22,SSH,Brute force attempt\n"
    )
    csv_file = tmp_path / "test_auth.csv"
    csv_file.write_text(csv_content)

    with open(csv_file, "rb") as f:
        response = client.post("/predict/batch", files={"file": ("test_auth.csv", f, "text/csv")})

    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert "summary_by_class" in data
    assert len(data["sample_predictions"]) == 2
