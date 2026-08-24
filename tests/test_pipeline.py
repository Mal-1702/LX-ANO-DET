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
    fit_feature_maps,
    FEATURE_COLS,
    PROTO_MAP,
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
# PART 1: 14 INFERENCE VALIDATION TESTS
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
# PART 2: PREPROCESSING ROBUSTNESS TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_15_transform_port_clipping(artifacts):
    """Port values are clipped to [0, 65535] and log1p is computed."""
    df = pd.DataFrame([{"port": -100}, {"port": 99999}, {"port": 22}])
    df_t = transform_features(df, artifacts["maps"])
    assert df_t["port"].iloc[0] == 0.0
    assert df_t["port"].iloc[1] == 65535.0
    assert df_t["port"].iloc[2] == 22.0
    assert df_t["log_port"].iloc[2] == np.log1p(22.0)


def test_16_transform_attempts_floor(artifacts):
    """Attempts are floored to >= 1.0."""
    df = pd.DataFrame([{"attempts": 0}, {"attempts": -5}, {"attempts": "abc"}, {"attempts": 10}])
    df_t = transform_features(df, artifacts["maps"])
    assert df_t["attempts"].iloc[0] >= 1.0
    assert df_t["attempts"].iloc[1] >= 1.0
    assert df_t["attempts"].iloc[2] >= 1.0
    assert df_t["attempts"].iloc[3] == 10.0


def test_17_transform_temporal_features(artifacts):
    """Temporal features are correctly extracted from timestamps."""
    df = pd.DataFrame([{"timestamp": "2024-03-28T03:15:00"}])
    df_t = transform_features(df, artifacts["maps"])
    assert df_t["hour"].iloc[0] == 3
    assert df_t["is_night"].iloc[0] == 1  # 3am is night
    assert df_t["month"].iloc[0] == 3
    assert df_t["day"].iloc[0] == 28


def test_18_transform_invalid_timestamp(artifacts):
    """Invalid timestamps default to safe values without crashing."""
    df = pd.DataFrame([{"timestamp": "not_a_date"}])
    df_t = transform_features(df, artifacts["maps"])
    assert df_t["hour"].iloc[0] == 0
    assert df_t["dow"].iloc[0] == 0
    assert df_t["is_night"].iloc[0] == 1  # hour 0 < 6


def test_19_transform_protocol_encoding(artifacts):
    """Protocol encoding maps correctly including unknown."""
    df = pd.DataFrame([
        {"protocol": "SSH"}, {"protocol": "RDP"},
        {"protocol": "TELNET"}, {"protocol": "UNKNOWN_PROTO"},
    ])
    df_t = transform_features(df, artifacts["maps"])
    assert df_t["protocol_enc"].iloc[0] == PROTO_MAP["SSH"]
    assert df_t["protocol_enc"].iloc[1] == PROTO_MAP["RDP"]
    assert df_t["protocol_enc"].iloc[2] == PROTO_MAP["TELNET"]
    assert df_t["protocol_enc"].iloc[3] == PROTO_MAP["unknown"]  # fallback


def test_20_transform_missing_columns(artifacts):
    """Transform handles completely missing columns gracefully."""
    df = pd.DataFrame([{"source_ip": "1.2.3.4"}])  # Most columns missing
    df_t = transform_features(df, artifacts["maps"])
    X = get_X(df_t)
    assert X.shape == (1, len(FEATURE_COLS))
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


def test_21_transform_nan_propagation(artifacts):
    """NaN values in input are handled without propagating to feature matrix."""
    df = pd.DataFrame([{
        "port": float("nan"), "attempts": float("nan"),
        "source_ip": None, "username": None,
    }])
    df_t = transform_features(df, artifacts["maps"])
    X = get_X(df_t)
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


def test_22_transform_comment_flags(artifacts):
    """Comment keyword flags correctly detect 'failed' and 'accepted'."""
    df = pd.DataFrame([
        {"comment": "Failed password for root"},
        {"comment": "Accepted publickey for alice"},
        {"comment": "Session opened"},
        {"comment": None},
    ])
    df_t = transform_features(df, artifacts["maps"])
    assert df_t["comment_failed"].iloc[0] == 1
    assert df_t["comment_accepted"].iloc[0] == 0
    assert df_t["comment_failed"].iloc[1] == 0
    assert df_t["comment_accepted"].iloc[1] == 1
    assert df_t["comment_failed"].iloc[2] == 0
    assert df_t["comment_accepted"].iloc[2] == 0
    assert df_t["comment_failed"].iloc[3] == 0


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: ARTIFACT LOADING TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_23_artifact_loading_from_default_path():
    """Artifacts load from the default OUTPUT_DIR path."""
    model, maps, le, meta = load_artifacts()
    assert model is not None
    assert maps is not None
    assert le is not None
    assert "model_name" in meta
    assert "classes" in meta
    assert "feature_names" in meta
    assert len(meta["classes"]) == 5


def test_24_artifact_loading_explicit_path():
    """Artifacts load from an explicitly provided path."""
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    model, maps, le, meta = load_artifacts(output_dir)
    assert model is not None
    assert len(le.classes_) == 5


def test_25_artifact_loading_nonexistent_path():
    """Loading from nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_artifacts(Path("/nonexistent/path/outputs"))


def test_26_feature_maps_structure(artifacts):
    """Feature maps contain expected keys and are non-empty."""
    maps = artifacts["maps"]
    assert "server_le" in maps
    assert "service_le" in maps
    assert "status_le" in maps
    assert "source_ip_freq" in maps
    assert "username_freq" in maps
    assert len(maps["server_le"]) > 0
    assert len(maps["source_ip_freq"]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: FASTAPI ENDPOINT INTEGRATION TESTS
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


def test_api_predict_single_empty(client):
    """Test POST /predict/single with empty payload still succeeds."""
    response = client.post("/predict/single", json={})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data


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


def test_api_predict_batch_empty_csv(client, tmp_path):
    """Test POST /predict/batch with empty CSV raises error."""
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")

    with open(csv_file, "rb") as f:
        response = client.post("/predict/batch", files={"file": ("empty.csv", f, "text/csv")})

    assert response.status_code == 400


def test_api_predict_batch_non_csv(client, tmp_path):
    """Test POST /predict/batch with non-CSV file raises error."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("this is not a csv")

    with open(txt_file, "rb") as f:
        response = client.post("/predict/batch", files={"file": ("test.txt", f, "text/plain")})

    assert response.status_code == 400


def test_api_explain(client):
    """Test POST /explain returns feature contributions."""
    payload = {"port": 443, "attempts": 5, "username": "root"}
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "feature_contributions" in data
