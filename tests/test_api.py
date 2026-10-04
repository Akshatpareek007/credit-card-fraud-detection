"""
Tests for FastAPI Service Endpoints, Authentication, Validation, and Rate Limiting.
"""

import os
import tempfile
import pytest
from fastapi.testclient import TestClient

import src.api as api_module
from src.api import app, API_KEY, ADMIN_API_KEY, DEMO_MODE
from src.ledger import AuditLedger

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    # Set isolated database path for test runs
    test_db = str(tmp_path / "test_api_ledger.db")
    api_module.load_artifacts()
    api_module.ledger = AuditLedger(db_path=test_db)
    yield



def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "model_version" in data


def test_unauthenticated_predict_rejected():
    payload = {"Amount": 100.0}
    res = client.post("/predict", json=payload)
    assert res.status_code == 401


def test_predict_valid_request():
    payload = {"Amount": 150.0, "Hour": 14, "V14": -2.5, "V17": -1.8}
    headers = {"X-API-Key": API_KEY}
    res = client.post("/predict", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "fraud_score" in data
    assert "decision" in data
    assert "model_version" in data
    assert "threshold" in data


def test_predict_invalid_negative_amount():
    payload = {"Amount": -50.0}
    headers = {"X-API-Key": API_KEY}
    res = client.post("/predict", json=payload, headers=headers)
    assert res.status_code == 422


def test_predict_invalid_type_rejected():
    payload_invalid = {"Amount": "not-a-number"}
    headers = {"X-API-Key": API_KEY}
    res = client.post("/predict", json=payload_invalid, headers=headers)
    assert res.status_code == 422


def test_batch_oversized_rejected():
    batch = [{"Amount": 10.0}] * 501
    headers = {"X-API-Key": API_KEY}
    res = client.post("/predict-batch", json={"transactions": batch}, headers=headers)
    assert res.status_code in [413, 422]


def test_ledger_verify_and_entries():
    headers = {"X-API-Key": API_KEY}
    res_v = client.get("/ledger/verify", headers=headers)
    assert res_v.status_code == 200
    assert res_v.json()["is_valid"] is True

    res_e = client.get("/ledger/entries", headers=headers)
    assert res_e.status_code == 200
    assert "entries" in res_e.json()


def test_demo_tamper_simulation():
    headers = {"X-API-Key": API_KEY}
    tamper_payload = {
        "admin_key": ADMIN_API_KEY,
        "block_index": 0,
        "new_fraud_score": 0.9999
    }

    if DEMO_MODE:
        res = client.post("/demo/tamper", json=tamper_payload, headers=headers)
        assert res.status_code == 200
    else:
        res = client.post("/demo/tamper", json=tamper_payload, headers=headers)
        assert res.status_code == 403
