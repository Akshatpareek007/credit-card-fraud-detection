"""
FastAPI Service Module with SHAP Explanations, Pydantic Validation, Rate Limiting, API Key Authentication, and Audit Ledger Routing.
"""

import os
import json
import time
import io
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
import shap

from fastapi import FastAPI, HTTPException, Security, Depends, status, Request, UploadFile, File, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from src.features import FEATURE_NAMES
from src.ledger import AuditLedger, DEFAULT_DB_PATH

# Environment & Config
API_KEY = os.getenv("API_KEY", "sentinel-secret-key-123")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "admin-super-secret-key-456")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
RATE_LIMIT_PER_MINUTE = 100

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Global Loaded Artifacts
pipeline = None
calibrator = None
model = None
explainer = None
model_version = "2.0.0"
threshold = 0.9870
ledger = None

# Rate limiter memory
request_counts: Dict[str, List[float]] = {}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


app = FastAPI(
    title="Sentinel Fraud AI & Audit Ledger API",
    description="Production-grade API for real-time Credit Card Fraud Detection with SHAP explanations and Tamper-Evident Ledger",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_artifacts():
    global pipeline, calibrator, model, explainer, model_version, threshold, ledger
    pipeline_path = os.path.join(MODELS_DIR, "pipeline.pkl")
    calibrator_path = os.path.join(MODELS_DIR, "calibrator.pkl")
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_artifacts.json")

    if os.path.exists(pipeline_path):
        pipeline = joblib.load(pipeline_path)
    if os.path.exists(calibrator_path):
        calibrator = joblib.load(calibrator_path)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = None

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            model_version = meta.get("model_version", model_version)
            threshold = meta.get("optimal_threshold", threshold)

    ledger = AuditLedger(db_path=DEFAULT_DB_PATH)


@app.on_event("startup")
def startup_event():
    load_artifacts()


# Authentication & Rate Limiting Dependencies
async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Access denied."
        )
    return api_key


async def rate_limiter(request: Request, api_key: str = Security(api_key_header)):
    client_ip = request.client.host if request.client else "127.0.0.1"
    key_id = api_key or client_ip
    now = time.time()

    if key_id not in request_counts:
        request_counts[key_id] = []

    # Clean requests older than 60 seconds
    request_counts[key_id] = [t for t in request_counts[key_id] if now - t < 60.0]

    if len(request_counts[key_id]) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_PER_MINUTE} requests per minute."
        )

    request_counts[key_id].append(now)


# Pydantic Schemas with Strict Finite Validation
class SingleTransactionInput(BaseModel):
    Amount: float = Field(..., ge=0.0, description="Transaction amount (>= 0.0)")
    Time: Optional[float] = Field(3600.0, description="Seconds elapsed since first transaction")
    Hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of the day (0-23)")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0

    @validator("*", pre=True)
    def validate_finite(cls, v):
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                raise ValueError("Infinite or NaN values are not allowed.")
        return v

    class Config:
        allow_inf_nan = False


class BatchTransactionInput(BaseModel):
    transactions: List[SingleTransactionInput] = Field(..., max_items=500, description="Batch list of transactions (max 500)")

    class Config:
        allow_inf_nan = False


class TamperRequest(BaseModel):
    admin_key: str = Field(..., description="Admin secret key")
    block_index: int = Field(..., ge=0, description="Index of block to tamper")
    new_fraud_score: float = Field(0.9999, description="Tampered fraud score")



# Routes
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "calibrator_loaded": calibrator is not None,
        "pipeline_loaded": pipeline is not None,
        "model_version": model_version,
        "threshold": threshold,
        "demo_mode": DEMO_MODE
    }


@app.post("/predict", dependencies=[Depends(verify_api_key), Depends(rate_limiter)])
def predict_single(tx: SingleTransactionInput, background_tasks: BackgroundTasks):
    if model is None or calibrator is None or pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML Model artifacts are not loaded on server."
        )

    # Convert Pydantic payload to DataFrame
    raw_dict = tx.dict()
    df_raw = pd.DataFrame([raw_dict])

    # Transform features using pipeline fitted on train only
    X_scaled = pipeline.transform(df_raw)

    # Predict raw & calibrated probabilities
    raw_prob = float(model.predict_proba(X_scaled)[0, 1])
    calibrated_prob = float(calibrator.calibrate(np.array([raw_prob]))[0])

    is_fraud = bool(calibrated_prob >= threshold)
    decision = "FRAUD" if is_fraud else "APPROVE"

    # SHAP TreeExplainer calculation
    shap_contributions = []
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):  # Binary classifier list output
                vals = shap_values[1][0]
            else:
                vals = shap_values[0]

            for name, val in zip(FEATURE_NAMES, vals):
                shap_contributions.append({"feature": name, "shap_value": float(val)})
            shap_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            shap_contributions = shap_contributions[:5]
        except Exception:
            shap_contributions = []

    # Log payload entry to ledger asynchronously in background task
    if ledger is not None:
        background_tasks.add_task(
            ledger.record_entry,
            payload=raw_dict,
            decision=decision,
            fraud_score=calibrated_prob,
            model_version=model_version,
            threshold=threshold
        )

    return {
        "fraud_score": round(calibrated_prob, 6),
        "raw_score": round(raw_prob, 6),
        "is_fraud": is_fraud,
        "decision": decision,
        "threshold": threshold,
        "model_version": model_version,
        "top_shap_features": shap_contributions
    }


@app.post("/predict-batch", dependencies=[Depends(verify_api_key), Depends(rate_limiter)])
def predict_batch(batch: BatchTransactionInput, background_tasks: BackgroundTasks):
    if model is None or calibrator is None or pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML Model artifacts are not loaded on server."
        )

    if len(batch.transactions) > 500:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Batch payload exceeds max limit of 500 transactions."
        )

    df_raw = pd.DataFrame([tx.dict() for tx in batch.transactions])
    X_scaled = pipeline.transform(df_raw)

    raw_probs = model.predict_proba(X_scaled)[:, 1]
    calibrated_probs = calibrator.calibrate(raw_probs)

    results = []
    fraud_count = 0

    for idx, (tx, prob) in enumerate(zip(batch.transactions, calibrated_probs)):
        is_fraud = bool(prob >= threshold)
        dec = "FRAUD" if is_fraud else "APPROVE"
        if is_fraud:
            fraud_count += 1

        results.append({
            "id": idx + 1,
            "fraud_score": round(float(prob), 6),
            "is_fraud": is_fraud,
            "decision": dec
        })

        if ledger is not None:
            background_tasks.add_task(
                ledger.record_entry,
                payload=tx.dict(),
                decision=dec,
                fraud_score=float(prob),
                model_version=model_version,
                threshold=threshold
            )

    return {
        "total": len(results),
        "fraud_count": fraud_count,
        "legit_count": len(results) - fraud_count,
        "model_version": model_version,
        "threshold": threshold,
        "results": results
    }


@app.get("/ledger/verify", dependencies=[Depends(verify_api_key)])
def verify_ledger():
    if ledger is None:
        raise HTTPException(status_code=503, detail="Ledger database not initialized.")
    is_valid, message = ledger.verify_chain()
    return {
        "is_valid": is_valid,
        "message": message
    }


@app.get("/ledger/entries", dependencies=[Depends(verify_api_key)])
def get_ledger_entries(limit: int = 50, offset: int = 0):
    if ledger is None:
        raise HTTPException(status_code=503, detail="Ledger database not initialized.")
    entries = ledger.get_entries(limit=limit, offset=offset)
    return {
        "count": len(entries),
        "entries": entries
    }


@app.post("/demo/tamper")
def simulate_ledger_tamper(req: TamperRequest):
    if not DEMO_MODE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tamper simulation is disabled when DEMO_MODE=false."
        )

    if req.admin_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key for tamper simulation."
        )

    if ledger is None:
        raise HTTPException(status_code=503, detail="Ledger database not initialized.")

    ledger.tamper_block(block_index=req.block_index, new_fraud_score=req.new_fraud_score)
    is_valid, message = ledger.verify_chain()

    return {
        "tampered_block": req.block_index,
        "verification_result": {
            "is_valid": is_valid,
            "message": message
        }
    }
