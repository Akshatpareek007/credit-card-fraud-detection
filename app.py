from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import json
import numpy as np
import pandas as pd
import os
import time
import io
from blockchain import Blockchain

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time ML API using LightGBM & StandardScaler with Blockchain Ledger Integration",
    version="1.0.0"
)

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "creditcard.csv")

# Global variables for loaded artifacts
model = None
scaler = None
model_info = {}
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Hour"]
OPTIMAL_THRESHOLD = 0.9870245620674087

# Global Blockchain Instance
blockchain = Blockchain(difficulty=2)



def load_artifacts():
    global model, scaler, model_info, OPTIMAL_THRESHOLD
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH, "r") as f:
            model_info = json.load(f)
            OPTIMAL_THRESHOLD = model_info.get("best_threshold", model_info.get("threshold", OPTIMAL_THRESHOLD))


@app.get("/")
def read_root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Credit Card Fraud Detection API is running."}


@app.on_event("startup")
def startup_event():
    load_artifacts()
    # Seed initial transactions into blockchain for demonstration
    if len(blockchain.chain) == 1:  # Only genesis block exists
        blockchain.add_transaction(amount=14.99, fraud_probability=0.002, is_fraud=False, risk_level="LOW", hour=3, feature_summary="Online Subscription Payment")
        blockchain.add_transaction(amount=529.00, fraud_probability=0.991, is_fraud=True, risk_level="CRITICAL", hour=0, feature_summary="High-risk Anomalous Withdrawal")
        blockchain.add_transaction(amount=75.50, fraud_probability=0.015, is_fraud=False, risk_level="LOW", hour=14, feature_summary="Grocery Supermarket Store")
        # Pending txs reach 3, auto-mining creates Block #1!



class TransactionInput(BaseModel):
    Amount: Optional[float] = Field(default=100.0, description="Transaction Amount in Euros")
    Time: Optional[float] = Field(default=3600.0, description="Seconds elapsed since first transaction")
    Hour: Optional[int] = Field(default=None, description="Hour of the day (0-23). Auto-calculated from Time if omitted")
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


def extract_feature_vector(tx: TransactionInput) -> np.ndarray:
    hour_val = tx.Hour
    if hour_val is None:
        if tx.Time is not None:
            hour_val = int((tx.Time // 3600) % 24)
        else:
            hour_val = 12

    vector = [
        tx.V1, tx.V2, tx.V3, tx.V4, tx.V5, tx.V6, tx.V7, tx.V8, tx.V9, tx.V10,
        tx.V11, tx.V12, tx.V13, tx.V14, tx.V15, tx.V16, tx.V17, tx.V18, tx.V19, tx.V20,
        tx.V21, tx.V22, tx.V23, tx.V24, tx.V25, tx.V26, tx.V27, tx.V28,
        float(hour_val)
    ]
    return np.array(vector, dtype=np.float64).reshape(1, -1)


def compute_feature_contributions(raw_vector: np.ndarray, scaled_vector: np.ndarray) -> List[Dict[str, Any]]:
    # Top influential features estimation using feature importance & scaled deviation
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # normalize importances
        total_imp = np.sum(importances) if np.sum(importances) > 0 else 1.0
        norm_imp = importances / total_imp

        scaled_flat = scaled_vector.flatten()
        contributions = []
        for name, imp, val, raw in zip(FEATURE_NAMES, norm_imp, scaled_flat, raw_vector.flatten()):
            score = float(imp * abs(val))
            contributions.append({
                "feature": name,
                "importance_score": float(imp),
                "scaled_value": float(val),
                "raw_value": float(raw),
                "impact": float(score)
            })

        contributions.sort(key=lambda x: x["impact"], reverse=True)
        return contributions[:6]
    return []


@app.get("/api/health")
def get_health():
    results_path = os.path.join(os.path.dirname(__file__), "reports", "results.json")
    artifacts_path = os.path.join(os.path.dirname(__file__), "models", "model_artifacts.json")
    
    results_data = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results_data = json.load(f)
    elif os.path.exists(artifacts_path):
        with open(artifacts_path, "r") as f:
            results_data = json.load(f)

    return {
        "status": "online",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_name": model_info.get("best_model", "LightGBM"),
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "features": FEATURE_NAMES,
        "metrics": model_info.get("metrics", {}),
        "reports": results_data,
        "blockchain_summary": {
            "chain_length": len(blockchain.chain),
            "pending_count": len(blockchain.pending_transactions),
            "is_valid": blockchain.validate_chain().get("is_valid", True)
        }
    }



@app.post("/api/predict")
def predict_single(tx: TransactionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded properly on backend.")

    start_time = time.time()

    # Prepare vector
    raw_vec = extract_feature_vector(tx)
    scaled_vec = scaler.transform(raw_vec)

    # Predict probability
    probs = model.predict_proba(scaled_vec)[0]
    fraud_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
    is_fraud = bool(fraud_prob >= OPTIMAL_THRESHOLD)

    # Determine risk level
    if fraud_prob < 0.2:
        risk_level = "LOW"
    elif fraud_prob < 0.7:
        risk_level = "MODERATE"
    elif fraud_prob < OPTIMAL_THRESHOLD:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    contributions = compute_feature_contributions(raw_vec, scaled_vec)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Log transaction to Blockchain Ledger
    bc_log = blockchain.add_transaction(
        amount=tx.Amount or 0.0,
        fraud_probability=fraud_prob,
        is_fraud=is_fraud,
        risk_level=risk_level,
        hour=int(raw_vec[0][-1]),
        feature_summary=f"Top Driver: {contributions[0]['feature']}" if contributions else "Standard Vector"
    )

    return {
        "fraud_probability": fraud_prob,
        "fraud_percentage": round(fraud_prob * 100, 2),
        "is_fraud": is_fraud,
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "threshold_percentage": round(OPTIMAL_THRESHOLD * 100, 2),
        "risk_level": risk_level,
        "amount": tx.Amount,
        "hour": raw_vec[0][-1],
        "top_feature_drivers": contributions,
        "latency_ms": latency_ms,
        "blockchain": {
            "tx_id": bc_log["transaction"]["tx_id"],
            "tx_hash": bc_log["transaction"]["tx_hash"],
            "status": "MINED_IN_BLOCK" if bc_log["mined_block"] else "QUEUED_IN_PENDING_POOL",
            "block_index": bc_log["mined_block"]["index"] if bc_log["mined_block"] else None,
            "pending_count": bc_log["pending_count"]
        }
    }



@app.post("/api/predict-batch")
def predict_batch(transactions: List[TransactionInput]):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded properly on backend.")

    if not transactions:
        return {"total": 0, "fraud_count": 0, "legit_count": 0, "results": []}

    vectors = np.vstack([extract_feature_vector(tx) for tx in transactions])
    scaled_vectors = scaler.transform(vectors)
    probs = model.predict_proba(scaled_vectors)[:, 1]

    results = []
    fraud_count = 0
    total_amount_at_risk = 0.0

    for idx, (tx, prob) in enumerate(zip(transactions, probs)):
        is_fraud = bool(prob >= OPTIMAL_THRESHOLD)
        if is_fraud:
            fraud_count += 1
            if tx.Amount:
                total_amount_at_risk += tx.Amount

        results.append({
            "id": idx + 1,
            "amount": tx.Amount,
            "hour": int(tx.Hour if tx.Hour is not None else ((tx.Time or 3600) // 3600 % 24)),
            "fraud_probability": float(prob),
            "fraud_percentage": round(float(prob) * 100, 2),
            "is_fraud": is_fraud,
            "recommendation": "BLOCK & REVIEW" if is_fraud else "APPROVE"
        })

    return {
        "total": len(transactions),
        "fraud_count": fraud_count,
        "legit_count": len(transactions) - fraud_count,
        "fraud_rate_pct": round((fraud_count / len(transactions)) * 100, 2),
        "total_amount_at_risk": round(total_amount_at_risk, 2),
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "results": results
    }


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV file.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Build transaction inputs
    tx_list = []
    for _, row in df.iterrows():
        tx_dict = {}
        for col in df.columns:
            if col in TransactionInput.__fields__:
                tx_dict[col] = float(row[col])
        tx_list.append(TransactionInput(**tx_dict))

    return predict_batch(tx_list)


@app.get("/api/sample")
def get_sample_transactions():
    """Return real curated sample transactions directly from dataset for UI quick loading."""
    if os.path.exists(DATASET_PATH):
        try:
            # Read small chunk of dataset
            df = pd.read_csv(DATASET_PATH, nrows=5000)
            legit_rows = df[df['Class'] == 0].sample(n=min(3, len(df[df['Class'] == 0])))
            fraud_rows = df[df['Class'] == 1].sample(n=min(3, len(df[df['Class'] == 1])))

            def row_to_dict(row):
                d = row.to_dict()
                # Compute Hour
                d['Hour'] = int((d['Time'] // 3600) % 24)
                return d

            return {
                "legitimate": [row_to_dict(r) for _, r in legit_rows.iterrows()],
                "fraudulent": [row_to_dict(r) for _, r in fraud_rows.iterrows()]
            }
        except Exception as e:
            pass

    # Default fallback sample data if dataset file isn't loaded
    return {
        "legitimate": [
            {
                "Amount": 14.99, "Time": 14200.0, "Hour": 3,
                "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
                "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
                "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
                "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
                "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
                "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
                "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053
            }
        ],
        "fraudulent": [
            {
                "Amount": 529.00, "Time": 406.0, "Hour": 0,
                "V1": -2.312227, "V2": 1.951992, "V3": -1.609851, "V4": 3.997906,
                "V5": -0.522188, "V6": -1.426545, "V7": -2.537387, "V8": 1.391657,
                "V9": -2.770089, "V10": -2.772272, "V11": 3.202033, "V12": -2.899907,
                "V13": -0.595222, "V14": -4.289254, "V15": 0.389724, "V16": -1.140747,
                "V17": -2.830056, "V18": -0.016822, "V19": 0.416956, "V20": 0.126911,
                "V21": 0.517232, "V22": -0.035049, "V23": -0.465211, "V24": 0.320198,
                "V25": 0.044519, "V26": 0.177840, "V27": 0.261145, "V28": -0.143276
            }
        ]
    }


class TamperRequest(BaseModel):
    block_index: int = Field(default=1, description="Index of block to tamper")
    transaction_index: int = Field(default=0, description="Index of transaction inside block")
    new_amount: float = Field(default=9999.0, description="Tampered transaction amount")
    flip_fraud: bool = Field(default=True, description="Whether to flip fraud status")


@app.get("/api/blockchain/chain")
def get_blockchain_chain():
    """Retrieve full blockchain ledger, pending transaction pool, and validation status."""
    return blockchain.get_chain_data()


@app.post("/api/blockchain/mine")
def mine_block():
    """Mine pending transactions into a new Block using Proof-of-Work."""
    mined_block = blockchain.mine_pending_transactions()
    return {
        "message": f"Block #{mined_block.index} mined successfully!",
        "mined_block": mined_block.to_dict(),
        "chain_data": blockchain.get_chain_data()
    }


@app.get("/api/blockchain/validate")
def validate_blockchain():
    """Validate SHA-256 cryptographic hashes and block linkages across the whole chain."""
    return blockchain.validate_chain()


@app.post("/api/blockchain/tamper")
def tamper_blockchain(req: TamperRequest):
    """Simulate a cyber attack tampering with transaction data in a block to test detection."""
    result = blockchain.simulate_tamper(
        block_index=req.block_index,
        transaction_index=req.transaction_index,
        new_amount=req.new_amount,
        flip_fraud=req.flip_fraud
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return {
        "result": result,
        "validation_status": blockchain.validate_chain()
    }


@app.post("/api/blockchain/repair")
def repair_blockchain():
    """Repair tampered hashes and re-mine PoW nonces to restore blockchain validity."""
    result = blockchain.repair_chain()
    return {
        "result": result,
        "validation_status": blockchain.validate_chain()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

