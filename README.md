# 🛡️ Sentinel Fraud AI — Credit Card Fraud Detection & Audit Ledger

[![Python 3.12](https://img.shields.io/badge/Python-3.12%2B-blue.svg?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Pytest](https://img.shields.io/badge/Pytest-Passed%20(20%2F20)-brightgreen.svg?logo=pytest)](https://docs.pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade Machine Learning and Cryptographic Audit Ledger project for detecting credit card transaction fraud. Built strictly with original code, temporal data splitting, Platt probability calibration, financial cost minimization, SHAP explainability, and an append-only Ed25519-signed SQLite audit ledger.

---

## 📌 Problem Statement

Credit card transaction fraud poses severe financial risks to banking networks. In real-world payment systems:
1. **Extreme Class Imbalance:** Fraudulent transactions comprise less than **0.2%** of total card volume. Standard accuracy is misleading; optimization requires **Precision-Recall AUC (PR-AUC)** and **Expected Financial Cost Minimization**.
2. **Temporal Distribution Shift:** Financial fraud patterns evolve over time. Random k-fold cross-validation causes temporal data leakage; model evaluation requires strict **chronological splitting**.
3. **Auditability & Regulatory Compliance:** Payment processors require tamper-evident logging of AI model versioning, probability scores, decision thresholds, and cryptographic signatures for audit trials.

---

## 📂 Dataset & Citation

- **Dataset Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Dataset Specs:** 284,807 transactions recorded over two days in September 2013 by European cardholders.
- **Features:** 30 total features (`Time`, `Amount`, `V1`–`V28` anonymized PCA components, and `Class`).

> ⚠️ `creditcard.csv` (~150 MB) is **not committed** to this repository per repository guidelines (`.gitignore`).

### Download Instructions
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place `creditcard.csv` in the project root directory before executing the training pipeline.

### Required Citation
```bibtex
@misc{ulb_creditcard_fraud,
  author = {Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi},
  title = {Credit Card Fraud Detection Dataset},
  year = {2015},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud}}
}
```

---

## 🏗️ Project Structure

```
credit_card_fraud_detection/
├── src/
│   ├── __init__.py         # Package root
│   ├── data.py             # CSV loading, validation, deduplication, temporal split
│   ├── features.py         # Feature engineering (Hour, LogAmount) & StandardScaler pipeline
│   ├── evaluate.py         # Metrics, cost threshold minimization, bootstrap CIs, calibration
│   ├── ledger.py           # Tamper-evident SQLite WAL ledger with Ed25519 signatures
│   ├── train.py            # Model tuning, Platt calibration, & artifact generation
│   └── api.py              # Production FastAPI server with SHAP & rate limiting
├── tests/
│   ├── __init__.py
│   ├── test_data.py        # Temporal split ordering & scaler leakage tests
│   ├── test_ledger.py      # Canonical serialization, Ed25519 signatures, tamper detection
│   ├── test_evaluate.py    # Metric reproducibility & cost optimization tests
│   └── test_api.py         # FastAPI authentication, rate limiting, & Pydantic tests
├── models/
│   ├── lightgbm_model.txt  # Native LightGBM model export (no pickle)
│   ├── lightgbm_model.txt.sha256 # SHA-256 checksum file
│   └── model_artifacts.json# Model versioning & threshold metadata
├── reports/
│   ├── results.json        # Generated evaluation metrics & benchmark comparisons
│   └── ledger.db           # SQLite WAL audit ledger database
├── requirements.txt        # Dependency manifest
├── LICENSE                 # MIT License
└── README.md
```

---

## 🔬 Methodology & Empirical Results

*(All metrics generated automatically by `python -m src.train` and stored in `reports/results.json`)*

### 1. Data Cleaning & Temporal Splitting
* **Raw Rows:** 284,807
* **Deduplication:** **1,081** exact duplicate rows removed before splitting $\rightarrow$ **283,726** clean transactions.
* **Strict Temporal Split (ordered by `Time`):**
  * **Train (70%):** 198,608 rows (366 frauds, 0.184% fraud rate) | Time: $0.0\text{s} \rightarrow 132,906.0\text{s}$
  * **Validation (15%):** 42,558 rows (55 frauds, 0.129% fraud rate) | Time: $132,906.0\text{s} \rightarrow 151,320.0\text{s}$
  * **Test (15%):** 42,560 rows (52 frauds, 0.122% fraud rate) | Time: $151,320.0\text{s} \rightarrow 172,792.0\text{s}$

### 2. Preprocessing & Leakage Prevention
* `StandardScaler` is fitted **strictly on the training set** inside an `sklearn.pipeline.Pipeline`.
* Engineered features: `Hour = (Time // 3600) % 24` and `LogAmount = log1p(Amount)`.

### 3. Validation Set Model & Imbalance Strategy Benchmark

| Candidate Model | Imbalance Strategy | Val PR-AUC | Val ROC-AUC | Val F1-Score | Expected Val Cost |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** ✦ | **None** | **0.8784** | **0.9857** | **0.8710** | **$3,200.00** |
| Random Forest | None (Baseline) | 0.8675 | 0.9828 | 0.7218 | $3,800.00 |
| LightGBM | Class Weight | 0.8536 | 0.9796 | 0.5104 | $3,880.00 |
| Logistic Regression | SMOTE | 0.8412 | 0.9788 | 0.5581 | $4,190.00 |
| Random Forest | Random Undersampling | 0.8351 | 0.9811 | 0.8000 | $4,640.00 |
| Logistic Regression | Class Weight | 0.8239 | 0.9815 | 0.3953 | $3,980.00 |
| Decision Tree | None | 0.5999 | 0.7781 | 0.7551 | $9,060.00 |

*Winning Architecture:* **XGBoost Classifier** selected based on top Validation PR-AUC (`0.8784`).

### 4. Platt Probability Calibration & Expected Cost Threshold Minimization
* **Platt Scaling:** Fitted on validation set probabilities using logistic regression in logit space.
* **Cost Function:** Configured with $\text{cost}_{\text{FN}} = \$500.00$ (uncaught fraud) and $\text{cost}_{\text{FP}} = \$10.00$ (false alarm friction).
* **Optimal Threshold:** Chosen on the validation set minimizing total expected financial cost $\rightarrow$ **`0.0740`**.

### 5. Final Test Set Evaluation (Evaluated ONCE on Held-Out Test Set)

| Metric | Test Value | 95% Bootstrap Confidence Interval (1000 Resamples) |
| :--- | :---: | :---: |
| **ROC-AUC** | **0.9844** | `[0.9725, 0.9938]` |
| **PR-AUC** | **0.7533** | `[0.6385, 0.8568]` |
| **F1-Score** | **0.6441** | `[0.5357, 0.7424]` |
| **Recall** | **0.7308** | `[0.6041, 0.8438]` |
| **Precision** | **0.5758** | `[0.4507, 0.6936]` |
| **Total Test Cost** | **$7,280.00** | — |

**Test Confusion Matrix:**
* True Negatives (TN): `42,480`
* False Positives (FP): `28`
* False Negatives (FN): `14`
* True Positives (TP): `38`

---

## ⛓️ Audit Ledger & Security Architecture

The system includes a custom, single-node SQLite WAL audit ledger ([`src/ledger.py`](file:///d:/PROJECTS/credit_card_fraud_detection/src/ledger.py)):

### Key Security Features
- **Canonical Payload Hashing:** Payload entries undergo deterministic key-sorted JSON serialization with 8-decimal float formatting and zero NaN/Inf tolerance before SHA-256 hashing.
- **Ed25519 Cryptographic Signatures:** Every block header is signed with an Ed25519 private key (`LEDGER_ED25519_KEY`).
- **Zero Raw Data Storage:** The ledger records **only** `payload_hash`, `decision`, `fraud_score`, `model_version`, `threshold`, and cryptographic signatures. No raw card numbers or features are stored.
- **Non-Blocking Background Logging:** Ledger writes execute asynchronously in background tasks guarded by a thread-safe single writer lock.

> 📢 **Honest Disclosure:** This ledger is **tamper-EVIDENT**, single-node, and intended for auditability. It is **not** a replacement for a decentralized multi-party consensus blockchain (e.g., Ethereum or Hyperledger Fabric).

---

## 🚀 How to Run

### 1. Installation
```bash
git clone https://github.com/Akshatpareek007/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execute Model Training Pipeline
```bash
python -m src.train
```

### 3. Run PyTest Test Suite
```bash
pytest -v
```

### 4. Start FastAPI Production Server
```bash
uvicorn src.api:app --reload --port 8000
```
* **API Documentation:** `http://127.0.0.1:8000/docs`
* **Default API Key:** `X-API-Key: sentinel-secret-key-123`

---

## 📦 Dependency License Table

| Library | License | Usage Purpose |
| :--- | :--- | :--- |
| `scikit-learn` | BSD 3-Clause | Data preprocessing, pipelines, tree classifiers |
| `pandas` | BSD 3-Clause | DataFrame manipulation and temporal indexing |
| `numpy` | BSD 3-Clause | Vectorized numerical operations & metrics |
| `imbalanced-learn` | MIT | SMOTE and random undersampling strategies |
| `lightgbm` | MIT | Native gradient boosted tree classifier export |
| `xgboost` | Apache 2.0 | Gradient boosting candidate classifier |
| `shap` | MIT | TreeExplainer model interpretability |
| `cryptography` | Apache 2.0 / BSD | Ed25519 digital signature signing & verification |
| `fastapi` | MIT | REST API web framework |
| `uvicorn` | BSD 3-Clause | ASGI web server |
| `pytest` | MIT | Automated unit & integration testing framework |

---

## ⚠️ Limitations & Disclaimers

1. **Research & Educational Demo:** This project is a research prototype demonstration.
2. **Static Historical Dataset (2013):** The dataset spans 48 hours of European transactions from September 2013 and does not reflect real-time 2026 fraud vectors.
3. **Anonymized PCA Components:** Features V1–V28 are PCA-anonymized, limiting domain-specific feature engineering (e.g., merchant category code, geo-distance).

---

## 🙋 Acknowledgements & Credits

- **Dataset Authors:** ULB (Université Libre de Bruxelles) Machine Learning Group (Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi).
- **Libraries:** Thanks to the open-source maintainers of `scikit-learn`, `LightGBM`, `XGBoost`, `FastAPI`, `SHAP`, and `Cryptography`.

---
*License:* **MIT License** — See [`LICENSE`](file:///d:/PROJECTS/credit_card_fraud_detection/LICENSE) for full text.
