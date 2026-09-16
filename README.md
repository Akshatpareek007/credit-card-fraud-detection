# 💳 Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?logo=scikitlearn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-222?logo=github)](https://rajneeshbabu.github.io/credit-card-fraud-detection/)

🌐 **[View Project Page →](https://rajneeshbabu.github.io/credit-card-fraud-detection/)**

An end-to-end machine learning project for detecting fraudulent credit card transactions. Combines a tuned **XGBoost** classifier with a **PyTorch autoencoder** for anomaly detection — with full SHAP explainability and GPU acceleration.

---

## 📌 Problem Statement

Credit card fraud is a critical challenge for financial institutions. This project builds a binary classifier on a highly imbalanced dataset where only **0.17%** of transactions are fraudulent. Standard accuracy is meaningless here — the project focuses on **Precision-Recall AUC** and **F1-Score**.

---

## 📂 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions (September 2013, European cardholders)
- **Features:** 30 total — `Time`, `Amount`, and `V1`–`V28` (PCA-anonymised)
- **Target:** `Class` — `0` = Legitimate, `1` = Fraud
- **Imbalance:** Only 492 fraud cases (0.172%)

> ⚠️ `creditcard.csv` is not included in this repo (~150 MB). Download it from Kaggle and place it in the project root before running the notebook.

---

## 🗂️ Project Structure

```
credit-card-fraud-detection/
│
├── ccfd.ipynb          # Main notebook — full pipeline (run this)
├── requirements.txt    # Python dependencies
├── README.md
├── .gitignore
│
├── creditcard.csv      # Dataset — download from Kaggle, not in repo
└── models/             # Auto-generated after running the notebook
    ├── best_model.pkl      # Tuned XGBoost classifier (780 KB)
    ├── scaler.pkl          # Fitted StandardScaler (1.7 KB)
    ├── autoencoder.pt      # PyTorch autoencoder weights (49 KB)
    └── model_info.json     # Thresholds, metrics, best params
```

---

## 🔬 Pipeline (inside `ccfd.ipynb`)

### 1. Data Loading & EDA
- 284,807 transactions, 0 null values
- Class distribution: 99.83% legit, 0.17% fraud
- Transaction amount & hour-of-day analysis by class
- Pearson correlation heatmap — top features correlated with fraud
- KDE plots for PCA features (V1–V10) by class

### 2. Feature Engineering & Preprocessing
- Extract `Hour` from `Time` column — captures daily fraud patterns
- Drop raw `Time` and `Amount` (replaced by engineered/scaled version)
- **Stratified 80/20 train/test split** — 227,845 train, 56,962 test
- `StandardScaler` fit **only on training data** (no leakage)
- **SMOTE** applied **only on training data**: 394 fraud → 227,451 fraud (balanced)

### 3. Model Comparison
All four candidates trained on the SMOTE-balanced set, evaluated on the held-out test set:

| Model | ROC-AUC | PR-AUC | F1 (0.5 thresh) |
|-------|---------|--------|-----------------|
| Logistic Regression | 0.9705 | 0.7173 | 0.1036 |
| Random Forest | 0.9692 | 0.8543 | 0.8333 |
| **XGBoost** ✦ | **0.9835** | **0.8658** | 0.8195 |
| LightGBM | 0.9417 | 0.7253 | 0.8137 |

**XGBoost** selected as best model by PR-AUC.

### 4. Hyperparameter Tuning — Random Search (No Cross-Validation)
- SMOTE training data split once: 80% fit set, 20% validation set
- 40 random parameter combinations sampled and scored on the validation set
- Best configuration found:

```
n_estimators    = 400
max_depth       = 5
learning_rate   = 0.2
subsample       = 0.8
colsample_bytree= 1.0
scale_pos_weight= 5
```

- Final model retrained on the **full** SMOTE training set with these params

### 5. Threshold Tuning & Final Evaluation
- Default threshold (0.50) → F1 = **0.7905**
- Optimal threshold (0.9837) → F1 = **0.8696**

**Final classification report (tuned XGBoost, threshold=0.9837):**

```
              precision    recall  f1-score   support
       Legit     1.0000    1.0000    0.9998     56,864
       Fraud     0.9302    0.8163    0.8696         98
    accuracy                         0.9996     56,962
```

### 6. SHAP Explainability
- `TreeExplainer` global feature importance bar chart
- Beeswarm plot — feature impact direction on fraud predictions

### 7. PyTorch Autoencoder — Anomaly Detection
- Architecture: `29 → 64 → 32 → 16 → 32 → 64 → 29` (9,453 parameters)
- Trained on **227,451 legitimate transactions only**
- GPU accelerated (Tesla T4 / Apple MPS / CPU — auto-detected)
- Early stopping + ReduceLROnPlateau scheduler
- Legit mean reconstruction error: **0.082** | Fraud mean: **14.769**
- Autoencoder PR-AUC: **0.6330** | F1: **0.7136**

### 8. Ensemble
- `ensemble_score = 0.6 × XGBoost_prob + 0.4 × normalised_ae_error`
- Ensemble PR-AUC: **0.8320** | F1: **0.8743**

### 9. Save Artefacts
All saved to `models/`: `best_model.pkl`, `scaler.pkl`, `autoencoder.pt`, `model_info.json`

---

## 📊 Final Results

| Model | ROC-AUC | PR-AUC | F1 | Notes |
|-------|---------|--------|----|-------|
| Logistic Regression | 0.9705 | 0.7173 | 0.1036 | Baseline |
| Random Forest | 0.9692 | 0.8543 | 0.8333 | — |
| XGBoost (base) | 0.9835 | 0.8658 | 0.8195 | Best model |
| **XGBoost (tuned)** | **0.9857** | **0.8614** | **0.8696** | **After threshold tuning** |
| LightGBM | 0.9417 | 0.7253 | 0.8137 | — |
| Autoencoder (PyTorch) | — | 0.6330 | 0.7136 | Anomaly detection only |
| **Ensemble** | — | **0.8320** | **0.8743** | XGBoost + Autoencoder |

> Fraud detection precision: **93%** — Recall: **82%** — catching 80 out of 98 fraud cases in the test set.

---

## ⚙️ Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/rajneeshbabu/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install PyTorch

```bash
pip install torch torchvision torchaudio
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

---

## 🚀 Run the Notebook

```bash
jupyter notebook ccfd.ipynb
```

Run all cells top to bottom. Expected runtime: **~15–30 minutes** (depends on hardware). The notebook auto-detects GPU (CUDA / Apple MPS) and falls back to CPU.

---

## 🧠 Key Learnings

- **Accuracy is misleading** — predicting "Legit" every time gives 99.83% accuracy but catches 0 frauds
- **PR-AUC is the right metric** for fraud detection on imbalanced data
- **SMOTE and Scaler must be fit on training data only** — prevents data leakage
- **No cross-validation** — hyperparameter search uses a single held-out validation split (faster, no overfitting to folds)
- **Threshold tuning matters** — moving from 0.5 to 0.9837 improved F1 from 0.79 to 0.87
- **Autoencoder reconstruction gap** — fraud transactions have ~180× higher reconstruction error than legit ones (14.77 vs 0.08)
- **SHAP** confirms V14, V17, V12 as the strongest fraud indicators

---

## 📄 License

MIT License

---

## 🙋 Author

**Rajneesh Babu**  
GitHub: [@rajneeshbabu](https://github.com/rajneeshbabu)

---

## ⭐ Acknowledgements

- Dataset: [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) via Kaggle
- SHAP: [slundberg/shap](https://github.com/slundberg/shap)
- PyTorch: [pytorch.org](https://pytorch.org)
