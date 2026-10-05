# ❤️ Cardio AI — Cardiovascular Risk Prediction & Explainable AI Dashboard

> **Educational & Research Prototype** — Not a medical device. Always consult a qualified healthcare professional for clinical decisions.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-red)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.52-purple)](https://shap.readthedocs.io)

---

## 1. Project Overview

Cardio AI is a full-stack ML application that predicts **cardiovascular disease risk** from 13 clinical features (UCI Cleveland Heart Disease schema). It combines a production-grade ML pipeline with an **8-section interactive Streamlit dashboard** and a **FastAPI backend**, making it suitable for:

- 🎓 College project demonstrations
- 🏆 Hackathon showcases
- 📁 Resume / GitHub portfolio
- 🔬 ML explainability research

---

## 2. Problem Statement

Cardiovascular disease is a leading cause of mortality worldwide. Early identification of high-risk individuals can enable timely intervention. This project demonstrates how a calibrated ML model — paired with Explainable AI (SHAP) — can surface interpretable risk signals from routine clinical measurements.

---

## 3. Features

| Feature | Description |
|---|---|
| 🩺 **Patient Risk Assessment** | 3-column clinical input form with validation, sample presets, and grouped sections |
| 📊 **Executive Dashboard** | Model KPIs, last patient summary, and ML pipeline mermaid diagram |
| 🔍 **Explainable AI (SHAP)** | Waterfall plots, global feature importance, GenAI plain-language summary |
| ⚠️ **Clinical Risk Factor Analysis** | AHA/ACC educational benchmarks for BP, cholesterol, HR, ST depression |
| 📈 **Model Performance** | Confusion matrix, ROC curve, precision-recall curve, cross-validation comparison |
| 🔬 **Dataset Insights & EDA** | Target distribution, age histogram, correlation heatmap, feature stats |
| 🧬 **Model Architecture & Registry** | CV comparison table, pipeline metadata, full architecture description |
| 📜 **Prediction History** | Session-based audit log with CSV export |

---

## 4. Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.65 (multi-page dashboard) |
| Backend | FastAPI + Uvicorn (REST API) |
| ML | scikit-learn 1.9.1 (SVM + CalibratedClassifierCV) |
| Explainability | SHAP 0.52 (model-agnostic PermutationExplainer) |
| GenAI | OpenAI API (falls back to deterministic template) |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Validation | Pydantic v2 |
| Testing | pytest + Streamlit AppTest |

---

## 5. ML Workflow

```
Raw Clinical Data (13 features)
        ↓
  Input Validation (Pydantic)
        ↓
  Preprocessing Pipeline
  ├─ StandardScaler  (5 numerical features)
  └─ OneHotEncoder   (8 categorical features)
        ↓            → 21 engineered features
  CalibratedClassifierCV(SVC, method='sigmoid', cv=5)
        ↓
  Risk Probability Score  [0.0 – 1.0]
        ↓
  SHAP PermutationExplainer  (model-agnostic)
        ↓
  Feature Attribution Evidence Bundle
        ↓
  GenAI Plain-Language Summary  (OpenAI / template fallback)
        ↓
  Explainable Risk Report
```

---

## 6. Dataset

| Property | Value |
|---|---|
| Schema | UCI Cleveland Heart Disease (13 features) |
| Features | `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal` |
| Target | Binary: `0` = No disease, `1` = Disease present |
| Split | 80% train / 20% test (stratified, seed=42) |
| Preprocessing | StandardScaler + OneHotEncoder (drop='first'), fitted on train split only |

---

## 7. Model

**Winner (5-fold CV on training split):** `CalibratedClassifierCV(SVC(C=1, kernel='rbf'), method='sigmoid', cv=5)`

### Cross-Validation Comparison

| Model | CV ROC-AUC (mean ± std) | Selected |
|---|---|---|
| Logistic Regression | 0.790 ± 0.063 | |
| Decision Tree | 0.741 ± 0.054 | |
| Random Forest | 0.771 ± 0.085 | |
| **SVM (RBF)** | **0.805 ± 0.062** | ✅ |
| Neural Network (TF) | 0.731 ± 0.075 | |

### Final Test-Set Metrics (computed once)

| Metric | Value |
|---|---|
| Accuracy | 0.6964 |
| Precision | 0.7037 |
| Recall | 0.6786 |
| F1 Score | 0.6909 |
| ROC-AUC | 0.7895 |
| PR-AUC | 0.7792 |
| Brier Score | 0.1896 |

> **Note:** These metrics reflect the current dataset (synthetic/sample). Results on a larger, real-world clinical dataset would differ.

---

## 8. Explainable AI

- **Method:** `shap.Explainer` (model-agnostic PermutationExplainer) wrapping `predict_proba`
- **Output space:** Probability space — all SHAP contributions are in [−1, +1] probability units
- **Global importance:** Cached at startup using 15 background samples
- **Local explanations:** Per-patient waterfall-style SHAP bar chart
- **GenAI layer:** Safety-validated plain-language summary (keyword denylist + Pydantic schema)

---

## 9. Performance Metrics

See [`reports/model_comparison.md`](reports/model_comparison.md) and the generated plots in `reports/`:

| Plot | File |
|---|---|
| Confusion Matrix | `reports/confusion_matrix.png` |
| ROC Curve | `reports/roc_curve.png` |
| Precision-Recall Curve | `reports/pr_curve.png` |
| Feature Importance | `reports/feature_importance.png` |

---

## 10. Project Structure

```
cardio-ai/
├── app/
│   └── streamlit_app.py        # 8-section Streamlit dashboard
├── api/
│   └── main.py                 # FastAPI backend (predict, explain, generate-explanation)
├── src/
│   ├── data/
│   │   ├── preprocessing.py    # Leak-free sklearn pipeline
│   │   └── validation.py       # Pydantic input validation
│   ├── explainability/
│   │   └── shap_explainer.py   # SHAP evidence layer + global importance cache
│   ├── models/
│   │   ├── train.py            # Training script (CV selection + final fit)
│   │   └── evaluate.py         # Metrics + evaluation plots
│   └── utils/
│       └── config.py           # Central config (env vars, paths)
├── models/
│   ├── best_model.pkl
│   ├── preprocessing_pipeline.pkl
│   ├── feature_columns.json
│   ├── metadata.json
│   └── background_sample.csv
├── data/
│   └── sample_synthetic_heart.csv
├── reports/
│   ├── model_comparison.md
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── feature_importance.png
├── tests/                      # pytest test suite (47 tests)
├── .env                        # Environment variables
├── requirements.txt
└── README.md
```

---

## 11. How to Run

### Prerequisites
- Python 3.12
- Install dependencies: `pip install -r requirements.txt`

### Step 1 — Train the Model (if not already trained)
```powershell
py -3.12 -m src.models.train --data data/sample_synthetic_heart.csv
```

### Step 2 — Start the FastAPI Backend
```powershell
py -3.12 -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Step 3 — Start the Streamlit Dashboard
```powershell
py -3.12 -m streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

### Step 4 — Open in Browser
- **Dashboard:** http://127.0.0.1:8501
- **API Docs:** http://127.0.0.1:8000/docs

---

## 12. Demo Flow

1. Open the dashboard → **Patient Risk Assessment** (default page)
2. Select a sample profile → **Apply Selected Profile**
3. Click **🚀 Run Comprehensive Assessment**
4. View the risk probability + risk card
5. Navigate to **🔍 Explainable AI** to see SHAP feature attributions
6. Navigate to **⚠️ Clinical Risk Factor Analysis** to see AHA/ACC benchmarks
7. Navigate to **📈 Model Performance** to view evaluation metrics
8. Navigate to **📜 Prediction History** to export session data as CSV

---

## 13. API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Backend health check |
| `/metadata` | GET | Model name, version, feature list |
| `/predict` | POST | Returns risk probability |
| `/explain` | POST | Returns SHAP feature attributions |
| `/generate-explanation` | POST | Returns GenAI plain-language summary |

Full interactive docs at **http://127.0.0.1:8000/docs**

---

## 14. Testing

```powershell
py -3.12 -m pytest tests/ -v          # 47 unit + integration tests
py -3.12 -X utf8 scratch/test_e2e.py  # Full dashboard E2E (AppTest)
```

---

## 15. Disclaimer

> This application is built for **educational and research purposes only**.  
> It is **not** a medical device, clinical decision support tool, or diagnostic system.  
> Do not use model predictions to make clinical decisions.  
> Always consult a qualified healthcare professional.

---

## 16. Author

Built as an AI/ML portfolio project demonstrating end-to-end ML engineering:  
training pipeline → REST API → explainable AI → professional dashboard.
