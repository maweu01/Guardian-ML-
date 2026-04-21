# GUARDIAN ML

**Geospatial Unified Anomaly Risk Detection & Intelligence Analysis Network**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.4-F7931E?style=flat-square&logo=scikitlearn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

GUARDIAN ML is a **production-grade, end-to-end machine learning system** for geospatial risk assessment, anomaly detection, and predictive analytics. It ingests multi-format datasets, applies a rigorous preprocessing pipeline, trains an ensemble of ML models, and delivers composite risk scores with a full visual intelligence dashboard.

Designed to operate at the intersection of geospatial science and applied machine learning — built for researchers, analysts, and field deployment teams.

---

## Scientific Motivation

Risk assessment in geospatial domains is a multi-signal problem. No single model or metric captures the full complexity of spatially-structured, temporally-evolving risk. GUARDIAN ML addresses this by:

1. **Ensemble learning** — combining Random Forest, XGBoost, and Logistic Regression to reduce prediction variance
2. **Unsupervised anomaly detection** — IsolationForest identifies structural outliers that supervised models may miss
3. **Composite risk scoring** — integrates model confidence, anomaly evidence, spatial density, and temporal recency into a single [0,1] risk index
4. **Reproducible science** — fixed random seeds, logged pipelines, persisted artifacts, YAML-driven configuration

---

## Features

| Feature                        | Description                                              |
|--------------------------------|----------------------------------------------------------|
| Multi-format ingest            | CSV, JSON, GeoJSON, XLSX                                 |
| 10-step preprocessing          | Imputation, scaling, outlier removal, feature extraction |
| Geospatial feature engineering | Auto-derives lat/lon features and KDE density index      |
| Temporal feature extraction    | Parses datetime columns to cyclical features             |
| Multi-model training           | RF + XGBoost + LR + IsolationForest                      |
| 5-fold cross-validation        | Stratified, with mean ± std reporting                    |
| Composite risk scoring         | 4-component weighted risk index                          |
| 7 visualization types          | Plotly dark-theme charts, geospatial map                 |
| RESTful API                    | 18 endpoints with Swagger UI                             |
| Web dashboard                  | Dark cartographic UI, no framework dependencies          |
| Structured logging             | Loguru with file rotation and console output             |
| Full test suite                | pytest + httpx async tests (25+ test cases)              |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  GUARDIAN ML                         │
│                                                     │
│  Frontend (HTML/CSS/JS + Plotly.js)                 │
│      │                                              │
│      ▼                                              │
│  FastAPI Backend                                    │
│  ├── /upload   → File ingest + schema inspect       │
│  ├── /process  → 10-step preprocessing pipeline     │
│  ├── /train    → Multi-model + cross-validation     │
│  ├── /predict  → Inference + risk scoring           │
│  └── /visualize → 7 Plotly chart endpoints          │
│      │                                              │
│      ▼                                              │
│  Core ML Engine                                     │
│  ├── GuardianPreprocessor                           │
│  ├── GuardianTrainer (RF + XGB + LR + IsoForest)   │
│  ├── RiskScorer (composite 4-component)             │
│  └── Visualizer (Plotly dark theme)                 │
└─────────────────────────────────────────────────────┘
```

For full architecture documentation, see [`docs/system_architecture.md`](docs/system_architecture.md).

---

## Installation

### Requirements
- Python 3.10 or higher
- pip

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/Guardian-ML.git
cd Guardian-ML

# 2. Install dependencies (creates venv automatically)
./run.sh install

# 3. Start the server
./run.sh start

# 4. Open the dashboard
# Navigate to http://localhost:8000
```

### Manual Installation

```bash
cd Guardian-ML
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
cd backend
python main.py
```

---

## Usage Guide

### Step 1 — Upload Data

Navigate to **http://localhost:8000**, then:
- Drag and drop a CSV/JSON/GeoJSON/XLSX file, or click to browse
- GUARDIAN auto-detects: target column, latitude/longitude columns
- Inspect the schema preview and data quality warnings

**Minimum dataset requirements:**
- ≥ 10 rows
- At least one numeric feature column
- Optional: `latitude`, `longitude`, `label` columns (auto-detected by name)

### Step 2 — Preprocess

- Select scaling method (Standard recommended for general use)
- Override target column if auto-detection was incorrect
- Click **RUN PREPROCESSING**

### Step 3 — Train Models

- Select one or more models to train
- Click **BEGIN TRAINING** — cross-validation runs automatically
- Review the comparative metrics table

### Step 4 — Predict

- Select model (defaults to best by F1)
- Click **RUN PREDICTION**
- Review the risk dashboard (high / medium / low counts)

### Step 5 — Visualize

All 7 charts load automatically:
- **Risk Distribution** — histogram of composite risk scores
- **Model Comparison** — grouped bar chart of all metrics
- **Feature Importance** — top-20 features for best model
- **Confusion Matrix** — heatmap with true/predicted labels
- **Anomaly Timeline** — anomaly score series with flags
- **Cross-Validation Scores** — F1 mean ± std per model
- **Geospatial Risk Map** — scatter map (requires lat/lon)

---

## API Documentation

Interactive Swagger UI: **http://localhost:8000/api/docs**

### Key Endpoints

```
POST /upload/
    Body:  multipart/form-data { file }
    Returns: job_id, schema, inferred columns, warnings

POST /process/
    Body:  { job_id, target_col?, scaling? }
    Returns: feature_names, split sizes, preprocessing stats

POST /train/
    Body:  { job_id, models?: ["random_forest", "xgboost", ...] }
    Returns: training report, best model, per-model metrics

POST /predict/
    Body:  { job_id, model_name? }
    Returns: risk_scores, anomaly_flags, risk_summary

GET  /visualize/all?job_id=<id>
    Returns: all 7 Plotly figure JSON objects

GET  /api/docs
    Swagger UI (interactive)
```

### Example Workflow (curl)

```bash
# Upload
JOB=$(curl -s -X POST http://localhost:8000/upload/ \
  -F "file=@my_data.csv" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['job_id'])")

# Process
curl -s -X POST http://localhost:8000/process/ \
  -H "Content-Type: application/json" \
  -d "{\"job_id\": \"$JOB\", \"target_col\": \"label\"}"

# Train
curl -s -X POST http://localhost:8000/train/ \
  -H "Content-Type: application/json" \
  -d "{\"job_id\": \"$JOB\"}"

# Predict
curl -s -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d "{\"job_id\": \"$JOB\"}"

# Get all charts
curl -s "http://localhost:8000/visualize/all?job_id=$JOB" | python3 -m json.tool
```

---

## Configuration

All system parameters are controlled via `configs/config.yaml`:

```yaml
ml:
  random_seed: 42
  test_size: 0.2
  models:
    random_forest:
      enabled: true
      n_estimators: 200
    xgboost:
      enabled: true
      learning_rate: 0.05
    isolation_forest:
      contamination: 0.1
  preprocessing:
    scaling: standard        # standard | minmax | robust
    outlier_method: iqr      # iqr | zscore
```

---

## Running Tests

```bash
./run.sh test
```

Tests cover:
- Health check
- File upload (CSV, JSON, invalid formats)
- Preprocessing (all 3 scalers)
- Training (metric validity, model selection)
- Prediction (risk score bounds, anomaly flags)
- Visualization (Plotly figure structure)
- Utility functions (validators, risk engine, preprocessor)

---

## Deployment

### Development
```bash
./run.sh dev    # hot reload enabled
```

### Production
```bash
./run.sh start  # uvicorn with config from config.yaml
```

### Docker (optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r backend/requirements.txt
EXPOSE 8000
CMD ["python", "backend/main.py"]
```

### Scaling to Multi-Worker
Replace the in-memory `SessionStore` with Redis:
```python
# utils/session.py — swap to aioredis
import aioredis
redis = aioredis.from_url("redis://localhost")
```

---

## ML Methodology

See [`docs/methodology.md`](docs/methodology.md) for full technical detail including:
- Mathematical formulations for all algorithms
- Preprocessing pipeline equations
- Composite risk scoring derivation
- Evaluation protocol and reproducibility guarantees
- Academic references

---

## Repository Structure

```
Guardian-ML/
├── backend/
│   ├── main.py              FastAPI application
│   ├── requirements.txt     Dependencies
│   ├── api/                 Route handlers
│   ├── core/                ML pipeline + visualizer + risk engine
│   ├── data/                Preprocessor
│   └── utils/               Logger, helpers, session
├── frontend/
│   ├── index.html           Dashboard UI
│   ├── style.css            Dark cartographic theme
│   └── app.js               Pipeline controller
├── configs/config.yaml      System configuration
├── docs/                    Architecture + methodology
├── tests/test_api.py        Full test suite
├── run.sh                   Unified CLI script
└── README.md                This document
```

---

## License

MIT License — see `LICENSE` for details.

---

## Citation

If you use GUARDIAN ML in research, please cite:

```bibtex
@software{guardian_ml_2024,
  title     = {GUARDIAN ML: Geospatial Unified Anomaly Risk Detection \& Intelligence Analysis Network},
  version   = {1.0.0},
  year      = {2024},
  url       = {https://github.com/your-org/Guardian-ML}
}
```
