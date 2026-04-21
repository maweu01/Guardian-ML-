# GUARDIAN ML — System Architecture

**Version:** 1.0.0  
**Classification:** Technical Reference Document  
**Standard:** Production Software Architecture

---

## 1. System Overview

GUARDIAN ML (Geospatial Unified Anomaly Risk Detection & Intelligence Analysis Network) is a production-grade, end-to-end machine learning system for geospatial risk assessment, anomaly detection, and predictive analytics.

The system ingests multi-format geospatial datasets, applies a rigorous preprocessing pipeline, trains an ensemble of supervised classifiers and an unsupervised anomaly detector, and produces composite risk scores with full visual intelligence output.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GUARDIAN ML SYSTEM                               │
│                                                                           │
│  ┌──────────────┐    ┌───────────────────────────────────────────────┐   │
│  │   FRONTEND   │    │                  BACKEND                       │   │
│  │              │    │                                               │   │
│  │  index.html  │◄──►│  FastAPI Application (main.py)               │   │
│  │  style.css   │    │                                               │   │
│  │  app.js      │    │  ┌─────────┐  ┌──────────┐  ┌────────────┐  │   │
│  │              │    │  │  /upload│  │ /process │  │  /train    │  │   │
│  │  Plotly.js   │    │  │  Router │  │  Router  │  │  Router    │  │   │
│  │  (charts)    │    │  └────┬────┘  └────┬─────┘  └─────┬──────┘  │   │
│  └──────────────┘    │       │             │               │          │   │
│                       │  ┌────▼─────────────▼───────────────▼──────┐ │   │
│                       │  │          CORE ML PIPELINE                │ │   │
│                       │  │                                          │ │   │
│                       │  │  GuardianPreprocessor                    │ │   │
│                       │  │  ├─ Type coercion                       │ │   │
│                       │  │  ├─ Feature engineering (geo/temporal)  │ │   │
│                       │  │  ├─ Outlier handling (IQR/Z-score)      │ │   │
│                       │  │  ├─ Imputation (median/mean/mode)       │ │   │
│                       │  │  └─ Scaling (Standard/MinMax/Robust)    │ │   │
│                       │  │                                          │ │   │
│                       │  │  GuardianTrainer                         │ │   │
│                       │  │  ├─ Random Forest (sklearn)             │ │   │
│                       │  │  ├─ XGBoost                             │ │   │
│                       │  │  ├─ Logistic Regression                 │ │   │
│                       │  │  ├─ Isolation Forest (anomaly)          │ │   │
│                       │  │  └─ StratifiedKFold CV                  │ │   │
│                       │  │                                          │ │   │
│                       │  │  RiskScorer                              │ │   │
│                       │  │  ├─ Model probability (50%)             │ │   │
│                       │  │  ├─ Anomaly score (25%)                 │ │   │
│                       │  │  ├─ Spatial KDE density (15%)           │ │   │
│                       │  │  └─ Temporal recency weight (10%)       │ │   │
│                       │  └──────────────────────────────────────────┘ │   │
│                       │                                               │   │
│                       │  ┌──────────────┐  ┌──────────────────────┐  │   │
│                       │  │  SessionStore │  │   Visualization      │  │   │
│                       │  │  (in-memory)  │  │   Engine (Plotly)    │  │   │
│                       │  └──────────────┘  └──────────────────────┘  │   │
│                       └───────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
Guardian-ML/
│
├── backend/                    Python backend application
│   ├── main.py                 FastAPI app entrypoint, router registration
│   ├── requirements.txt        Pinned Python dependencies
│   │
│   ├── api/                    RESTful API layer
│   │   ├── upload.py           POST /upload/         — file ingest
│   │   ├── process.py          POST /process/        — preprocessing
│   │   ├── train.py            POST /train/          — model training
│   │   ├── predict.py          POST /predict/        — inference
│   │   └── visualize.py        GET  /visualize/*     — chart generation
│   │
│   ├── core/                   Core ML and analytics engine
│   │   ├── ml_pipeline.py      GuardianTrainer, model registry, CV
│   │   ├── risk_engine.py      RiskScorer, composite scoring, KDE
│   │   └── visualizer.py       Plotly chart generators (dark theme)
│   │
│   ├── data/                   Data handling layer
│   │   └── preprocessor.py     GuardianPreprocessor (10-step pipeline)
│   │
│   ├── models/                 Saved model artifacts (joblib .pkl)
│   │   └── saved/              Per-job subdirectories
│   │
│   └── utils/                  Cross-cutting utilities
│       ├── logger.py           Loguru structured logging
│       ├── helpers.py          Validators, encoders, file I/O
│       └── session.py          Thread-safe in-memory job store
│
├── frontend/                   Web UI (pure HTML/CSS/JS)
│   ├── index.html              Application shell
│   ├── style.css               Dark cartographic theme
│   └── app.js                  Pipeline controller, API client
│
├── configs/
│   └── config.yaml             Full system configuration
│
├── docs/
│   ├── system_architecture.md  This document
│   └── methodology.md          ML methodology and algorithms
│
├── tests/
│   └── test_api.py             pytest + httpx async test suite
│
├── data/                       Runtime data directories (gitignored)
│   ├── uploads/                Uploaded files
│   ├── processed/              Processed datasets
│   └── exports/                Generated reports
│
├── logs/                       Application logs (gitignored)
├── run.sh                      Unified run/install/test script
└── README.md                   Project documentation
```

---

## 4. API Endpoints

| Method | Path                           | Description                           |
|--------|--------------------------------|---------------------------------------|
| GET    | `/health`                      | System health check                   |
| GET    | `/`                            | Serve frontend HTML                   |
| POST   | `/upload/`                     | Upload CSV / JSON / GeoJSON / XLSX    |
| GET    | `/upload/jobs`                 | List active job IDs                   |
| GET    | `/upload/job/{job_id}`         | Get job metadata                      |
| POST   | `/process/`                    | Run preprocessing pipeline            |
| POST   | `/train/`                      | Train models + cross-validation       |
| GET    | `/train/status/{job_id}`       | Get training report                   |
| POST   | `/predict/`                    | Predict on test split                 |
| POST   | `/predict/upload`              | Predict on new uploaded file          |
| GET    | `/visualize/risk-distribution` | Risk score histogram                  |
| GET    | `/visualize/model-comparison`  | Multi-model metric comparison         |
| GET    | `/visualize/feature-importance`| Feature importance chart              |
| GET    | `/visualize/confusion-matrix`  | Confusion matrix heatmap              |
| GET    | `/visualize/geospatial`        | Geospatial risk scatter map           |
| GET    | `/visualize/anomaly-timeline`  | Anomaly detection timeline            |
| GET    | `/visualize/cv-scores`         | Cross-validation score bands          |
| GET    | `/visualize/all`               | All charts bundle                     |
| GET    | `/api/docs`                    | Swagger UI interactive documentation  |

---

## 5. Data Flow

```
User Upload (CSV/JSON/GeoJSON/XLSX)
        │
        ▼
   File Validation & Schema Inspection
   (extension check, size limit, dtype inference)
        │
        ▼
   GuardianPreprocessor.fit_transform()
   ├── Duplicate removal
   ├── Type coercion
   ├── Geospatial feature extraction (if lat/lon present)
   ├── Temporal feature extraction (if datetime columns present)
   ├── Categorical label encoding
   ├── Outlier handling (IQR clipping)
   ├── Median imputation
   ├── Standard / MinMax / Robust scaling
   └── Stratified 70/10/20 train/val/test split
        │
        ▼
   GuardianTrainer.train()
   ├── RandomForestClassifier (sklearn)
   ├── XGBClassifier (xgboost)
   ├── LogisticRegression (sklearn)
   ├── IsolationForest (anomaly, unsupervised)
   └── StratifiedKFold(5) cross-validation for all
        │
        ▼
   Evaluation & Model Selection
   (F1-weighted on validation set → best model selected)
        │
        ▼
   RiskScorer.compute()
   ├── Model probability component (50%)
   ├── Anomaly score component (25%)
   ├── Spatial KDE density index (15%)
   └── Temporal recency weight (10%)
        │
        ▼
   Composite Risk Score [0, 1]
   ├── LOW    < 0.33
   ├── MEDIUM 0.33 – 0.66
   └── HIGH   > 0.66
        │
        ▼
   Plotly Visualization Engine
   └── 7 chart types → JSON → Frontend render
```

---

## 6. Technology Stack

| Layer          | Technology              | Version    |
|----------------|-------------------------|------------|
| API Framework  | FastAPI                 | 0.111.0    |
| ASGI Server    | Uvicorn                 | 0.29.0     |
| ML Core        | scikit-learn            | 1.4.2      |
| Gradient Boost | XGBoost                 | 2.0.3      |
| Geospatial     | GeoPandas, Shapely      | 0.14.4     |
| Visualization  | Plotly                  | 5.22.0     |
| Logging        | Loguru                  | 0.7.2      |
| Config         | PyYAML                  | 6.0.1      |
| Data           | pandas, numpy           | 2.2.2      |
| Testing        | pytest + httpx          | 8.2.0      |
| Frontend       | HTML5 / CSS3 / ES6 JS   | Native     |
| Chart Client   | Plotly.js               | 2.32.0     |

---

## 7. Session Management

The system uses a thread-safe in-memory `SessionStore` (singleton) to track job state across API calls:

```
job_id → {
    upload_path:       str
    filename:          str
    target_col:        str | None
    lat_col:           str | None
    lon_col:           str | None
    schema:            dict
    preprocessor:      GuardianPreprocessor
    preprocess_result: dict (X_train, X_val, X_test, y_*, features)
    trainer:           GuardianTrainer
    report:            dict
    predictions:       dict
    risk_summary:      dict
    models_dir:        str
    best_model:        str
}
```

**Scaling note:** For multi-worker production deployments, replace `SessionStore` with a Redis backend using `aioredis`. The interface contract (`.get()`, `.set()`, `.exists()`) is designed for drop-in replacement.

---

## 8. Failure Modes & Mitigations

| Failure                       | Cause                                 | Mitigation                                      |
|-------------------------------|---------------------------------------|-------------------------------------------------|
| Upload rejected               | Wrong extension / file > 100 MB       | HTTP 415 / 413 with clear error message         |
| Preprocessing crash           | All-NaN column or zero-variance        | Constant column removal + IQR clipping          |
| Training fails (no target)    | Missing/misspelled target column      | HTTP 400 with guidance message                  |
| XGBoost import error          | xgboost not installed                 | `pip install -r requirements.txt`               |
| Geospatial map empty          | No lat/lon columns in dataset         | HTTP 400 with explanation                       |
| Session lost (server restart) | In-memory store cleared               | Re-upload and re-process; use Redis for persist |
| Chart render fails            | Kaleido missing (for export)          | Install `kaleido==0.2.1`                        |
