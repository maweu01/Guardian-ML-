"""
GUARDIAN ML — Test Suite
pytest + httpx async tests for all API endpoints.
"""

import io
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from httpx import AsyncClient, ASGITransport

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from main import app


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv_bytes() -> bytes:
    """Generate a realistic synthetic geospatial risk dataset."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "latitude":    np.random.uniform(-5,  5, n),
        "longitude":   np.random.uniform(30, 42, n),
        "temperature": np.random.normal(28, 5, n),
        "humidity":    np.random.uniform(40, 90, n),
        "elevation":   np.random.uniform(0, 2000, n),
        "population":  np.random.randint(500, 100000, n),
        "infrastructure_score": np.random.uniform(0, 1, n),
        "conflict_events": np.random.poisson(2, n).astype(float),
        "ndvi":        np.random.uniform(-0.1, 0.8, n),
        "label":       np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    # Inject a few NaN values for robustness testing
    df.loc[np.random.choice(n, 10, replace=False), "temperature"] = np.nan
    return df.to_csv(index=False).encode()


@pytest.fixture
def sample_csv_no_target() -> bytes:
    """CSV without a target column (for anomaly-only use-case)."""
    np.random.seed(99)
    n = 80
    df = pd.DataFrame({
        "latitude":   np.random.uniform(-5, 5, n),
        "longitude":  np.random.uniform(30, 42, n),
        "feature_a":  np.random.normal(0, 1, n),
        "feature_b":  np.random.uniform(0, 10, n),
    })
    return df.to_csv(index=False).encode()


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as c:
        yield c


# ─── Health ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    """API health check returns operational status."""
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "operational"
    assert body["system"] == "GUARDIAN ML"


@pytest.mark.asyncio
async def test_root(client):
    """Root endpoint returns HTML or JSON."""
    r = await client.get("/")
    assert r.status_code == 200


# ─── Upload ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_csv(client, sample_csv_bytes):
    """Upload a valid CSV returns job_id and schema."""
    r = await client.post(
        "/upload/",
        files={"file": ("test_dataset.csv", sample_csv_bytes, "text/csv")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert "job_id" in body["data"]
    assert body["data"]["schema"]["shape"][0] == 200
    assert body["data"]["schema"]["shape"][1] == 10
    assert body["data"]["inferred_target"] == "label"
    assert body["data"]["inferred_lat"]    == "latitude"
    assert body["data"]["inferred_lon"]    == "longitude"


@pytest.mark.asyncio
async def test_upload_unsupported_extension(client):
    """Uploading .txt should be rejected."""
    r = await client.post(
        "/upload/",
        files={"file": ("data.txt", b"col1,col2\n1,2", "text/plain")},
    )
    assert r.status_code == 415


@pytest.mark.asyncio
async def test_upload_json(client):
    """JSON upload is accepted and parsed."""
    records = [{"lat": 1.0, "lon": 36.0, "val": i, "label": i % 2} for i in range(50)]
    payload = json.dumps(records).encode()
    r = await client.post(
        "/upload/",
        files={"file": ("data.json", payload, "application/json")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["schema"]["shape"][0] == 50


@pytest.mark.asyncio
async def test_list_jobs(client, sample_csv_bytes):
    """List jobs returns at least one job after upload."""
    await client.post(
        "/upload/",
        files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
    )
    r = await client.get("/upload/jobs")
    assert r.status_code == 200
    assert isinstance(r.json()["data"]["jobs"], list)
    assert len(r.json()["data"]["jobs"]) >= 1


# ─── Process ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process(client, sample_csv_bytes):
    """Preprocessing pipeline returns valid splits and feature count."""
    up = await client.post(
        "/upload/",
        files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
    )
    job_id = up.json()["data"]["job_id"]

    r = await client.post("/process/", json={"job_id": job_id, "target_col": "label"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    d = body["data"]
    assert d["splits"]["train"] > 0
    assert d["splits"]["val"]   > 0
    assert d["splits"]["test"]  > 0
    assert len(d["feature_names"]) > 0
    assert "stats" in d


@pytest.mark.asyncio
async def test_process_missing_job(client):
    """Processing a non-existent job_id returns 404."""
    r = await client.post("/process/", json={"job_id": "nonexistent_job_xyz"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_process_scaling_variants(client, sample_csv_bytes):
    """All three scalers complete without error."""
    for scaler in ("standard", "minmax", "robust"):
        up = await client.post(
            "/upload/",
            files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
        )
        job_id = up.json()["data"]["job_id"]
        r = await client.post(
            "/process/",
            json={"job_id": job_id, "target_col": "label", "scaling": scaler},
        )
        assert r.status_code == 200, f"Failed for scaler={scaler}: {r.text}"


# ─── Train ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_train(client, sample_csv_bytes):
    """Full training pipeline completes with valid best model."""
    # Upload
    up = await client.post(
        "/upload/",
        files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
    )
    job_id = up.json()["data"]["job_id"]

    # Process
    await client.post("/process/", json={"job_id": job_id, "target_col": "label"})

    # Train (only RF and LR for speed in tests)
    r = await client.post(
        "/train/",
        json={"job_id": job_id, "models": ["random_forest", "logistic_regression"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    d = body["data"]
    assert d["best_model"] in ("random_forest", "logistic_regression")
    assert len(d["models_trained"]) == 2
    assert "results" in d
    for model_name in d["models_trained"]:
        assert "validation" in d["results"][model_name]
        val = d["results"][model_name]["validation"]
        assert "accuracy" in val
        assert "f1" in val
        assert 0.0 <= val["accuracy"] <= 1.0
        assert 0.0 <= val["f1"]       <= 1.0


@pytest.mark.asyncio
async def test_train_without_process(client, sample_csv_bytes):
    """Training without preprocessing returns 400."""
    up = await client.post(
        "/upload/",
        files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
    )
    job_id = up.json()["data"]["job_id"]
    r = await client.post("/train/", json={"job_id": job_id})
    assert r.status_code == 400


# ─── Predict ──────────────────────────────────────────────────────────────────

@pytest.fixture
async def trained_job(client, sample_csv_bytes):
    """Fixture: returns a fully trained job_id."""
    up = await client.post(
        "/upload/",
        files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
    )
    job_id = up.json()["data"]["job_id"]
    await client.post("/process/", json={"job_id": job_id, "target_col": "label"})
    await client.post(
        "/train/",
        json={"job_id": job_id, "models": ["random_forest", "logistic_regression"]},
    )
    return job_id


@pytest.mark.asyncio
async def test_predict(client, trained_job):
    """Prediction returns risk scores and anomaly flags."""
    r = await client.post("/predict/", json={"job_id": trained_job})
    assert r.status_code == 200
    body = r.json()
    d    = body["data"]
    assert len(d["predictions"])   > 0
    assert len(d["risk_scores"])   > 0
    assert len(d["anomaly_flags"]) > 0
    assert d["risk_summary"]["n_samples"] > 0
    for score in d["risk_scores"]:
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_predict_specific_model(client, trained_job):
    """Prediction with specified model name works."""
    r = await client.post(
        "/predict/",
        json={"job_id": trained_job, "model_name": "random_forest"},
    )
    assert r.status_code == 200
    assert r.json()["data"]["model_used"] == "random_forest"


# ─── Visualize ────────────────────────────────────────────────────────────────

@pytest.fixture
async def predicted_job(client, trained_job):
    """Fixture: fully trained + predicted job."""
    await client.post("/predict/", json={"job_id": trained_job})
    return trained_job


@pytest.mark.asyncio
async def test_viz_risk_distribution(client, predicted_job):
    """Risk distribution chart returns valid Plotly figure."""
    r = await client.get(f"/visualize/risk-distribution?job_id={predicted_job}")
    assert r.status_code == 200
    fig = r.json()["data"]["figure"]
    assert "data" in fig
    assert "layout" in fig


@pytest.mark.asyncio
async def test_viz_model_comparison(client, predicted_job):
    r = await client.get(f"/visualize/model-comparison?job_id={predicted_job}")
    assert r.status_code == 200
    assert "figure" in r.json()["data"]


@pytest.mark.asyncio
async def test_viz_feature_importance(client, predicted_job):
    r = await client.get(f"/visualize/feature-importance?job_id={predicted_job}")
    assert r.status_code == 200
    assert "figure" in r.json()["data"]


@pytest.mark.asyncio
async def test_viz_confusion_matrix(client, predicted_job):
    r = await client.get(f"/visualize/confusion-matrix?job_id={predicted_job}")
    assert r.status_code == 200
    assert "figure" in r.json()["data"]


@pytest.mark.asyncio
async def test_viz_all(client, predicted_job):
    """All charts bundle endpoint returns multiple figures."""
    r = await client.get(f"/visualize/all?job_id={predicted_job}")
    assert r.status_code == 200
    charts = r.json()["data"]["charts"]
    assert "risk_distribution" in charts
    assert "model_comparison"  in charts
    assert "feature_importance" in charts


# ─── Data Utility Tests ───────────────────────────────────────────────────────

def test_helpers_validate_dataframe():
    from utils.helpers import validate_dataframe
    df_good = pd.DataFrame({"a": range(50), "b": range(50)})
    assert validate_dataframe(df_good) == []

    df_empty = pd.DataFrame()
    issues = validate_dataframe(df_empty)
    assert any("empty" in i.lower() for i in issues)


def test_helpers_infer_columns():
    from utils.helpers import infer_target_column, infer_coordinate_columns
    df = pd.DataFrame({"latitude": [1], "longitude": [2], "label": [0]})
    assert infer_target_column(df) == "label"
    lat, lon = infer_coordinate_columns(df)
    assert lat == "latitude"
    assert lon == "longitude"


def test_preprocessor_pipeline():
    from data.preprocessor import GuardianPreprocessor
    import yaml
    cfg_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "latitude":  np.random.uniform(-5, 5, n),
        "longitude": np.random.uniform(30, 42, n),
        "feature_x": np.random.normal(0, 1, n),
        "feature_y": np.random.uniform(0, 10, n),
        "label":     np.random.choice([0, 1], n),
    })

    p = GuardianPreprocessor(cfg)
    result = p.fit_transform(df, target_col="label")

    assert result["X_train"].shape[0] > 0
    assert result["X_val"].shape[0]   > 0
    assert result["X_test"].shape[0]  > 0
    assert len(result["feature_names"]) > 0
    assert "stats" in result


def test_risk_engine():
    from core.risk_engine import RiskScorer
    scorer = RiskScorer({})
    scores = np.array([0.1, 0.4, 0.8, 0.9, 0.2])
    composite = scorer.compute(model_probs=scores)
    assert composite.shape == (5,)
    assert all(0.0 <= v <= 1.0 for v in composite)

    levels = scorer.classify(composite)
    assert all(l in ("low", "medium", "high") for l in levels)

    summary = scorer.summarize(composite)
    assert summary["n_samples"] == 5
    assert "mean_risk" in summary
    assert "level_counts" in summary
