"""
GUARDIAN ML — /visualize API Router
Generates and serves all visualization artifacts.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List

import numpy as np
import pandas as pd
import json
import yaml

from core.visualizer import (
    plot_risk_distribution,
    plot_model_comparison,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_geospatial_risk,
    plot_anomaly_timeline,
    plot_cv_scores,
    export_figure_base64,
)
from core.risk_engine import risk_score_dataframe, RiskScorer
from utils.helpers import load_dataframe, success_response, NumpyEncoder
from utils.session import session
from utils.logger import setup_logger

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

logger = setup_logger("guardian.api.visualize")
router = APIRouter()


def _require_job(job_id: str, *keys: str):
    """Raise 404/400 if job or required keys are missing."""
    if not session.exists(job_id):
        raise HTTPException(404, f"Job '{job_id}' not found.")
    for key in keys:
        if session.get(job_id, key) is None:
            raise HTTPException(400, f"Missing data: '{key}'. Check pipeline order.")


@router.get("/risk-distribution", summary="Risk score histogram")
async def viz_risk_distribution(job_id: str = Query(...)):
    _require_job(job_id, "predictions")
    predictions = session.get(job_id, "predictions")
    fig = plot_risk_distribution(predictions["risk_scores"])
    return JSONResponse(success_response(data={"figure": fig}, job_id=job_id))


@router.get("/model-comparison", summary="Multi-model metric comparison")
async def viz_model_comparison(job_id: str = Query(...)):
    _require_job(job_id, "report")
    report  = session.get(job_id, "report")
    results = report.get("results", {})
    fig     = plot_model_comparison(results)
    return JSONResponse(success_response(data={"figure": fig}, job_id=job_id))


@router.get("/feature-importance", summary="Feature importance chart")
async def viz_feature_importance(
    job_id:     str = Query(...),
    model_name: Optional[str] = Query(None),
    top_n:      int = Query(15, ge=5, le=30),
):
    _require_job(job_id, "report")
    report  = session.get(job_id, "report")
    results = report.get("results", {})
    name    = model_name or report.get("best_model", "random_forest")

    if name not in results:
        raise HTTPException(404, f"Model '{name}' not in results.")

    importance = results[name].get("feature_importance", [])
    fig = plot_feature_importance(importance, model_name=name, top_n=top_n)
    return JSONResponse(success_response(data={"figure": fig}, job_id=job_id))


@router.get("/confusion-matrix", summary="Confusion matrix heatmap")
async def viz_confusion_matrix(
    job_id:     str = Query(...),
    model_name: Optional[str] = Query(None),
):
    _require_job(job_id, "report")
    report  = session.get(job_id, "report")
    results = report.get("results", {})
    name    = model_name or report.get("best_model", "random_forest")

    if name not in results:
        raise HTTPException(404, f"Model '{name}' not in results.")

    cm = results[name].get("validation", {}).get("confusion_matrix", [[]])
    fig = plot_confusion_matrix(cm)
    return JSONResponse(success_response(data={"figure": fig}, job_id=job_id))


@router.get("/geospatial", summary="Geospatial risk map")
async def viz_geospatial(job_id: str = Query(...)):
    _require_job(job_id, "predictions", "preprocess_result")

    upload_path = session.get(job_id, "upload_path")
    predictions = session.get(job_id, "predictions")
    lat_col     = session.get(job_id, "lat_col")
    lon_col     = session.get(job_id, "lon_col")

    if not upload_path or not lat_col or not lon_col:
        raise HTTPException(
            400,
            "Geospatial map requires lat/lon columns in the dataset. "
            "Ensure your data has columns named 'latitude'/'longitude' (or similar)."
        )

    try:
        df   = load_dataframe(upload_path)
        prep = session.get(job_id, "preprocessor")
        X    = prep.transform(df)
        trainer = session.get(job_id, "trainer")
        preds = trainer.predict(X)

        scored_df = risk_score_dataframe(
            df, preds,
            lat_col=lat_col,
            lon_col=lon_col,
            config=CONFIG,
        )

        fig = plot_geospatial_risk(
            scored_df,
            lat_col=lat_col,
            lon_col=lon_col,
            risk_col="guardian_risk_score",
        )

        return JSONResponse(
            json.loads(json.dumps(
                success_response(data={"figure": fig}, job_id=job_id),
                cls=NumpyEncoder,
            ))
        )

    except Exception as e:
        logger.error(f"[{job_id}] Geospatial viz failed: {e}")
        raise HTTPException(500, detail=str(e))


@router.get("/anomaly-timeline", summary="Anomaly detection timeline")
async def viz_anomaly_timeline(job_id: str = Query(...)):
    _require_job(job_id, "predictions")
    predictions = session.get(job_id, "predictions")

    if "anomaly_scores" not in predictions:
        raise HTTPException(400, "No anomaly scores available. Ensure IsolationForest is enabled.")

    fig = plot_anomaly_timeline(
        scores=predictions["anomaly_scores"],
        flags=predictions.get("anomaly_flags", [False] * len(predictions["anomaly_scores"])),
    )
    return JSONResponse(success_response(data={"figure": fig}, job_id=job_id))


@router.get("/cv-scores", summary="Cross-validation score comparison")
async def viz_cv_scores(job_id: str = Query(...)):
    _require_job(job_id, "report")
    report  = session.get(job_id, "report")
    results = report.get("results", {})
    fig = plot_cv_scores(results)
    return JSONResponse(success_response(data={"figure": fig}, job_id=job_id))


@router.get("/all", summary="Return all available visualization figures")
async def viz_all(job_id: str = Query(...)):
    """Returns all generated charts as a bundle for the frontend dashboard."""
    _require_job(job_id, "predictions", "report")

    report      = session.get(job_id, "report")
    predictions = session.get(job_id, "predictions")
    results     = report.get("results", {})
    best_model  = report.get("best_model", "")

    charts = {}

    charts["risk_distribution"] = plot_risk_distribution(predictions["risk_scores"])
    charts["model_comparison"]  = plot_model_comparison(results)
    charts["cv_scores"]         = plot_cv_scores(results)

    if best_model in results:
        charts["feature_importance"] = plot_feature_importance(
            results[best_model].get("feature_importance", []), model_name=best_model
        )
        cm = results[best_model].get("validation", {}).get("confusion_matrix", [])
        if cm:
            charts["confusion_matrix"] = plot_confusion_matrix(cm)

    if "anomaly_scores" in predictions:
        charts["anomaly_timeline"] = plot_anomaly_timeline(
            scores=predictions["anomaly_scores"],
            flags=predictions.get("anomaly_flags", []),
        )

    return JSONResponse(
        json.loads(json.dumps(
            success_response(
                data={"charts": charts, "best_model": best_model},
                job_id=job_id,
            ),
            cls=NumpyEncoder,
        ))
    )
