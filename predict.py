"""
GUARDIAN ML — /predict API Router
Runs inference, risk scoring, and returns structured predictions.
"""

from pathlib import Path
from io import StringIO
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List

import numpy as np
import pandas as pd
import yaml

from core.risk_engine import risk_score_dataframe, RiskScorer
from utils.helpers import load_dataframe, success_response, NumpyEncoder
from utils.session import session
from utils.logger import setup_logger

import json

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

logger = setup_logger("guardian.api.predict")
router = APIRouter()


class PredictRequest(BaseModel):
    job_id:     str            = Field(..., description="Job ID with trained models")
    model_name: Optional[str] = Field(None, description="Specific model to use (defaults to best)")


@router.post("/", summary="Run prediction on test split (trained job)")
async def predict_test_set(req: PredictRequest):
    """
    Run predictions on the held-out test set from a trained job.
    Returns risk scores, anomaly flags, and a summary.
    """
    job_id = req.job_id
    if not session.exists(job_id):
        raise HTTPException(404, f"Job '{job_id}' not found.")

    trainer = session.get(job_id, "trainer")
    if not trainer:
        raise HTTPException(400, "No trained model found. Run /train first.")

    prep_result = session.get(job_id, "preprocess_result")
    X_test = prep_result.get("X_test")
    y_test = prep_result.get("y_test")

    if X_test is None:
        raise HTTPException(400, "No test split available.")

    try:
        predictions = trainer.predict(X_test, model_name=req.model_name)
        scorer = RiskScorer(CONFIG)
        risk_scores = np.array(predictions["risk_scores"])
        summary = scorer.summarize(risk_scores)

        session.set(job_id, "predictions", predictions)
        session.set(job_id, "risk_summary", summary)

        logger.info(
            f"[{job_id}] Prediction: {len(X_test)} samples, "
            f"high-risk: {summary['high_risk_count']}"
        )

        return JSONResponse(
            json.loads(json.dumps(
                success_response(
                    data={
                        "job_id":      job_id,
                        "model_used":  predictions["model_used"],
                        "n_samples":   predictions["n_samples"],
                        "predictions": predictions["predictions"],
                        "risk_scores": predictions["risk_scores"],
                        "anomaly_flags": predictions.get("anomaly_flags", []),
                        "risk_summary":  summary,
                    },
                    message="Prediction complete.",
                    job_id=job_id,
                ),
                cls=NumpyEncoder,
            ))
        )

    except Exception as e:
        logger.error(f"[{job_id}] Prediction failed: {e}")
        raise HTTPException(500, detail=str(e))


@router.post("/upload", summary="Predict on a new uploaded file")
async def predict_new_file(
    file: UploadFile = File(...),
    job_id: str = Query(..., description="Job ID with trained model and preprocessor"),
    model_name: Optional[str] = Query(None),
):
    """
    Upload a new CSV/JSON file and run predictions using a previously trained model.
    Applies the same preprocessor fitted during /process.
    """
    if not session.exists(job_id):
        raise HTTPException(404, f"Job '{job_id}' not found.")

    trainer     = session.get(job_id, "trainer")
    preprocessor= session.get(job_id, "preprocessor")

    if not trainer:
        raise HTTPException(400, "No trained model. Run /train first.")
    if not preprocessor:
        raise HTTPException(400, "No fitted preprocessor. Re-run /process.")

    try:
        contents = await file.read()
        suffix   = Path(file.filename).suffix.lower()

        tmp_path = Path("/tmp") / f"{job_id}_new{suffix}"
        with open(tmp_path, "wb") as f_out:
            f_out.write(contents)

        df = load_dataframe(tmp_path)
        lat_col = session.get(job_id, "lat_col")
        lon_col = session.get(job_id, "lon_col")

        X_new = preprocessor.transform(df)
        predictions = trainer.predict(X_new, model_name=model_name)

        scored_df = risk_score_dataframe(
            df, predictions,
            lat_col=lat_col,
            lon_col=lon_col,
            config=CONFIG,
        )

        scorer  = RiskScorer(CONFIG)
        summary = scorer.summarize(np.array(predictions["risk_scores"]))

        records = json.loads(scored_df.to_json(orient="records", default_handler=str))

        return JSONResponse(
            json.loads(json.dumps(
                success_response(
                    data={
                        "job_id":       job_id,
                        "model_used":   predictions["model_used"],
                        "n_samples":    int(len(df)),
                        "risk_summary": summary,
                        "records":      records[:500],  # Cap for API response
                    },
                    message="New file predicted.",
                    job_id=job_id,
                ),
                cls=NumpyEncoder,
            ))
        )

    except Exception as e:
        logger.error(f"[{job_id}] New-file prediction failed: {e}")
        raise HTTPException(500, detail=str(e))
