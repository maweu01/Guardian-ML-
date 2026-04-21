"""
GUARDIAN ML — /train API Router
Triggers multi-model training, cross-validation, and saves artifacts.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List

import yaml
from core.ml_pipeline import GuardianTrainer
from utils.helpers import success_response, ensure_dirs
from utils.session import session
from utils.logger import setup_logger

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

logger = setup_logger("guardian.api.train")
router = APIRouter()


class TrainRequest(BaseModel):
    job_id:     str   = Field(..., description="Job ID with preprocessed data")
    models:     Optional[List[str]] = Field(
        None,
        description="Model subset to train (e.g. ['random_forest', 'xgboost']). "
                    "Defaults to all enabled models."
    )


@router.post("/", summary="Train ML models on preprocessed data")
async def train_models(req: TrainRequest):
    """
    Trains all (or selected) models:
    - Random Forest
    - XGBoost
    - Logistic Regression
    - IsolationForest (anomaly detection)

    Runs stratified K-fold cross-validation.
    Persists all models to disk.
    Returns comparative metrics for all models.
    """
    job_id = req.job_id
    if not session.exists(job_id):
        raise HTTPException(404, f"Job '{job_id}' not found.")

    prep_result = session.get(job_id, "preprocess_result")
    if prep_result is None:
        raise HTTPException(400, "No preprocessed data found. Run /process first.")

    X_train = prep_result["X_train"]
    X_val   = prep_result["X_val"]
    feat    = prep_result["feature_names"]

    y_train = prep_result.get("y_train")
    y_val   = prep_result.get("y_val")

    if y_train is None or y_val is None:
        raise HTTPException(
            400,
            "No target column found in preprocessed data. "
            "Re-run /process with a valid target_col."
        )

    # Optionally override which models are enabled
    cfg = CONFIG.copy()
    if req.models:
        for model_key in cfg.get("ml", {}).get("models", {}):
            cfg["ml"]["models"][model_key]["enabled"] = model_key in req.models

    try:
        trainer = GuardianTrainer(cfg)
        report  = trainer.train(X_train, y_train, X_val, y_val, feat)

        # Save to disk
        save_dir = Path(cfg["ml"]["models_dir"]) / job_id
        ensure_dirs(str(save_dir))
        trainer.save(str(save_dir))

        # Save preprocessor alongside models
        preprocessor = session.get(job_id, "preprocessor")
        if preprocessor:
            preprocessor.save(str(save_dir))

        session.set(job_id, "trainer",      trainer)
        session.set(job_id, "report",       report)
        session.set(job_id, "models_dir",   str(save_dir))
        session.set(job_id, "best_model",   report.get("best_model"))

        logger.info(
            f"[{job_id}] Training complete. "
            f"Best: {report.get('best_model')}"
        )

        return JSONResponse(success_response(
            data=report,
            message="Training complete.",
            job_id=job_id,
        ))

    except Exception as e:
        logger.error(f"[{job_id}] Training failed: {e}")
        raise HTTPException(500, detail=str(e))


@router.get("/status/{job_id}", summary="Get training status and report")
async def training_status(job_id: str):
    report = session.get(job_id, "report")
    if report is None:
        raise HTTPException(404, f"No training report for job '{job_id}'.")
    return success_response(data=report, job_id=job_id)
