"""
GUARDIAN ML — /process API Router
Triggers preprocessing pipeline on an uploaded dataset.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

import yaml
from data.preprocessor import GuardianPreprocessor
from utils.helpers import load_dataframe, success_response
from utils.session import session
from utils.logger import setup_logger

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

logger = setup_logger("guardian.api.process")
router = APIRouter()


class ProcessRequest(BaseModel):
    job_id:     str = Field(..., description="Job ID from /upload")
    target_col: Optional[str] = Field(None, description="Target/label column name (override auto-detect)")
    scaling:    Optional[str] = Field(None, description="Scaling method: standard | minmax | robust")


@router.post("/", summary="Run preprocessing pipeline on uploaded data")
async def process_data(req: ProcessRequest):
    """
    Preprocesses the uploaded dataset:
    1. Type coercion
    2. Feature engineering (geospatial + temporal)
    3. Categorical encoding
    4. Outlier handling
    5. Imputation
    6. Scaling
    7. Train / val / test split
    """
    job_id = req.job_id
    if not session.exists(job_id):
        raise HTTPException(404, f"Job '{job_id}' not found. Please upload a file first.")

    upload_path = session.get(job_id, "upload_path")
    if not upload_path:
        raise HTTPException(400, "No uploaded file found for this job.")

    target_col = req.target_col or session.get(job_id, "target_col")

    try:
        df = load_dataframe(upload_path)
        logger.info(f"[{job_id}] Processing {df.shape} — target: {target_col}")

        # Override config scaling if provided
        cfg = CONFIG.copy()
        if req.scaling:
            cfg.setdefault("ml", {}).setdefault("preprocessing", {})["scaling"] = req.scaling

        preprocessor = GuardianPreprocessor(cfg)
        result = preprocessor.fit_transform(df, target_col=target_col)

        # Persist preprocessor and split metadata
        session.set(job_id, "preprocessor",   preprocessor)
        session.set(job_id, "target_col",     target_col)
        session.set(job_id, "feature_names",  result["feature_names"])
        session.set(job_id, "splits",         {
            "X_train_shape": list(result["X_train"].shape),
            "X_val_shape":   list(result["X_val"].shape),
            "X_test_shape":  list(result["X_test"].shape),
        })
        session.set(job_id, "preprocess_result", result)

        logger.info(
            f"[{job_id}] Preprocessing done. "
            f"Features: {len(result['feature_names'])}, "
            f"Train: {result['X_train'].shape[0]}"
        )

        return JSONResponse(success_response(
            data={
                "job_id":        job_id,
                "feature_names": result["feature_names"],
                "splits": {
                    "train": int(result["X_train"].shape[0]),
                    "val":   int(result["X_val"].shape[0]),
                    "test":  int(result["X_test"].shape[0]),
                },
                "stats":         result["stats"],
                "warnings":      result["warnings"],
            },
            message="Preprocessing complete.",
            job_id=job_id,
        ))

    except Exception as e:
        logger.error(f"[{job_id}] Preprocessing failed: {e}")
        raise HTTPException(500, detail=str(e))
