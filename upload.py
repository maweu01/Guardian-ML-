"""
GUARDIAN ML — /upload API Router
Handles file ingestion, validation, and schema inspection.
"""

import shutil
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

import yaml
from utils.helpers import (
    generate_job_id, load_dataframe, validate_dataframe,
    infer_target_column, infer_coordinate_columns, success_response, error_response,
)
from utils.session import session
from utils.logger import setup_logger

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

logger   = setup_logger("guardian.api.upload")
router   = APIRouter()
UPLOAD_DIR = Path(CONFIG["data"]["upload_dir"])
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = set(CONFIG["data"]["allowed_extensions"])
MAX_SIZE_MB  = CONFIG["data"]["max_file_size_mb"]


@router.post("/", summary="Upload a dataset (CSV / JSON / GeoJSON / XLSX)")
async def upload_file(
    file: UploadFile = File(...),
    job_id: str = Query(default=None, description="Existing job ID to attach to"),
):
    """
    Upload a dataset file. Returns a job_id and schema summary.

    - Validates file extension and size.
    - Parses into DataFrame.
    - Infers target and coordinate columns.
    - Returns column list, dtypes, shape, and data quality warnings.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {suffix}. Allowed: {list(ALLOWED_EXTS)}",
        )

    jid = job_id or generate_job_id()
    save_path = UPLOAD_DIR / f"{jid}{suffix}"

    try:
        contents = await file.read()
        if len(contents) > MAX_SIZE_MB * 1024 * 1024:
            raise HTTPException(413, f"File exceeds {MAX_SIZE_MB} MB limit.")

        with open(save_path, "wb") as f_out:
            f_out.write(contents)

        df = load_dataframe(save_path)

        warnings = validate_dataframe(df)
        target   = infer_target_column(df)
        lat, lon = infer_coordinate_columns(df)

        schema = {
            "columns": list(df.columns),
            "dtypes":  {col: str(dt) for col, dt in df.dtypes.items()},
            "shape":   list(df.shape),
            "missing_pct": df.isnull().mean().round(4).to_dict(),
            "sample":  df.head(5).fillna("").to_dict(orient="records"),
        }

        session.set(jid, "upload_path",  str(save_path))
        session.set(jid, "filename",     file.filename)
        session.set(jid, "target_col",   target)
        session.set(jid, "lat_col",      lat)
        session.set(jid, "lon_col",      lon)
        session.set(jid, "schema",       schema)

        logger.info(f"[{jid}] Uploaded {file.filename} → {df.shape}")

        return JSONResponse(success_response(
            data={
                "job_id":         jid,
                "filename":       file.filename,
                "schema":         schema,
                "inferred_target":target,
                "inferred_lat":   lat,
                "inferred_lon":   lon,
                "warnings":       warnings,
            },
            message="File uploaded and parsed successfully.",
            job_id=jid,
        ))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}")
        raise HTTPException(500, detail=str(e))


@router.get("/jobs", summary="List active job IDs")
async def list_jobs():
    return success_response(data={"jobs": session.list_jobs()})


@router.get("/job/{job_id}", summary="Get job metadata")
async def get_job(job_id: str):
    job = session.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    safe_job = {k: v for k, v in job.items() if k not in ("preprocessor", "trainer")}
    return success_response(data=safe_job, job_id=job_id)
