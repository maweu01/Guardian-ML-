"""
GUARDIAN ML — General Utilities
Shared helper functions: validation, file I/O, response builders.
"""

import json
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ─── Type Converters ─────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy dtypes."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def safe_json(data: Any) -> str:
    """Serialize data to JSON safely, handling numpy types."""
    return json.dumps(data, cls=NumpyEncoder, indent=2)


# ─── Unique ID / Hashing ─────────────────────────────────────────────────────

def generate_job_id() -> str:
    """Generate a unique job identifier."""
    return f"guardian_{uuid.uuid4().hex[:12]}"


def file_hash(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── File I/O ─────────────────────────────────────────────────────────────────

def ensure_dirs(*paths: Union[str, Path]) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_dataframe(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a DataFrame from CSV, JSON, GeoJSON, or XLSX.
    Raises ValueError for unsupported formats.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    loaders = {
        ".csv":     lambda p: pd.read_csv(p),
        ".json":    lambda p: pd.read_json(p),
        ".geojson": lambda p: _load_geojson_df(p),
        ".xlsx":    lambda p: pd.read_excel(p),
        ".xls":     lambda p: pd.read_excel(p),
    }

    if suffix not in loaders:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: {list(loaders)}")

    return loaders[suffix](path)


def _load_geojson_df(filepath: Path) -> pd.DataFrame:
    """Flatten a GeoJSON FeatureCollection into a DataFrame."""
    with open(filepath) as f:
        gj = json.load(f)

    rows = []
    for feature in gj.get("features", []):
        row = feature.get("properties", {}).copy()
        geom = feature.get("geometry", {})
        if geom.get("type") == "Point":
            coords = geom.get("coordinates", [None, None])
            row["longitude"] = coords[0]
            row["latitude"]  = coords[1]
        rows.append(row)

    return pd.DataFrame(rows)


# ─── Response Builders ────────────────────────────────────────────────────────

def success_response(data: Any = None, message: str = "OK", job_id: str = None) -> Dict:
    return {
        "status":    "success",
        "message":   message,
        "job_id":    job_id or generate_job_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data":      data,
    }


def error_response(message: str, detail: str = None, code: int = 500) -> Dict:
    return {
        "status":    "error",
        "message":   message,
        "detail":    detail,
        "code":      code,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── Data Validation ──────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, min_rows: int = 10) -> List[str]:
    """
    Validate a DataFrame for ML readiness.
    Returns list of warning/error strings (empty = OK).
    """
    issues = []

    if df.empty:
        issues.append("DataFrame is empty.")
        return issues

    if len(df) < min_rows:
        issues.append(f"Too few rows: {len(df)} (minimum {min_rows}).")

    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.5].index.tolist()
    if high_missing:
        issues.append(f"High missingness (>50%) in columns: {high_missing}")

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate rows.")

    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        issues.append(f"Constant (zero-variance) columns: {constant_cols}")

    return issues


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristically infer a binary target column.
    Looks for columns named 'label', 'target', 'class', 'risk', 'anomaly'.
    """
    candidates = ["label", "target", "class", "risk", "anomaly", "y"]
    for col in df.columns:
        if col.lower() in candidates:
            return col
    return None


def infer_coordinate_columns(df: pd.DataFrame):
    """
    Returns (lat_col, lon_col) names by heuristic column name matching.
    """
    lat_candidates = ["latitude", "lat", "y_coord", "y"]
    lon_candidates = ["longitude", "lon", "lng", "x_coord", "x"]

    lat_col = next((c for c in df.columns if c.lower() in lat_candidates), None)
    lon_col = next((c for c in df.columns if c.lower() in lon_candidates), None)
    return lat_col, lon_col


# ─── Timestamp ────────────────────────────────────────────────────────────────

def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
