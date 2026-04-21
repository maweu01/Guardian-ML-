"""
GUARDIAN ML — Data Preprocessing Pipeline
Handles imputation, scaling, outlier removal, encoding, and feature validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

from utils.logger import setup_logger
from utils.helpers import validate_dataframe, infer_coordinate_columns, ensure_dirs

logger = setup_logger("guardian.preprocessing")


# ─── Scaler Registry ──────────────────────────────────────────────────────────
SCALERS = {
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
    "robust":   RobustScaler,
}


class GuardianPreprocessor:
    """
    End-to-end preprocessing pipeline for GUARDIAN ML.

    Steps:
      1. Schema inspection
      2. Duplicate & constant column removal
      3. Type coercion
      4. Outlier clipping / removal
      5. Imputation
      6. Encoding (categorical → numeric)
      7. Geospatial feature extraction
      8. Temporal feature extraction
      9. Scaling
      10. Train / validation / test split
    """

    def __init__(self, config: Dict):
        cfg = config.get("ml", {}).get("preprocessing", {})
        self.imputation_strategy: str = cfg.get("imputation_strategy", "median")
        self.scaling:             str = cfg.get("scaling", "standard")
        self.outlier_method:      str = cfg.get("outlier_method", "iqr")
        self.outlier_threshold:  float = cfg.get("outlier_threshold", 3.0)

        split_cfg = config.get("ml", {})
        self.test_size:       float = split_cfg.get("test_size", 0.2)
        self.validation_size: float = split_cfg.get("validation_size", 0.1)
        self.random_seed:       int = split_cfg.get("random_seed", 42)

        geo_cfg = config.get("ml", {}).get("feature_engineering", {})
        self.geo_features:  bool = geo_cfg.get("geospatial_features", True)
        self.temp_features: bool = geo_cfg.get("temporal_features", True)

        self._scaler:   Optional[Any] = None
        self._imputer:  Optional[SimpleImputer] = None
        self._feature_names: List[str] = []
        self._cat_mappings:  Dict[str, Dict] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Dict:
        """
        Fit the preprocessor on df and return train/val/test splits.

        Returns:
            {
                "X_train", "X_val", "X_test",
                "y_train", "y_val", "y_test",  (if target_col given)
                "feature_names",
                "stats",
                "warnings",
            }
        """
        logger.info(f"Preprocessing: {df.shape[0]} rows × {df.shape[1]} cols")
        warnings = validate_dataframe(df)

        # ── 1. Clone to avoid side effects ───────────────────────────────────
        df = df.copy()

        # ── 2. Remove duplicates ─────────────────────────────────────────────
        before = len(df)
        df = df.drop_duplicates()
        if len(df) < before:
            logger.info(f"Removed {before - len(df)} duplicate rows")

        # ── 3. Separate target ────────────────────────────────────────────────
        y = None
        if target_col and target_col in df.columns:
            y = df[target_col].copy()
            df = df.drop(columns=[target_col])

        # ── 4. Remove constant columns ────────────────────────────────────────
        constant = [c for c in df.columns if df[c].nunique() <= 1]
        if constant:
            logger.info(f"Dropping constant columns: {constant}")
            df = df.drop(columns=constant)

        # ── 5. Type coercion ──────────────────────────────────────────────────
        df = self._coerce_types(df)

        # ── 6. Feature engineering ────────────────────────────────────────────
        if self.geo_features:
            df = self._add_geospatial_features(df)
        if self.temp_features:
            df = self._add_temporal_features(df)

        # ── 7. Encode categoricals ────────────────────────────────────────────
        df = self._encode_categoricals(df)

        # ── 8. Select numeric only ────────────────────────────────────────────
        df = df.select_dtypes(include=[np.number])

        # ── 9. Outlier handling ───────────────────────────────────────────────
        df = self._handle_outliers(df)

        # ── 10. Imputation ───────────────────────────────────────────────────
        self._imputer = SimpleImputer(strategy=self.imputation_strategy)
        X = self._imputer.fit_transform(df)
        self._feature_names = list(df.columns)

        # ── 11. Scaling ───────────────────────────────────────────────────────
        ScalerClass = SCALERS.get(self.scaling, StandardScaler)
        self._scaler = ScalerClass()
        X = self._scaler.fit_transform(X)

        # ── 12. Split ─────────────────────────────────────────────────────────
        splits = self._split(X, y)

        stats = self._compute_stats(df, y)
        logger.info(f"Preprocessing complete. Features: {len(self._feature_names)}")

        return {
            **splits,
            "feature_names": self._feature_names,
            "stats": stats,
            "warnings": warnings,
        }

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        if self._imputer is None or self._scaler is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform() first.")

        df = df.copy()
        df = self._coerce_types(df)
        if self.geo_features:
            df = self._add_geospatial_features(df)
        if self.temp_features:
            df = self._add_temporal_features(df)
        df = self._encode_categoricals(df)
        df = df.select_dtypes(include=[np.number])

        # Align to training columns
        missing = set(self._feature_names) - set(df.columns)
        for col in missing:
            df[col] = np.nan
        df = df[self._feature_names]

        X = self._imputer.transform(df)
        return self._scaler.transform(X)

    def save(self, directory: str):
        """Persist fitted preprocessor artifacts."""
        ensure_dirs(directory)
        joblib.dump(self._imputer,        f"{directory}/imputer.pkl")
        joblib.dump(self._scaler,         f"{directory}/scaler.pkl")
        joblib.dump(self._feature_names,  f"{directory}/feature_names.pkl")
        joblib.dump(self._cat_mappings,   f"{directory}/cat_mappings.pkl")
        logger.info(f"Preprocessor saved to {directory}")

    @classmethod
    def load(cls, directory: str, config: Dict) -> "GuardianPreprocessor":
        """Load a previously saved preprocessor."""
        p = cls(config)
        p._imputer       = joblib.load(f"{directory}/imputer.pkl")
        p._scaler        = joblib.load(f"{directory}/scaler.pkl")
        p._feature_names = joblib.load(f"{directory}/feature_names.pkl")
        p._cat_mappings  = joblib.load(f"{directory}/cat_mappings.pkl")
        logger.info(f"Preprocessor loaded from {directory}")
        return p

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt numeric coercion on object columns that look numeric."""
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                pass
        return df

    def _add_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived geospatial features if lat/lon are available."""
        lat_col, lon_col = infer_coordinate_columns(df)
        if lat_col and lon_col:
            lat = df[lat_col]
            lon = df[lon_col]
            df["geo_lat_rad"]     = np.radians(lat)
            df["geo_lon_rad"]     = np.radians(lon)
            df["geo_distance_eq"] = np.sqrt(lat**2 + lon**2)
            # Quadrant encoding
            df["geo_quadrant"]    = (
                (lat >= 0).astype(int) * 2 + (lon >= 0).astype(int)
            )
            logger.debug("Geospatial features added.")
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from datetime columns."""
        for col in df.columns:
            if df[col].dtype in ["object", "datetime64[ns]"]:
                try:
                    ts = pd.to_datetime(df[col], infer_datetime_format=True)
                    df[f"{col}_year"]    = ts.dt.year
                    df[f"{col}_month"]   = ts.dt.month
                    df[f"{col}_dayofweek"] = ts.dt.dayofweek
                    df[f"{col}_hour"]    = ts.dt.hour
                    df[f"{col}_quarter"] = ts.dt.quarter
                    df = df.drop(columns=[col])
                    logger.debug(f"Temporal features extracted from: {col}")
                except Exception:
                    pass
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode remaining categorical columns."""
        for col in df.select_dtypes(include=["object", "category"]).columns:
            unique_vals = df[col].dropna().unique().tolist()
            mapping = {v: i for i, v in enumerate(sorted(map(str, unique_vals)))}
            self._cat_mappings[col] = mapping
            df[col] = df[col].map(lambda x: mapping.get(str(x), -1))
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers using IQR or z-score method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if self.outlier_method == "iqr":
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[numeric_cols] = df[numeric_cols].clip(lower=lower, upper=upper, axis=1)

        elif self.outlier_method == "zscore":
            from scipy import stats
            z = np.abs(stats.zscore(df[numeric_cols].fillna(0)))
            df[numeric_cols] = df[numeric_cols].where(
                z < self.outlier_threshold, other=np.nan
            )

        return df

    def _split(self, X: np.ndarray, y=None) -> Dict:
        """Produce train / validation / test splits."""
        indices = np.arange(len(X))

        val_ratio = self.validation_size / (1 - self.test_size)

        idx_trainval, idx_test = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_seed
        )
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=val_ratio, random_state=self.random_seed
        )

        out = {
            "X_train": X[idx_train],
            "X_val":   X[idx_val],
            "X_test":  X[idx_test],
        }

        if y is not None:
            y_arr = np.array(y)
            out["y_train"] = y_arr[idx_train]
            out["y_val"]   = y_arr[idx_val]
            out["y_test"]  = y_arr[idx_test]

        logger.info(
            f"Split → train:{len(idx_train)}, "
            f"val:{len(idx_val)}, test:{len(idx_test)}"
        )
        return out

    def _compute_stats(self, df: pd.DataFrame, y=None) -> Dict:
        """Compute summary statistics for the report."""
        stats = {
            "n_rows":     int(len(df)),
            "n_features": int(len(self._feature_names)),
            "missing_pct": float(df.isnull().mean().mean() * 100),
        }
        if y is not None:
            vc = pd.Series(y).value_counts(normalize=True).to_dict()
            stats["class_distribution"] = {str(k): float(v) for k, v in vc.items()}
        return stats
