"""
GUARDIAN ML — Risk Scoring Engine
Composite geospatial risk scoring, spatial aggregation, and risk-level classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from utils.logger import setup_logger

logger = setup_logger("guardian.risk_engine")


# ─── Risk Level Thresholds ────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "low":    (0.0,  0.33),
    "medium": (0.33, 0.66),
    "high":   (0.66, 1.01),
}

RISK_WEIGHTS_DEFAULT = {
    "model_score":    0.50,
    "anomaly_score":  0.25,
    "spatial_density":0.15,
    "temporal_weight":0.10,
}


class RiskScorer:
    """
    Computes composite risk scores combining:
      - ML model probability output
      - Anomaly detection score (IsolationForest)
      - Spatial density index
      - Temporal recency weight

    Outputs a normalized [0, 1] risk score per sample.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        weights_cfg = self.config.get("risk_weights", RISK_WEIGHTS_DEFAULT)
        total = sum(weights_cfg.values()) or 1.0
        # Normalize weights to sum to 1
        self.weights = {k: v / total for k, v in weights_cfg.items()}

    def compute(
        self,
        model_probs:     np.ndarray,                    # shape (N,)
        anomaly_scores:  Optional[np.ndarray] = None,   # shape (N,), raw IsoForest scores
        lat:             Optional[np.ndarray] = None,   # shape (N,)
        lon:             Optional[np.ndarray] = None,   # shape (N,)
        timestamps:      Optional[np.ndarray] = None,   # shape (N,) datetime64
    ) -> np.ndarray:
        """
        Compute composite risk scores.

        Args:
            model_probs:    Model's positive-class probability [0, 1].
            anomaly_scores: Raw anomaly scores from IsolationForest (lower = more anomalous).
            lat:            Latitude values.
            lon:            Longitude values.
            timestamps:     Datetime array for temporal weighting.

        Returns:
            composite_risk: np.ndarray of shape (N,), values in [0, 1].
        """
        N = len(model_probs)
        scores = np.zeros(N)

        # ── 1. Model score component ────────────────────────────────────────
        w_model = self.weights.get("model_score", 0.5)
        scores += w_model * np.clip(model_probs, 0, 1)

        # ── 2. Anomaly score component ──────────────────────────────────────
        w_anom = self.weights.get("anomaly_score", 0.25)
        if anomaly_scores is not None:
            # IsolationForest: lower score → more anomalous → higher risk
            norm_anom = self._normalize(-anomaly_scores)
            scores += w_anom * norm_anom
        else:
            # Redistribute weight to model score
            scores += w_anom * np.clip(model_probs, 0, 1)

        # ── 3. Spatial density index ────────────────────────────────────────
        w_spatial = self.weights.get("spatial_density", 0.15)
        if lat is not None and lon is not None:
            density = self._spatial_density_index(lat, lon)
            scores += w_spatial * density
        else:
            scores += w_spatial * np.clip(model_probs, 0, 1)

        # ── 4. Temporal recency weight ──────────────────────────────────────
        w_temp = self.weights.get("temporal_weight", 0.10)
        if timestamps is not None:
            temp_w = self._temporal_recency_weight(timestamps)
            scores = scores * (1.0 + w_temp * temp_w)

        return np.clip(scores, 0.0, 1.0)

    def classify(self, risk_scores: np.ndarray) -> List[str]:
        """Map continuous risk scores to categorical risk levels."""
        levels = []
        for s in risk_scores:
            for level, (lo, hi) in RISK_THRESHOLDS.items():
                if lo <= s < hi:
                    levels.append(level)
                    break
            else:
                levels.append("high")
        return levels

    def summarize(self, risk_scores: np.ndarray) -> Dict:
        """Produce a risk summary report."""
        levels = self.classify(risk_scores)
        level_series = pd.Series(levels)

        summary = {
            "n_samples":        int(len(risk_scores)),
            "mean_risk":        float(np.mean(risk_scores)),
            "median_risk":      float(np.median(risk_scores)),
            "max_risk":         float(np.max(risk_scores)),
            "min_risk":         float(np.min(risk_scores)),
            "std_risk":         float(np.std(risk_scores)),
            "level_counts":     level_series.value_counts().to_dict(),
            "level_pct":        (level_series.value_counts(normalize=True) * 100).round(1).to_dict(),
            "high_risk_count":  int((risk_scores >= 0.66).sum()),
            "high_risk_pct":    float((risk_scores >= 0.66).mean() * 100),
        }
        return summary

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]."""
        xmin, xmax = x.min(), x.max()
        if xmax == xmin:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    @staticmethod
    def _spatial_density_index(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Approximate local spatial density via KDE bandwidth estimation.
        Points in dense clusters → higher density → higher risk contribution.
        """
        from sklearn.neighbors import KernelDensity

        coords = np.column_stack([lat, lon])
        kde = KernelDensity(bandwidth=0.5, kernel="gaussian")
        kde.fit(coords)
        log_density = kde.score_samples(coords)
        return RiskScorer._normalize(log_density)

    @staticmethod
    def _temporal_recency_weight(timestamps: np.ndarray) -> np.ndarray:
        """
        Compute exponential recency weight: more recent events → higher weight.
        Decays to zero for oldest events.
        """
        try:
            ts = pd.to_datetime(timestamps)
            unix = ts.astype(np.int64).values.astype(float)
            w = (unix - unix.min()) / (unix.max() - unix.min() + 1e-9)
            return w
        except Exception:
            return np.ones(len(timestamps)) * 0.5


def risk_score_dataframe(
    df: pd.DataFrame,
    predictions: Dict,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    ts_col:  Optional[str] = None,
    config:  Dict = None,
) -> pd.DataFrame:
    """
    Attach risk scores and level labels to a DataFrame.

    Args:
        df:          Source dataframe.
        predictions: Output dict from GuardianTrainer.predict().
        lat_col:     Latitude column name.
        lon_col:     Longitude column name.
        ts_col:      Timestamp column name.
        config:      System config dict.

    Returns:
        DataFrame with added columns:
            guardian_risk_score, guardian_risk_level,
            guardian_anomaly_flag, guardian_model_prob
    """
    scorer = RiskScorer(config)
    out    = df.copy().reset_index(drop=True)

    model_probs = np.array(predictions.get("risk_scores", predictions.get("predictions", [])))

    anomaly_scores = None
    if "anomaly_scores" in predictions:
        anomaly_scores = np.array(predictions["anomaly_scores"])

    lat = out[lat_col].values if lat_col and lat_col in out else None
    lon = out[lon_col].values if lon_col and lon_col in out else None
    ts  = out[ts_col].values  if ts_col  and ts_col  in out else None

    composite = scorer.compute(
        model_probs=model_probs,
        anomaly_scores=anomaly_scores,
        lat=lat,
        lon=lon,
        timestamps=ts,
    )

    out["guardian_model_prob"]   = model_probs
    out["guardian_risk_score"]   = composite
    out["guardian_risk_level"]   = scorer.classify(composite)
    out["guardian_anomaly_flag"] = predictions.get("anomaly_flags", [False] * len(out))

    logger.info(
        f"Risk scoring complete. "
        f"High: {(composite >= 0.66).sum()}, "
        f"Medium: {((composite >= 0.33) & (composite < 0.66)).sum()}, "
        f"Low: {(composite < 0.33).sum()}"
    )
    return out
