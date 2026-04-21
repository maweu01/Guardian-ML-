"""
GUARDIAN ML — Core ML Pipeline
Multi-model training, evaluation, cross-validation, and persistence.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
import xgboost as xgb

from utils.logger import setup_logger
from utils.helpers import ensure_dirs, utc_now_str

logger = setup_logger("guardian.ml_pipeline")


# ─── Model Registry ───────────────────────────────────────────────────────────

def build_models(config: Dict) -> Dict[str, Any]:
    """Instantiate all enabled models from config."""
    ml_cfg = config.get("ml", {}).get("models", {})
    seed   = config.get("ml", {}).get("random_seed", 42)
    models = {}

    if ml_cfg.get("random_forest", {}).get("enabled", True):
        rf_cfg = ml_cfg["random_forest"]
        models["random_forest"] = RandomForestClassifier(
            n_estimators    = rf_cfg.get("n_estimators", 200),
            max_depth       = rf_cfg.get("max_depth", 15),
            min_samples_split = rf_cfg.get("min_samples_split", 5),
            random_state    = seed,
            n_jobs          = rf_cfg.get("n_jobs", -1),
            class_weight    = "balanced",
        )

    if ml_cfg.get("xgboost", {}).get("enabled", True):
        xg_cfg = ml_cfg["xgboost"]
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators      = xg_cfg.get("n_estimators", 300),
            max_depth         = xg_cfg.get("max_depth", 6),
            learning_rate     = xg_cfg.get("learning_rate", 0.05),
            subsample         = xg_cfg.get("subsample", 0.8),
            colsample_bytree  = xg_cfg.get("colsample_bytree", 0.8),
            use_label_encoder = False,
            eval_metric       = "logloss",
            random_state      = seed,
            verbosity         = 0,
        )

    if ml_cfg.get("logistic_regression", {}).get("enabled", True):
        lr_cfg = ml_cfg["logistic_regression"]
        models["logistic_regression"] = LogisticRegression(
            max_iter     = lr_cfg.get("max_iter", 1000),
            C            = lr_cfg.get("C", 1.0),
            random_state = seed,
            class_weight = "balanced",
        )

    logger.info(f"Initialized models: {list(models.keys())}")
    return models


def build_anomaly_detector(config: Dict) -> Optional[IsolationForest]:
    """Build IsolationForest anomaly detector if enabled."""
    af_cfg = config.get("ml", {}).get("models", {}).get("isolation_forest", {})
    if not af_cfg.get("enabled", True):
        return None

    seed = config.get("ml", {}).get("random_seed", 42)
    return IsolationForest(
        contamination = af_cfg.get("contamination", 0.1),
        n_estimators  = af_cfg.get("n_estimators", 200),
        random_state  = af_cfg.get("random_state", seed),
        n_jobs        = -1,
    )


# ─── Training Engine ──────────────────────────────────────────────────────────

class GuardianTrainer:
    """
    Orchestrates multi-model training, cross-validation, evaluation,
    comparison, and model persistence for GUARDIAN ML.
    """

    def __init__(self, config: Dict):
        self.config     = config
        self.models_dir = config.get("ml", {}).get("models_dir", "models/saved")
        self.seed       = config.get("ml", {}).get("random_seed", 42)
        self.cv_folds   = config.get("ml", {}).get("evaluation", {}).get(
            "cross_validation_folds", 5
        )

        self.trained_models: Dict[str, Any] = {}
        self.results:        Dict[str, Dict] = {}
        self.best_model_name: Optional[str]  = None
        self.anomaly_detector: Optional[IsolationForest] = None

    # ── Core Training ─────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        feature_names: List[str],
    ) -> Dict:
        """
        Train all models, run cross-validation, evaluate on validation set.

        Returns:
            Full training report dict.
        """
        models = build_models(self.config)
        np.random.seed(self.seed)

        logger.info(f"Training {len(models)} models on {X_train.shape[0]} samples, "
                    f"{X_train.shape[1]} features.")

        for name, model in models.items():
            logger.info(f"Training: {name} ...")
            t0 = time.perf_counter()

            try:
                model.fit(X_train, y_train)
                elapsed = time.perf_counter() - t0

                val_metrics = self._evaluate(model, X_val, y_val)
                cv_metrics  = self._cross_validate(model, X_train, y_train)
                importance  = self._feature_importance(model, feature_names)

                self.trained_models[name] = model
                self.results[name] = {
                    "validation":    val_metrics,
                    "cross_val":     cv_metrics,
                    "feature_importance": importance,
                    "train_time_s":  round(elapsed, 4),
                }

                logger.info(
                    f"  {name} → val_f1={val_metrics['f1']:.4f}, "
                    f"val_auc={val_metrics.get('roc_auc', 'N/A')}, "
                    f"time={elapsed:.2f}s"
                )

            except Exception as e:
                logger.error(f"  {name} training failed: {e}")
                self.results[name] = {"error": str(e)}

        # Anomaly detection (unsupervised, on X_train)
        self.anomaly_detector = build_anomaly_detector(self.config)
        if self.anomaly_detector:
            logger.info("Fitting anomaly detector (IsolationForest) ...")
            self.anomaly_detector.fit(X_train)
            logger.info("Anomaly detector fitted.")

        self.best_model_name = self._select_best()
        return self._build_report(feature_names)

    def predict(
        self,
        X: np.ndarray,
        model_name: Optional[str] = None,
        include_anomaly: bool = True,
    ) -> Dict:
        """
        Run inference using specified (or best) model.

        Returns:
            {predictions, probabilities, risk_scores, anomaly_flags}
        """
        name = model_name or self.best_model_name
        if name not in self.trained_models:
            raise ValueError(f"Model '{name}' not trained. Available: {list(self.trained_models)}")

        model = self.trained_models[name]
        preds = model.predict(X)
        proba = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            risk_scores = proba[:, -1].tolist()
        else:
            risk_scores = preds.astype(float).tolist()

        output = {
            "model_used":   name,
            "predictions":  preds.tolist(),
            "risk_scores":  risk_scores,
            "n_samples":    int(len(X)),
        }

        if proba is not None:
            output["probabilities"] = proba.tolist()

        if include_anomaly and self.anomaly_detector:
            anomaly_raw   = self.anomaly_detector.predict(X)
            anomaly_score = self.anomaly_detector.score_samples(X)
            output["anomaly_flags"]  = (anomaly_raw == -1).tolist()
            output["anomaly_scores"] = anomaly_score.tolist()

        return output

    def save(self, directory: Optional[str] = None) -> str:
        """Persist all trained models and results."""
        save_dir = directory or self.models_dir
        ensure_dirs(save_dir)

        for name, model in self.trained_models.items():
            path = f"{save_dir}/{name}.pkl"
            joblib.dump(model, path)

        if self.anomaly_detector:
            joblib.dump(self.anomaly_detector, f"{save_dir}/isolation_forest.pkl")

        joblib.dump(self.results,         f"{save_dir}/results.pkl")
        joblib.dump(self.best_model_name, f"{save_dir}/best_model_name.pkl")

        logger.info(f"All models saved to: {save_dir}")
        return save_dir

    @classmethod
    def load(cls, directory: str, config: Dict) -> "GuardianTrainer":
        """Load a saved trainer with all models."""
        trainer = cls(config)
        save_dir = Path(directory)

        for pkl_path in save_dir.glob("*.pkl"):
            name = pkl_path.stem
            if name in ("results", "best_model_name", "isolation_forest"):
                continue
            trainer.trained_models[name] = joblib.load(pkl_path)

        iso_path = save_dir / "isolation_forest.pkl"
        if iso_path.exists():
            trainer.anomaly_detector = joblib.load(iso_path)

        results_path = save_dir / "results.pkl"
        if results_path.exists():
            trainer.results = joblib.load(results_path)

        best_path = save_dir / "best_model_name.pkl"
        if best_path.exists():
            trainer.best_model_name = joblib.load(best_path)

        logger.info(f"Trainer loaded from {directory}. "
                    f"Models: {list(trainer.trained_models)}")
        return trainer

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _evaluate(self, model, X: np.ndarray, y: np.ndarray) -> Dict:
        """Compute full classification metrics on a held-out set."""
        preds = model.predict(X)
        metrics = {
            "accuracy":  float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, average="weighted", zero_division=0)),
            "recall":    float(recall_score(y, preds, average="weighted", zero_division=0)),
            "f1":        float(f1_score(y, preds, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, preds).tolist(),
            "classification_report": classification_report(y, preds, output_dict=True),
        }

        if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
            proba = model.predict_proba(X)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y, proba))
            metrics["avg_precision"] = float(average_precision_score(y, proba))

        return metrics

    def _cross_validate(self, model, X: np.ndarray, y: np.ndarray) -> Dict:
        """Run stratified K-fold cross-validation."""
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]

        try:
            cv_results = cross_validate(
                model, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise"
            )
            return {
                "accuracy_mean":   float(cv_results["test_accuracy"].mean()),
                "accuracy_std":    float(cv_results["test_accuracy"].std()),
                "f1_mean":         float(cv_results["test_f1_weighted"].mean()),
                "f1_std":          float(cv_results["test_f1_weighted"].std()),
                "precision_mean":  float(cv_results["test_precision_weighted"].mean()),
                "recall_mean":     float(cv_results["test_recall_weighted"].mean()),
                "n_folds":         self.cv_folds,
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {"error": str(e)}

    def _feature_importance(self, model, feature_names: List[str]) -> List[Dict]:
        """Extract feature importances where available."""
        importance = None

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)

        if importance is None:
            return []

        pairs = sorted(
            zip(feature_names, importance.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"feature": f, "importance": round(float(v), 6)} for f, v in pairs[:20]]

    def _select_best(self) -> Optional[str]:
        """Select the best model by weighted F1 on validation set."""
        scored = {
            name: res["validation"]["f1"]
            for name, res in self.results.items()
            if "validation" in res and "f1" in res["validation"]
        }
        if not scored:
            return None
        best = max(scored, key=scored.get)
        logger.info(f"Best model: {best} (F1={scored[best]:.4f})")
        return best

    def _build_report(self, feature_names: List[str]) -> Dict:
        """Compile the full training report."""
        return {
            "timestamp":      utc_now_str(),
            "best_model":     self.best_model_name,
            "models_trained": list(self.trained_models.keys()),
            "n_features":     len(feature_names),
            "results":        self.results,
        }
