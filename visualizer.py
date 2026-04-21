"""
GUARDIAN ML — Visualization Engine
Generates Plotly-based charts, geospatial risk maps, and report figures.
"""

from __future__ import annotations

import io
import json
import base64
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils.logger import setup_logger

logger = setup_logger("guardian.visualizer")


# ─── Color Palette ────────────────────────────────────────────────────────────
GUARDIAN_PALETTE = {
    "background":  "#0a0e1a",
    "panel":       "#111827",
    "border":      "#1f2937",
    "accent_blue": "#3b82f6",
    "accent_cyan": "#06b6d4",
    "accent_red":  "#ef4444",
    "accent_green":"#22c55e",
    "accent_amber":"#f59e0b",
    "text":        "#e2e8f0",
    "subtext":     "#94a3b8",
}

RISK_COLORS = {
    "low":    "#22c55e",
    "medium": "#f59e0b",
    "high":   "#ef4444",
}


def _guardian_layout(title: str, height: int = 500) -> Dict:
    """Base Plotly layout matching GUARDIAN dark theme."""
    return dict(
        title=dict(
            text=title,
            font=dict(family="'Courier New', monospace", size=16, color=GUARDIAN_PALETTE["accent_cyan"]),
        ),
        paper_bgcolor=GUARDIAN_PALETTE["background"],
        plot_bgcolor=GUARDIAN_PALETTE["panel"],
        font=dict(family="'Courier New', monospace", color=GUARDIAN_PALETTE["text"], size=11),
        height=height,
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(gridcolor=GUARDIAN_PALETTE["border"], zerolinecolor=GUARDIAN_PALETTE["border"]),
        yaxis=dict(gridcolor=GUARDIAN_PALETTE["border"], zerolinecolor=GUARDIAN_PALETTE["border"]),
    )


# ─── Chart Generators ─────────────────────────────────────────────────────────

def plot_risk_distribution(risk_scores: List[float], title: str = "Risk Score Distribution") -> Dict:
    """Histogram of predicted risk scores."""
    scores = np.array(risk_scores)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=40,
        marker_color=GUARDIAN_PALETTE["accent_blue"],
        marker_line_color=GUARDIAN_PALETTE["accent_cyan"],
        marker_line_width=0.5,
        opacity=0.85,
        name="Risk Score",
    ))

    # Overlay risk-level bands
    for level, (lo, hi, color) in {
        "Low":    (0.0,  0.33, RISK_COLORS["low"]),
        "Medium": (0.33, 0.66, RISK_COLORS["medium"]),
        "High":   (0.66, 1.0,  RISK_COLORS["high"]),
    }.items():
        fig.add_vrect(
            x0=lo, x1=hi,
            fillcolor=color, opacity=0.08,
            line_width=0,
            annotation_text=level,
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color=color,
        )

    fig.update_layout(
        **_guardian_layout(title),
        xaxis_title="Risk Score",
        yaxis_title="Count",
    )
    return json.loads(fig.to_json())


def plot_model_comparison(results: Dict) -> Dict:
    """Bar chart comparing models across key metrics."""
    rows = []
    for model_name, res in results.items():
        if "validation" not in res:
            continue
        val = res["validation"]
        rows.append({
            "Model":     model_name.replace("_", " ").title(),
            "Accuracy":  val.get("accuracy", 0),
            "Precision": val.get("precision", 0),
            "Recall":    val.get("recall", 0),
            "F1":        val.get("f1", 0),
            "ROC-AUC":   val.get("roc_auc", 0),
        })

    if not rows:
        return {}

    df = pd.DataFrame(rows).set_index("Model")
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    colors  = [
        GUARDIAN_PALETTE["accent_blue"],
        GUARDIAN_PALETTE["accent_cyan"],
        GUARDIAN_PALETTE["accent_green"],
        GUARDIAN_PALETTE["accent_amber"],
        GUARDIAN_PALETTE["accent_red"],
    ]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        if metric in df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=df.index.tolist(),
                y=df[metric].tolist(),
                marker_color=color,
                opacity=0.85,
            ))

    fig.update_layout(
        **_guardian_layout("Model Performance Comparison", height=420),
        barmode="group",
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        legend=dict(
            bgcolor=GUARDIAN_PALETTE["panel"],
            bordercolor=GUARDIAN_PALETTE["border"],
        ),
    )
    return json.loads(fig.to_json())


def plot_feature_importance(
    importance_data: List[Dict],
    model_name: str = "Model",
    top_n: int = 15,
) -> Dict:
    """Horizontal bar chart of feature importances."""
    if not importance_data:
        return {}

    df = pd.DataFrame(importance_data).head(top_n).sort_values("importance")

    fig = go.Figure(go.Bar(
        x=df["importance"].tolist(),
        y=df["feature"].tolist(),
        orientation="h",
        marker=dict(
            color=df["importance"].tolist(),
            colorscale=[[0, GUARDIAN_PALETTE["accent_blue"]], [1, GUARDIAN_PALETTE["accent_cyan"]]],
            showscale=False,
        ),
    ))

    fig.update_layout(
        **_guardian_layout(f"Feature Importance — {model_name.replace('_', ' ').title()}", height=450),
        xaxis_title="Importance",
        yaxis_title="Feature",
        margin=dict(l=160, r=40, t=60, b=60),
    )
    return json.loads(fig.to_json())


def plot_confusion_matrix(cm: List[List[int]], labels: Optional[List[str]] = None) -> Dict:
    """Annotated confusion matrix heatmap."""
    cm_arr = np.array(cm)
    n      = len(cm_arr)
    labels = labels or [str(i) for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=cm_arr.tolist(),
        x=labels,
        y=labels,
        colorscale=[[0, GUARDIAN_PALETTE["panel"]], [1, GUARDIAN_PALETTE["accent_blue"]]],
        showscale=True,
        text=cm_arr.tolist(),
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
    ))

    fig.update_layout(
        **_guardian_layout("Confusion Matrix", height=400),
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return json.loads(fig.to_json())


def plot_geospatial_risk(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    risk_col: str,
    id_col: Optional[str] = None,
    title: str = "GUARDIAN — Geospatial Risk Map",
) -> Dict:
    """
    Scatter map of risk scores overlaid on geographic coordinates.
    Uses Plotly mapbox with OpenStreetMap tiles.
    """
    df = df.dropna(subset=[lat_col, lon_col, risk_col]).copy()
    df["_risk_level"] = pd.cut(
        df[risk_col],
        bins=[0, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)

    color_map = {"Low": RISK_COLORS["low"], "Medium": RISK_COLORS["medium"], "High": RISK_COLORS["high"]}

    hover_data = {lat_col: True, lon_col: True, risk_col: ":.3f"}
    if id_col and id_col in df.columns:
        hover_data[id_col] = True

    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        color="_risk_level",
        color_discrete_map=color_map,
        size=risk_col,
        size_max=18,
        hover_data=hover_data,
        zoom=3,
        mapbox_style="carto-darkmatter",
        title=title,
    )

    fig.update_layout(
        paper_bgcolor=GUARDIAN_PALETTE["background"],
        font=dict(family="'Courier New', monospace", color=GUARDIAN_PALETTE["text"], size=11),
        title=dict(
            font=dict(color=GUARDIAN_PALETTE["accent_cyan"], size=15),
        ),
        height=550,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            bgcolor=GUARDIAN_PALETTE["panel"],
            bordercolor=GUARDIAN_PALETTE["border"],
            title=dict(text="Risk Level", font=dict(color=GUARDIAN_PALETTE["subtext"])),
        ),
    )
    return json.loads(fig.to_json())


def plot_anomaly_timeline(
    scores: List[float],
    flags: List[bool],
    timestamps: Optional[List[str]] = None,
) -> Dict:
    """Time-series plot of anomaly scores with flagged points highlighted."""
    x = timestamps or list(range(len(scores)))
    scores_arr = np.array(scores)
    flags_arr  = np.array(flags)

    fig = go.Figure()

    # Baseline score line
    fig.add_trace(go.Scatter(
        x=x,
        y=scores_arr.tolist(),
        mode="lines",
        name="Anomaly Score",
        line=dict(color=GUARDIAN_PALETTE["accent_blue"], width=1.5),
    ))

    # Anomaly markers
    if flags_arr.any():
        anomaly_x = [x[i] for i in range(len(x)) if flags_arr[i]]
        anomaly_y = scores_arr[flags_arr].tolist()
        fig.add_trace(go.Scatter(
            x=anomaly_x,
            y=anomaly_y,
            mode="markers",
            name="Anomaly",
            marker=dict(
                color=GUARDIAN_PALETTE["accent_red"],
                size=10,
                symbol="circle",
                line=dict(color="white", width=1),
            ),
        ))

    fig.update_layout(
        **_guardian_layout("Anomaly Detection Timeline", height=380),
        xaxis_title="Sample Index / Timestamp",
        yaxis_title="Anomaly Score",
    )
    return json.loads(fig.to_json())


def plot_cv_scores(cv_results: Dict) -> Dict:
    """Cross-validation score distribution per model."""
    rows = []
    for model_name, res in cv_results.items():
        cv = res.get("cross_val", {})
        if "f1_mean" in cv:
            rows.append({
                "Model": model_name.replace("_", " ").title(),
                "F1 Mean": cv["f1_mean"],
                "F1 Std":  cv["f1_std"],
            })

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    fig = go.Figure(go.Bar(
        x=df["Model"].tolist(),
        y=df["F1 Mean"].tolist(),
        error_y=dict(type="data", array=df["F1 Std"].tolist(), visible=True, color=GUARDIAN_PALETTE["subtext"]),
        marker_color=GUARDIAN_PALETTE["accent_cyan"],
        opacity=0.85,
    ))

    fig.update_layout(
        **_guardian_layout("Cross-Validation F1 Scores (Mean ± Std)", height=380),
        xaxis_title="Model",
        yaxis_title="F1 Score",
        yaxis_range=[0, 1.05],
    )
    return json.loads(fig.to_json())


def export_figure_base64(fig_json: Dict, fmt: str = "png") -> str:
    """
    Convert a Plotly figure JSON dict to base64-encoded image.
    Requires kaleido.
    """
    import plotly.io as pio
    fig = go.Figure(fig_json)
    img_bytes = pio.to_image(fig, format=fmt, width=1200, height=600, scale=2)
    return base64.b64encode(img_bytes).decode("utf-8")
