"""
GUARDIAN ML — Geospatial Unified Anomaly Risk Detection & Intelligence Analysis Network
Main FastAPI Application Entry Point
"""

import uvicorn
import logging
import yaml
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from api.upload import router as upload_router
from api.process import router as process_router
from api.train import router as train_router
from api.predict import router as predict_router
from api.visualize import router as visualize_router
from utils.logger import setup_logger

# ─── Load Configuration ───────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# ─── Logger ───────────────────────────────────────────────────────────────────
logger = setup_logger("guardian_main", CONFIG.get("logging", {}))
logger.info("GUARDIAN ML system initializing...")

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="GUARDIAN ML",
    description=(
        "Geospatial Unified Anomaly Risk Detection & Intelligence Analysis Network. "
        "A production-grade ML pipeline for geospatial risk assessment, "
        "anomaly detection, and predictive analytics."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.get("cors", {}).get("origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static Files (Frontend) ──────────────────────────────────────────────────
FRONTEND_PATH = Path(__file__).parent.parent / "frontend"
if FRONTEND_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")


# ─── API Routers ──────────────────────────────────────────────────────────────
app.include_router(upload_router,    prefix="/upload",    tags=["Data Upload"])
app.include_router(process_router,   prefix="/process",   tags=["Preprocessing"])
app.include_router(train_router,     prefix="/train",     tags=["Model Training"])
app.include_router(predict_router,   prefix="/predict",   tags=["Prediction"])
app.include_router(visualize_router, prefix="/visualize", tags=["Visualization"])


# ─── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Serve the GUARDIAN ML frontend."""
    index_path = FRONTEND_PATH / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>GUARDIAN ML API Running</h1><p>Visit <a href='/api/docs'>/api/docs</a></p>")


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "operational",
        "system": "GUARDIAN ML",
        "version": "1.0.0",
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    host = CONFIG.get("server", {}).get("host", "0.0.0.0")
    port = CONFIG.get("server", {}).get("port", 8000)
    debug = CONFIG.get("server", {}).get("debug", False)

    logger.info(f"GUARDIAN ML starting on http://{host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info",
    )
