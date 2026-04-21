#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# GUARDIAN ML — Run Script
# Usage:
#   ./run.sh            → Start production server
#   ./run.sh dev        → Start dev server (hot reload)
#   ./run.sh test       → Run test suite
#   ./run.sh install    → Install dependencies
#   ./run.sh clean      → Remove generated artifacts
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
AMBER='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

BANNER="
╔══════════════════════════════════════════════════════════╗
║          G U A R D I A N   M L   v 1 . 0 . 0            ║
║   Geospatial Unified Anomaly Risk Detection & IAN        ║
╚══════════════════════════════════════════════════════════╝"

echo -e "${CYAN}${BANNER}${NC}"

COMMAND="${1:-start}"
BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/backend" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Ensure required directories exist ─────────────────────────
mkdir -p "${ROOT_DIR}/data/uploads" \
         "${ROOT_DIR}/data/processed" \
         "${ROOT_DIR}/data/exports" \
         "${ROOT_DIR}/models/saved" \
         "${ROOT_DIR}/logs"

case "$COMMAND" in

  install)
    echo -e "${AMBER}[GUARDIAN]${NC} Installing Python dependencies…"
    if [ ! -d "${ROOT_DIR}/venv" ]; then
      python3 -m venv "${ROOT_DIR}/venv"
      echo -e "${GREEN}[GUARDIAN]${NC} Virtual environment created."
    fi
    source "${ROOT_DIR}/venv/bin/activate"
    pip install --upgrade pip -q
    pip install -r "${BACKEND_DIR}/requirements.txt"
    echo -e "${GREEN}[GUARDIAN]${NC} ${BOLD}Dependencies installed successfully.${NC}"
    ;;

  start)
    echo -e "${GREEN}[GUARDIAN]${NC} Starting production server…"
    cd "${BACKEND_DIR}"
    if [ -d "${ROOT_DIR}/venv" ]; then
      source "${ROOT_DIR}/venv/bin/activate"
    fi
    export PYTHONPATH="${BACKEND_DIR}"
    python main.py
    ;;

  dev)
    echo -e "${AMBER}[GUARDIAN]${NC} Starting development server (hot reload)…"
    cd "${BACKEND_DIR}"
    if [ -d "${ROOT_DIR}/venv" ]; then
      source "${ROOT_DIR}/venv/bin/activate"
    fi
    export PYTHONPATH="${BACKEND_DIR}"
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
    ;;

  test)
    echo -e "${CYAN}[GUARDIAN]${NC} Running test suite…"
    cd "${ROOT_DIR}"
    if [ -d "${ROOT_DIR}/venv" ]; then
      source "${ROOT_DIR}/venv/bin/activate"
    fi
    export PYTHONPATH="${BACKEND_DIR}"
    pytest tests/ -v --tb=short --asyncio-mode=auto
    ;;

  clean)
    echo -e "${AMBER}[GUARDIAN]${NC} Cleaning generated artifacts…"
    rm -rf "${ROOT_DIR}/data/uploads/"*
    rm -rf "${ROOT_DIR}/data/processed/"*
    rm -rf "${ROOT_DIR}/data/exports/"*
    rm -rf "${ROOT_DIR}/models/saved/"*
    rm -rf "${ROOT_DIR}/logs/"*
    rm -rf "${ROOT_DIR}/__pycache__" "${BACKEND_DIR}/__pycache__"
    find "${ROOT_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${ROOT_DIR}" -name "*.pyc" -delete 2>/dev/null || true
    echo -e "${GREEN}[GUARDIAN]${NC} Clean complete."
    ;;

  *)
    echo -e "${RED}[GUARDIAN]${NC} Unknown command: ${COMMAND}"
    echo "Usage: ./run.sh [install|start|dev|test|clean]"
    exit 1
    ;;
esac
