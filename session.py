"""
GUARDIAN ML — Session State Manager
In-memory store for active job artifacts (uploaded files, preprocessors, trainers).
Thread-safe for single-process uvicorn deployments.
For multi-process: replace with Redis backend.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional
from utils.logger import setup_logger

logger = setup_logger("guardian.session")


class SessionStore:
    """Thread-safe in-memory key-value store for job state."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def set(self, job_id: str, key: str, value: Any) -> None:
        with self._lock:
            if job_id not in self._store:
                self._store[job_id] = {}
            self._store[job_id][key] = value

    def get(self, job_id: str, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(job_id, {}).get(key, default)

    def get_job(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            return self._store.get(job_id)

    def exists(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._store

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._store.pop(job_id, None)

    def list_jobs(self):
        with self._lock:
            return list(self._store.keys())


# Singleton session store — imported across all routers
session = SessionStore()
