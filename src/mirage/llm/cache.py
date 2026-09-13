"""SQLite prompt/response cache. The cache IS the reproducible dataset:
after a real run, experiments replay without API access.
Key = sha256(provider | model | temperature | prompt)."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path


class PromptCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            " key TEXT PRIMARY KEY, provider TEXT, model TEXT,"
            " prompt TEXT, response TEXT, created_at REAL)"
        )
        self._conn.commit()

    @staticmethod
    def make_key(provider: str, model: str, temperature: float, prompt: str) -> str:
        blob = f"{provider}|{model}|{temperature}|{prompt}".encode()
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, provider: str, model: str, prompt: str, response: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?)",
                (key, provider, model, prompt, response, time.time()),
            )
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            n, = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        return {"entries": int(n)}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
