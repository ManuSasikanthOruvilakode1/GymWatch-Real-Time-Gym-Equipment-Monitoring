"""SQLite persistence for usage sessions and state changes."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT NOT NULL,
    equipment   TEXT NOT NULL,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    duration_s  REAL
);
CREATE INDEX IF NOT EXISTS ix_sessions_camera_start ON sessions(camera_id, started_at);
CREATE TABLE IF NOT EXISTS state_changes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    state     TEXT NOT NULL,
    at        REAL NOT NULL
);
"""


class EventStore:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            # Sessions left open by an unclean shutdown cannot be trusted.
            self._conn.execute("DELETE FROM sessions WHERE ended_at IS NULL")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record_state(self, camera_id: str, equipment: str, state: str, at: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO state_changes(camera_id, state, at) VALUES (?, ?, ?)",
                (camera_id, state, at),
            )
            if state == "in_use":
                self._conn.execute(
                    "INSERT INTO sessions(camera_id, equipment, started_at) VALUES (?, ?, ?)",
                    (camera_id, equipment, at),
                )
            else:
                self._conn.execute(
                    "UPDATE sessions SET ended_at = ?, duration_s = ? - started_at "
                    "WHERE camera_id = ? AND ended_at IS NULL",
                    (at, at, camera_id),
                )

    def sessions(self, camera_id: str | None = None, limit: int = 50) -> list[dict]:
        q = "SELECT * FROM sessions"
        args: tuple = ()
        if camera_id:
            q += " WHERE camera_id = ?"
            args = (camera_id,)
        q += " ORDER BY started_at DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(q, (*args, limit)).fetchall()
        return [dict(r) for r in rows]

    def stats(self, window_s: float = 3600.0, now: float | None = None) -> list[dict]:
        """Per-camera utilisation, session count and mean duration over the last ``window_s``."""
        now = time.time() if now is None else now
        start = now - window_s
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT camera_id, equipment,
                       COUNT(*) AS sessions,
                       SUM(MIN(COALESCE(ended_at, :now), :now) - MAX(started_at, :start)) AS busy_s,
                       AVG(COALESCE(duration_s, :now - started_at)) AS mean_session_s
                FROM sessions
                WHERE COALESCE(ended_at, :now) > :start AND started_at < :now
                GROUP BY camera_id, equipment
                """,
                {"now": now, "start": start},
            ).fetchall()
        return [
            {
                "camera_id": r["camera_id"],
                "equipment": r["equipment"],
                "sessions": r["sessions"],
                "utilisation": round((r["busy_s"] or 0.0) / window_s, 4),
                "mean_session_s": round(r["mean_session_s"] or 0.0, 1),
                "window_s": window_s,
            }
            for r in rows
        ]
