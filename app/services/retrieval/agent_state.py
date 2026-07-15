import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

DEFAULT_DB_PATH = "/data/agentic_state/agentic_state.db"
DEFAULT_TTL_SECONDS = 20 * 60  # 20 minutes

_init_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn

    with _init_lock:
        if _conn is not None:
            return _conn

        db_path = os.getenv("AGENTIC_STATE_DB_PATH", DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_session_state (
              session_id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              pending INTEGER NOT NULL DEFAULT 0,
              pending_kind TEXT,
              pending_original_question TEXT,
              pending_selected_slugs_json TEXT,
              pending_clarifying_question TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              expires_at TEXT
            );
        """)
        conn.commit()
        _conn = conn
        return _conn

@dataclass(frozen=True)
class PendingClarification:
    session_id: str
    user_id: str
    pending_kind: Optional[str]
    original_question: str
    selected_slugs: list[str]
    clarifying_question: str

def get_pending(session_id: str, *, user_id: str) -> Optional[PendingClarification]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM agent_session_state WHERE session_id = ?", (session_id,)
    ).fetchone()

    if not row or not row["pending"] or row["user_id"] != user_id:
        return None

    if row["expires_at"] and row["expires_at"] <= _utc_now_iso():
        clear_pending(session_id)
        return None

    return PendingClarification(
        session_id=row["session_id"],
        user_id=row["user_id"],
        pending_kind=row["pending_kind"],
        original_question=row["pending_original_question"],
        selected_slugs=json.loads(row["pending_selected_slugs_json"]),
        clarifying_question=row["pending_clarifying_question"]
    )

def set_pending(*, session_id: str, user_id: str, kind: str, question: str, slugs: list[str], clarification: str):
    conn = _get_conn()
    now = _utc_now_iso()
    exp = datetime.fromtimestamp(time.time() + DEFAULT_TTL_SECONDS, tz=timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO agent_session_state 
        (session_id, user_id, pending, pending_kind, pending_original_question, 
         pending_selected_slugs_json, pending_clarifying_question, created_at, updated_at, expires_at)
        VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
          pending=1, pending_kind=excluded.pending_kind, 
          pending_original_question=excluded.pending_original_question,
          pending_selected_slugs_json=excluded.pending_selected_slugs_json,
          pending_clarifying_question=excluded.pending_clarifying_question,
          updated_at=excluded.updated_at, expires_at=excluded.expires_at
    """, (session_id, user_id, kind, question, json.dumps(slugs), clarification, now, now, exp))
    conn.commit()

def clear_pending(session_id: str):
    conn = _get_conn()
    conn.execute("UPDATE agent_session_state SET pending=0 WHERE session_id=?", (session_id,))
    conn.commit()
