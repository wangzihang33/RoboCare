from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from agent.troubleshooting.models import (
    DiagnosisState,
    DiagnosisStatus,
    HandoffTicket,
)
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path


class DiagnosisStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        configured = db_path or agent_conf.get(
            "diagnosis_db_path", "data/external/diagnosis_memory.db"
        )
        path = Path(configured)
        self.db_path = path if path.is_absolute() else Path(get_abs_path(str(path)))

    def save_state(self, state: DiagnosisState) -> None:
        self.ensure_initialized()
        state.touch()
        payload = json.dumps(state.to_dict(), ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO diagnosis_cases (
                    case_id, session_id, status, state_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    status = excluded.status,
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (
                    state.case_id,
                    state.session_id,
                    state.status.value,
                    payload,
                    state.created_at,
                    state.updated_at,
                ),
            )

    def get_active_state(self, session_id: str) -> DiagnosisState | None:
        self.ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT state_json
                FROM diagnosis_cases
                WHERE session_id = ? AND status IN (?, ?)
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (
                    session_id,
                    DiagnosisStatus.COLLECTING.value,
                    DiagnosisStatus.WAITING_FEEDBACK.value,
                ),
            ).fetchone()
        return self._state_from_row(row)

    def get_latest_state(self, session_id: str) -> DiagnosisState | None:
        self.ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT state_json
                FROM diagnosis_cases
                WHERE session_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return self._state_from_row(row)

    def save_handoff(self, ticket: HandoffTicket) -> None:
        self.ensure_initialized()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO handoff_tickets (
                    ticket_id, case_id, session_id, reason, ticket_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket.ticket_id,
                    ticket.case_id,
                    ticket.session_id,
                    ticket.reason,
                    json.dumps(ticket.to_dict(), ensure_ascii=False),
                    ticket.created_at,
                ),
            )

    def get_handoff(self, ticket_id: str) -> HandoffTicket | None:
        self.ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ticket_json FROM handoff_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
        if row is None:
            return None
        return HandoffTicket.from_dict(json.loads(row["ticket_json"]))

    def ensure_initialized(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnosis_cases (
                    case_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_diagnosis_cases_session_status
                ON diagnosis_cases(session_id, status, updated_at)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS handoff_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    ticket_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES diagnosis_cases(case_id)
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _state_from_row(row: sqlite3.Row | None) -> DiagnosisState | None:
        if row is None:
            return None
        return DiagnosisState.from_dict(json.loads(row["state_json"]))
