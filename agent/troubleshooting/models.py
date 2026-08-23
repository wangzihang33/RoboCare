from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
import uuid
from typing import Any


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


class DiagnosisStatus(StrEnum):
    COLLECTING = "collecting"
    WAITING_FEEDBACK = "waiting_feedback"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"


class DiagnosisAction(StrEnum):
    ASK_SYMPTOM = "ask_symptom"
    ASK_FEEDBACK = "ask_feedback"
    GIVE_STEP = "give_step"
    RESOLVE = "resolve"
    ESCALATE = "escalate"
    CANCEL = "cancel"


@dataclass
class StepAttempt:
    step_id: str
    instruction: str
    outcome: str = "pending"
    evidence_span: str = ""
    observation_source: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StepAttempt:
        return cls(
            step_id=str(value.get("step_id") or ""),
            instruction=str(value.get("instruction") or ""),
            outcome=str(value.get("outcome") or "pending"),
            evidence_span=str(value.get("evidence_span") or ""),
            observation_source=str(value.get("observation_source") or ""),
        )


@dataclass
class DiagnosisState:
    case_id: str
    session_id: str
    status: DiagnosisStatus = DiagnosisStatus.COLLECTING
    device_model: str = ""
    symptom_code: str = ""
    symptom_text: str = ""
    error_code: str = ""
    attempts: list[StepAttempt] = field(default_factory=list)
    current_step_id: str = ""
    risk_flags: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    escalation_reason: str = ""
    turn_count: int = 0
    clarification_count: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @classmethod
    def start(cls, session_id: str) -> DiagnosisState:
        return cls(
            case_id=f"case_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
        )

    @property
    def is_active(self) -> bool:
        return self.status in {
            DiagnosisStatus.COLLECTING,
            DiagnosisStatus.WAITING_FEEDBACK,
        }

    def touch(self) -> None:
        self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DiagnosisState:
        return cls(
            case_id=str(value["case_id"]),
            session_id=str(value["session_id"]),
            status=DiagnosisStatus(value.get("status", DiagnosisStatus.COLLECTING)),
            device_model=str(value.get("device_model") or ""),
            symptom_code=str(value.get("symptom_code") or ""),
            symptom_text=str(value.get("symptom_text") or ""),
            error_code=str(value.get("error_code") or ""),
            attempts=[
                StepAttempt.from_dict(item) for item in value.get("attempts", [])
            ],
            current_step_id=str(value.get("current_step_id") or ""),
            risk_flags=[str(item) for item in value.get("risk_flags", [])],
            evidence_ids=[str(item) for item in value.get("evidence_ids", [])],
            escalation_reason=str(value.get("escalation_reason") or ""),
            turn_count=int(value.get("turn_count", 0)),
            clarification_count=int(value.get("clarification_count", 0)),
            created_at=str(value.get("created_at") or _now()),
            updated_at=str(value.get("updated_at") or _now()),
        )


@dataclass
class HandoffTicket:
    ticket_id: str
    case_id: str
    session_id: str
    reason: str
    issue_summary: str
    device_model: str
    error_code: str
    attempted_steps: list[dict[str, str]]
    risk_flags: list[str]
    evidence_ids: list[str]
    created_at: str = field(default_factory=_now)

    @classmethod
    def from_state(cls, state: DiagnosisState) -> HandoffTicket:
        return cls(
            ticket_id=f"ticket_{uuid.uuid4().hex[:12]}",
            case_id=state.case_id,
            session_id=state.session_id,
            reason=state.escalation_reason or "manual_handoff",
            issue_summary=state.symptom_text or state.symptom_code or "故障现象未确认",
            device_model=state.device_model or "未提供",
            error_code=state.error_code or "未提供",
            attempted_steps=[asdict(item) for item in state.attempts],
            risk_flags=list(state.risk_flags),
            evidence_ids=list(state.evidence_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HandoffTicket:
        return cls(
            ticket_id=str(value["ticket_id"]),
            case_id=str(value["case_id"]),
            session_id=str(value["session_id"]),
            reason=str(value.get("reason") or "manual_handoff"),
            issue_summary=str(value.get("issue_summary") or "故障现象未确认"),
            device_model=str(value.get("device_model") or "未提供"),
            error_code=str(value.get("error_code") or "未提供"),
            attempted_steps=list(value.get("attempted_steps", [])),
            risk_flags=[str(item) for item in value.get("risk_flags", [])],
            evidence_ids=[str(item) for item in value.get("evidence_ids", [])],
            created_at=str(value.get("created_at") or _now()),
        )


@dataclass
class DiagnosisTurn:
    action: DiagnosisAction
    response: str
    state: DiagnosisState
    handoff: HandoffTicket | None = None
