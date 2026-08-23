from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable


class ObservationKind(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DiagnosisObservation:
    kind: ObservationKind
    evidence_span: str = ""
    reason_code: str = ""
    source: str = "rule"


class ObservationExtractor:
    def __init__(
        self,
        success_signals: tuple[str, ...] = (),
        failure_signals: tuple[str, ...] = (),
        fallback: Callable[[str], DiagnosisObservation] | None = None,
    ) -> None:
        self.success_signals = success_signals
        self.failure_signals = failure_signals
        self.fallback = fallback

    @classmethod
    def from_step(
        cls,
        step: dict[str, Any],
        fallback: Callable[[str], DiagnosisObservation] | None = None,
    ) -> ObservationExtractor:
        return cls(
            success_signals=_normalize_signals(step.get("success_signals")),
            failure_signals=_normalize_signals(step.get("failure_signals")),
            fallback=fallback,
        )

    def extract(self, message: str) -> DiagnosisObservation:
        text = " ".join((message or "").split())
        failure = _first_signal(text, self.failure_signals)
        if failure:
            return DiagnosisObservation(
                kind=ObservationKind.FAILURE,
                evidence_span=failure,
                reason_code="step_failure",
            )
        success = _first_signal(text, self.success_signals)
        if success:
            return DiagnosisObservation(
                kind=ObservationKind.SUCCESS,
                evidence_span=success,
                reason_code="step_success",
            )
        if self.fallback is not None:
            candidate = self.fallback(text)
            if isinstance(candidate, DiagnosisObservation):
                return candidate
        return DiagnosisObservation(kind=ObservationKind.UNKNOWN)


def _normalize_signals(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _first_signal(text: str, signals: tuple[str, ...]) -> str:
    matches = [signal for signal in signals if signal in text]
    return max(matches, key=len, default="")
