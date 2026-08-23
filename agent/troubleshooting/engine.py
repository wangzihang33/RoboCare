from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable

import yaml

from agent.troubleshooting.models import (
    DiagnosisAction,
    DiagnosisState,
    DiagnosisStatus,
    DiagnosisTurn,
    HandoffTicket,
    StepAttempt,
)
from agent.troubleshooting.observation import (
    DiagnosisObservation,
    ObservationExtractor,
    ObservationKind,
)
from agent.troubleshooting.knowledge import DiagnosticEvidence
from agent.troubleshooting.store import DiagnosisStore
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path


_RISK_SIGNALS = (
    "冒烟",
    "起火",
    "烧焦味",
    "糊味",
    "发烫",
    "电池鼓包",
    "漏电",
    "触电",
    "机身进水",
    "电池外壳鼓起",
    "电池鼓起来",
    "进水以后还在通电",
    "麻感",
    "鼓胀变形",
    "电池仓鼓胀",
    "进水了却还连着电源",
)
_HUMAN_SIGNALS = (
    "转人工",
    "人工客服",
    "真人客服",
    "找个真人",
    "真人处理",
    "联系售后",
    "安排售后",
    "售后人员",
    "接人工",
    "真人吧",
    "直接和真人沟通",
    "叫售后",
    "客服人员接手",
    "转给售后",
    "别再自动排查",
)
_RESOLVED_SIGNALS = (
    "恢复正常",
    "已经好了",
    "正常了",
    "解决了",
    "可以了",
    "好了",
    "回去了",
    "不响了",
    "在线了",
    "声音完全消失",
    "能用了",
    "不再漏",
    "恢复了",
)
_FAILED_SIGNALS = (
    "还是不行",
    "仍然",
    "仍未",
    "没有解决",
    "没解决",
    "问题还在",
    "没有改善",
    "没改善",
    "无效",
    "没用",
    "依然",
    "还是有声音",
    "仍然漏",
    "依然漏",
)
_CANCEL_SIGNALS = (
    "结束诊断",
    "停止诊断",
    "不用排查了",
    "不用再检查了",
    "先暂停",
    "先到这里",
    "暂时不查",
    "先不查",
    "先把诊断关掉",
    "先停一下诊断",
    "不用继续",
    "先放着吧",
)
_MODEL_PATTERN = re.compile(
    r"(?:型号|机型)\s*[:：]?\s*([A-Za-z0-9][A-Za-z0-9_-]{0,15})",
    re.IGNORECASE,
)
_ERROR_PATTERN = re.compile(r"\bE\d{2,4}\b", re.IGNORECASE)
_NON_DIAGNOSTIC_ERROR_WORDS = re.compile(
    r"设备|机器(?:人)?|当前|现在|提示|显示|错误码|报错|报|出现|是|为|了"
)


class TroubleshootingEngine:
    def __init__(
        self,
        store: DiagnosisStore | None = None,
        playbook_path: str | Path | None = None,
        observation_fallback: Callable[
            [str, dict[str, Any]], DiagnosisObservation
        ] | None = None,
        knowledge_retriever: Callable[[str], list[DiagnosticEvidence | dict[str, Any]]] | None = None,
        knowledge_resolver: Callable[
            [str, list[DiagnosticEvidence], tuple[str, ...]], dict[str, Any]
        ] | None = None,
    ) -> None:
        self.store = store or DiagnosisStore()
        self.observation_fallback = observation_fallback
        self.knowledge_retriever = knowledge_retriever
        self.knowledge_resolver = knowledge_resolver
        configured = playbook_path or get_abs_path("config/troubleshooting.yml")
        self.playbook_path = Path(configured)
        self.playbooks = self._load_playbooks()

    def has_active_session(self, session_id: str) -> bool:
        return self.store.get_active_state(session_id) is not None

    @staticmethod
    def is_diagnostic_followup(message: str) -> bool:
        text = " ".join((message or "").split())
        signals = (
            _RESOLVED_SIGNALS
            + _FAILED_SIGNALS
            + _HUMAN_SIGNALS
            + _RISK_SIGNALS
            + _CANCEL_SIGNALS
        )
        return any(signal in text for signal in signals)

    @staticmethod
    def requires_immediate_handoff(message: str) -> bool:
        text = " ".join((message or "").split())
        return any(signal in text for signal in _RISK_SIGNALS + _HUMAN_SIGNALS)

    def handle(
        self,
        session_id: str,
        message: str,
        history: list[dict[str, Any]] | None = None,
    ) -> DiagnosisTurn:
        text = " ".join((message or "").split())
        state = self.store.get_active_state(session_id)
        is_new_case = state is None
        if state is None:
            state = DiagnosisState.start(session_id)
        state.turn_count += 1
        self._update_state_slots(state, text)
        if is_new_case:
            self._bootstrap_from_history(state, history)
        self._attach_knowledge_evidence(state, text)

        if any(signal in text for signal in _CANCEL_SIGNALS):
            return self._cancel(state)
        if state.risk_flags:
            if not state.symptom_text:
                state.symptom_text = text
            return self._escalate(state, "high_risk_signal")
        if any(signal in text for signal in _HUMAN_SIGNALS):
            if not state.symptom_text:
                state.symptom_text = text
            return self._escalate(state, "user_requested_handoff")

        if state.status is DiagnosisStatus.WAITING_FEEDBACK:
            observation = self._extract_step_observation(state, text)
            if observation.kind is ObservationKind.SUCCESS:
                self._set_current_outcome(state, "success", observation)
                state.status = DiagnosisStatus.RESOLVED
                state.current_step_id = ""
                self.store.save_state(state)
                return DiagnosisTurn(
                    action=DiagnosisAction.RESOLVE,
                    response="已记录设备恢复正常，本次故障诊断结束。",
                    state=state,
                )
            terminal_signal = self._match_terminal_failure(state, text)
            if (
                terminal_signal
                and observation.kind is ObservationKind.FAILURE
            ):
                terminal_observation = DiagnosisObservation(
                    kind=ObservationKind.FAILURE,
                    evidence_span=terminal_signal,
                    reason_code="advanced_step_failed",
                )
                self._set_current_outcome(state, "failed", terminal_observation)
                return self._escalate(state, "advanced_step_failed")
            if observation.kind is ObservationKind.FAILURE:
                self._set_current_outcome(state, "failed", observation)
                return self._give_next_step(state)
            self.store.save_state(state)
            return DiagnosisTurn(
                action=DiagnosisAction.ASK_FEEDBACK,
                response="请确认上一步执行结果：设备已经恢复，还是问题仍未解决？",
                state=state,
            )

        if not state.symptom_code:
            state.clarification_count += 1
            if state.clarification_count >= 2:
                return self._escalate(state, "insufficient_diagnostic_evidence")
            self.store.save_state(state)
            if state.error_code:
                response = (
                    f"暂未找到错误码 {state.error_code} 对应的已审核故障定义。"
                    "请补充设备的具体表现，例如是否无法启动、主刷不转、"
                    "无法回充或持续异响。"
                )
            else:
                response = "请描述具体故障现象；如有错误码，也请一并提供。"
            return DiagnosisTurn(
                action=DiagnosisAction.ASK_SYMPTOM,
                response=response,
                state=state,
            )

        return self._give_next_step(state)

    def _give_next_step(self, state: DiagnosisState) -> DiagnosisTurn:
        playbook = self.playbooks.get(state.symptom_code)
        if self._uses_action_groups(playbook):
            return self._give_next_action_group(state, playbook)

        attempted_ids = {attempt.step_id for attempt in state.attempts}
        next_step = next(
            (
                step
                for step in (playbook or {}).get("steps", [])
                if step["id"] not in attempted_ids
            ),
            None,
        )
        if next_step is None:
            return self._escalate(state, "steps_exhausted")

        attempt = StepAttempt(
            step_id=str(next_step["id"]),
            instruction=str(next_step["instruction"]),
        )
        state.attempts.append(attempt)
        state.current_step_id = attempt.step_id
        state.status = DiagnosisStatus.WAITING_FEEDBACK
        evidence_id = str(playbook.get("evidence_id") or "")
        if evidence_id and evidence_id not in state.evidence_ids:
            state.evidence_ids.append(evidence_id)
        self.store.save_state(state)
        evidence_text = f"（依据：{evidence_id}）" if evidence_id else ""
        return DiagnosisTurn(
            action=DiagnosisAction.GIVE_STEP,
            response=(
                f"请先{attempt.instruction}。完成后告诉我“已恢复”或“仍未解决”。"
                f"{evidence_text}"
            ),
            state=state,
        )

    @staticmethod
    def _uses_action_groups(playbook: dict[str, Any] | None) -> bool:
        return bool(playbook and isinstance(playbook.get("action_groups"), list))

    def _give_next_action_group(
        self,
        state: DiagnosisState,
        playbook: dict[str, Any],
    ) -> DiagnosisTurn:
        groups = [
            group
            for group in playbook.get("action_groups", [])
            if isinstance(group, dict) and str(group.get("id") or "").strip()
        ]
        group_by_id = {str(group["id"]): group for group in groups}
        attempted_ids = {attempt.step_id for attempt in state.attempts}

        if not state.attempts:
            next_group_id = str(groups[0]["id"]) if groups else ""
        else:
            current_group = group_by_id.get(state.current_step_id)
            transition = str((current_group or {}).get("on_failure") or "")
            if transition in {"", "handoff"}:
                return self._escalate(state, "action_groups_exhausted")
            if transition == "resolved" or transition in attempted_ids:
                return self._escalate(state, "action_groups_exhausted")
            next_group_id = transition

        next_group = group_by_id.get(next_group_id)
        if next_group is None:
            return self._escalate(state, "action_groups_exhausted")

        actions = [
            str(action).strip()
            for action in next_group.get("actions", [])
            if str(action).strip()
        ]
        instruction = "；".join(actions)
        attempt = StepAttempt(
            step_id=next_group_id,
            instruction=instruction,
        )
        state.attempts.append(attempt)
        state.current_step_id = next_group_id
        state.status = DiagnosisStatus.WAITING_FEEDBACK
        evidence_id = str(playbook.get("evidence_id") or "")
        if evidence_id and evidence_id not in state.evidence_ids:
            state.evidence_ids.append(evidence_id)
        self.store.save_state(state)
        evidence_text = f"（依据：{evidence_id}）" if evidence_id else ""
        return DiagnosisTurn(
            action=DiagnosisAction.GIVE_STEP,
            response=(
                f"请按以下顺序完成：{instruction}。"
                f"完成后告诉我“已恢复”或“仍未解决”。{evidence_text}"
            ),
            state=state,
        )

    def _escalate(self, state: DiagnosisState, reason: str) -> DiagnosisTurn:
        state.status = DiagnosisStatus.ESCALATED
        state.escalation_reason = reason
        state.current_step_id = ""
        self.store.save_state(state)
        ticket = HandoffTicket.from_state(state)
        self.store.save_handoff(ticket)

        attempted = [
            attempt.instruction for attempt in state.attempts if attempt.outcome != "pending"
        ]
        attempted_text = "；".join(attempted) if attempted else "尚未执行排障步骤"
        safety_text = (
            "请立即停止使用并断开设备电源。"
            if reason == "high_risk_signal"
            else ""
        )
        return DiagnosisTurn(
            action=DiagnosisAction.ESCALATE,
            response=(
                f"{safety_text}已生成转人工工单 {ticket.ticket_id}。"
                f"交接摘要：故障现象：{ticket.issue_summary}；"
                f"已尝试：{attempted_text}；升级原因：{reason}。"
            ),
            state=state,
            handoff=ticket,
        )

    def _cancel(self, state: DiagnosisState) -> DiagnosisTurn:
        state.status = DiagnosisStatus.CANCELLED
        state.current_step_id = ""
        self.store.save_state(state)
        return DiagnosisTurn(
            action=DiagnosisAction.CANCEL,
            response="已结束本次故障诊断。",
            state=state,
        )

    def _update_state_slots(self, state: DiagnosisState, text: str) -> None:
        model_match = _MODEL_PATTERN.search(text)
        if model_match:
            state.device_model = model_match.group(1)

        error_match = _ERROR_PATTERN.search(text)
        if error_match:
            state.error_code = error_match.group().upper()

        if not state.symptom_code and state.error_code:
            matched_code = self._match_error_code(state.error_code)
            if matched_code:
                state.symptom_code = matched_code
                state.symptom_text = text

        for signal in _RISK_SIGNALS:
            if signal in text and signal not in state.risk_flags:
                state.risk_flags.append(signal)

    def _attach_knowledge_evidence(self, state: DiagnosisState, text: str) -> None:
        # Initial fault normalization retrieves knowledge. Feedback turns are
        # interpreted against the current step and should not append unrelated
        # documents from a broad search.
        if self.knowledge_retriever is None or state.status is not DiagnosisStatus.COLLECTING:
            return
        # Do not spend a retrieval call on an error code or boilerplate alone.
        # The fixed clarification response must come before semantic search.
        if not self._has_fault_description(text):
            return

        query_parts = [text]
        if state.device_model:
            query_parts.append(f"型号 {state.device_model}")
        if state.error_code:
            query_parts.append(state.error_code)
        if state.symptom_text:
            query_parts.append(state.symptom_text)

        try:
            raw_evidence = self.knowledge_retriever(" ".join(query_parts)) or []
        except Exception:
            # Knowledge retrieval must not make the safety state machine fail.
            return

        evidence_items: list[DiagnosticEvidence] = []
        for item in raw_evidence:
            evidence = self._coerce_evidence(item)
            if evidence is None:
                continue
            evidence_items.append(evidence)
            if evidence.evidence_id not in state.evidence_ids:
                state.evidence_ids.append(evidence.evidence_id)

        # A known error code has already populated symptom_code and bypasses
        # this branch. Metadata labels remain model hints; every descriptive
        # unknown-code/no-code input uses the same constrained resolver.
        if (
            not state.symptom_code
            and self.knowledge_resolver
            and evidence_items
            and self._has_fault_description(text)
        ):
            try:
                candidate = self.knowledge_resolver(
                    text,
                    evidence_items,
                    tuple(self.playbooks),
                ) or {}
            except Exception:
                candidate = {}
            symptom_code = str(candidate.get("symptom_code") or "")
            if symptom_code in self.playbooks:
                state.symptom_code = symptom_code
                state.symptom_text = text
                evidence_id = str(candidate.get("evidence_id") or "")
                if evidence_id and evidence_id not in state.evidence_ids:
                    state.evidence_ids.append(evidence_id)

    @staticmethod
    def _coerce_evidence(
        value: DiagnosticEvidence | dict[str, Any],
    ) -> DiagnosticEvidence | None:
        if isinstance(value, DiagnosticEvidence):
            return value
        if not isinstance(value, dict):
            return None
        evidence_id = str(value.get("evidence_id") or value.get("id") or "").strip()
        content = str(value.get("content") or value.get("text") or "").strip()
        if not evidence_id or not content:
            return None
        metadata = value.get("metadata")
        return DiagnosticEvidence(
            evidence_id=evidence_id,
            content=content,
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
            symptom_code=str(value.get("symptom_code") or ""),
        )

    def _match_error_code(self, error_code: str) -> str:
        for code, playbook in self.playbooks.items():
            if error_code in {
                str(value).upper() for value in playbook.get("error_codes", [])
            }:
                return code
        return ""

    @staticmethod
    def _has_fault_description(text: str) -> bool:
        without_code = _ERROR_PATTERN.sub("", text)
        without_boilerplate = _NON_DIAGNOSTIC_ERROR_WORDS.sub("", without_code)
        meaningful = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", without_boilerplate)
        return bool(meaningful)

    def _extract_step_observation(
        self,
        state: DiagnosisState,
        text: str,
    ) -> DiagnosisObservation:
        step = self._current_step(state)
        if step is not None:
            fallback = (
                lambda message: self.observation_fallback(message, step)
                if self.observation_fallback
                else None
            )
            observation = ObservationExtractor.from_step(
                step,
                fallback=fallback,
            ).extract(text)
            if observation.kind is not ObservationKind.UNKNOWN:
                return observation

        if self._contains_any(text, _FAILED_SIGNALS):
            return DiagnosisObservation(
                kind=ObservationKind.FAILURE,
                evidence_span=self._first_matching_signal(text, _FAILED_SIGNALS),
                reason_code="step_failure",
            )
        if self._contains_any(text, _RESOLVED_SIGNALS):
            return DiagnosisObservation(
                kind=ObservationKind.SUCCESS,
                evidence_span=self._first_matching_signal(text, _RESOLVED_SIGNALS),
                reason_code="step_success",
            )
        return DiagnosisObservation(kind=ObservationKind.UNKNOWN)

    def _current_step(self, state: DiagnosisState) -> dict[str, Any] | None:
        playbook = self.playbooks.get(state.symptom_code) or {}
        if self._uses_action_groups(playbook):
            group = next(
                (
                    item
                    for item in playbook.get("action_groups", [])
                    if isinstance(item, dict)
                    and str(item.get("id") or "") == state.current_step_id
                ),
                None,
            )
            if group is None:
                return None
            verification = group.get("verification") or {}
            actions = [
                str(action).strip()
                for action in group.get("actions", [])
                if str(action).strip()
            ]
            return {
                "id": str(group.get("id") or ""),
                "instruction": "；".join(actions),
                "success_signals": verification.get("success_signals") or [],
                "failure_signals": verification.get("failure_signals") or [],
            }
        return next(
            (
                step
                for step in playbook.get("steps", [])
                if str(step.get("id")) == state.current_step_id
            ),
            None,
        )

    def _match_terminal_failure(self, state: DiagnosisState, text: str) -> str:
        playbook = self.playbooks.get(state.symptom_code) or {}
        signals = playbook.get("terminal_failure_signals") or []
        matches = [str(signal) for signal in signals if str(signal) in text]
        return max(matches, key=len, default="")

    def _bootstrap_from_history(
        self,
        state: DiagnosisState,
        history: list[dict[str, Any]] | None,
    ) -> None:
        max_messages = int(agent_conf.get("max_history_messages", 8))
        user_messages = [
            " ".join(str(item.get("content") or "").split())
            for item in (history or [])[-max_messages:]
            if item.get("role") == "user" and item.get("content")
        ]
        for text in reversed(user_messages):
            if not state.device_model:
                model_match = _MODEL_PATTERN.search(text)
                if model_match:
                    state.device_model = model_match.group(1)
            if not state.error_code:
                error_match = _ERROR_PATTERN.search(text)
                if error_match:
                    state.error_code = error_match.group().upper()
            if not state.symptom_code:
                matched_code = self._match_error_code(state.error_code)
                if matched_code:
                    state.symptom_code = matched_code
                    state.symptom_text = text
            if state.device_model and state.error_code and state.symptom_code:
                break

    @staticmethod
    def _set_current_outcome(
        state: DiagnosisState,
        outcome: str,
        observation: DiagnosisObservation | None = None,
    ) -> None:
        for attempt in reversed(state.attempts):
            if attempt.step_id == state.current_step_id:
                attempt.outcome = outcome
                if observation is not None:
                    attempt.evidence_span = observation.evidence_span
                    attempt.observation_source = observation.source
                return

    @staticmethod
    def _contains_any(text: str, signals: tuple[str, ...]) -> bool:
        return any(signal in text for signal in signals)

    @staticmethod
    def _first_matching_signal(text: str, signals: tuple[str, ...]) -> str:
        matches = [signal for signal in signals if signal in text]
        return max(matches, key=len, default="")

    def _load_playbooks(self) -> dict[str, dict[str, Any]]:
        with self.playbook_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        symptoms = payload.get("symptoms") or {}
        if not isinstance(symptoms, dict):
            raise ValueError("troubleshooting.yml 的 symptoms 必须是对象")
        return symptoms
