from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from agent.troubleshooting.observation import DiagnosisObservation, ObservationKind
from utils.config_handler import rag_conf


def build_observation_model() -> Any:
    """Build the low-cost model used only for feedback observation parsing."""
    from model.factory import build_chat_model

    return build_chat_model(
        model_name=str(
            rag_conf.get("diagnosis_observation_model") or "deepseek-v4-flash"
        ),
        provider=str(rag_conf.get("diagnosis_observation_provider") or "deepseek"),
        api_key_env=str(
            rag_conf.get("diagnosis_observation_api_key_env")
            or rag_conf.get("chat_model_api_key_env")
            or "MAIN_DEEPSEEK_API_KEY"
        ),
        base_url=str(
            rag_conf.get("diagnosis_observation_base_url")
            or rag_conf.get("chat_model_base_url")
            or ""
        )
        or None,
    )


@dataclass(frozen=True)
class DiagnosticEvidence:
    """A retrieved knowledge fragment attached to a diagnosis case."""

    evidence_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    symptom_code: str = ""


def build_diagnostic_retriever(
    service: Any,
    *,
    top_k: int,
    use_reranker: bool = True,
    rerank_candidate_k: int | None = None,
):
    """Build the diagnosis retriever, with an explicit fast RRF-only mode."""
    if not use_reranker:
        return service.get_fusion_retriever(k=top_k)
    try:
        return service.get_fusion_rerank_retriever(
            k=top_k,
            candidate_k=rerank_candidate_k or max(top_k, 20),
        )
    except Exception:
        return service.get_fusion_retriever(k=top_k)


class LocalDiagnosticRetriever:
    """Retrieve troubleshooting evidence from the existing local Hybrid RAG."""

    def __init__(
        self,
        top_k: int = 5,
        rerank_candidate_k: int = 20,
        *,
        use_reranker: bool = True,
    ) -> None:
        self.top_k = max(1, int(top_k))
        self.rerank_candidate_k = max(self.top_k, int(rerank_candidate_k))
        self.use_reranker = bool(use_reranker)
        self._retriever = None
        self._fallback_retriever = None
        self._initialized = False

    def __call__(self, query: str) -> list[DiagnosticEvidence]:
        if not query.strip():
            return []
        if not self._initialized:
            from rag.vector_store import VectorStoreService

            service = VectorStoreService()
            self._retriever = build_diagnostic_retriever(
                service,
                top_k=self.top_k,
                use_reranker=self.use_reranker,
                rerank_candidate_k=self.rerank_candidate_k,
            )
            if self.use_reranker:
                self._fallback_retriever = service.get_fusion_retriever(k=self.top_k)
            self._initialized = True

        try:
            documents = self._retriever(query) if self._retriever else []
        except Exception:
            # Reranking is an optimization, not a safety dependency. A
            # transient rerank/API failure falls back to deterministic RRF.
            self._retriever = None
            documents = []
        if not documents and self._fallback_retriever is not None:
            documents = self._fallback_retriever(query) or []
        evidence: list[DiagnosticEvidence] = []
        for index, document in enumerate(documents):
            metadata = dict(getattr(document, "metadata", {}) or {})
            content = str(getattr(document, "page_content", "") or "").strip()
            if not content:
                continue
            evidence_id = str(
                metadata.get("doc_id")
                or metadata.get("card_id")
                or metadata.get("source")
                or f"local-rag-{index}"
            )
            evidence.append(
                DiagnosticEvidence(
                    evidence_id=evidence_id,
                    content=content,
                    metadata=metadata,
                    symptom_code=str(metadata.get("symptom_code") or ""),
                )
            )
        return evidence


class LLMKnowledgeResolver:
    """Map retrieved evidence to a known playbook with constrained JSON output."""

    def __init__(self, model: Any | None = None, min_confidence: float = 0.75) -> None:
        self.model = model
        self.min_confidence = min_confidence

    def __call__(
        self,
        query: str,
        evidence: list[DiagnosticEvidence],
        allowed_symptoms: tuple[str, ...],
    ) -> dict[str, Any]:
        if not evidence or not allowed_symptoms:
            return {}
        model = self.model or self._load_model()
        context = "\n\n".join(
            (
                f"[{item.evidence_id}] 候选标签: "
                f"{item.symptom_code or '未标注'}\n{item.content[:1800]}"
            )
            for item in evidence[:6]
        )
        prompt = (
            "你是客服故障归一器。只能根据给定资料判断故障类型，禁止补充资料外的事实。\n"
            "如果用户只提供了资料中未明确说明的错误码，必须返回空字符串；"
            "不得根据错误码格式猜测故障类型。\n"
            "候选标签只作为提示，不能替代用户描述与证据原文的一致性判断。\n"
            "只有用户描述和某条证据共同支持唯一故障类型时返回 MATCH；"
            "多个类型均可能时返回 AMBIGUOUS；没有证据支持时返回 NO_MATCH。\n"
            "MATCH 时 evidence_span 必须逐字引用对应 evidence_id 中的原文。\n"
            f"允许的故障类型: {', '.join(allowed_symptoms)}\n"
            f"用户描述: {query}\n"
            f"检索资料:\n{context}\n\n"
            "只输出 JSON，不要 Markdown。格式为："
            '{"decision":"MATCH、AMBIGUOUS 或 NO_MATCH",'
            '"symptom_code":"允许列表中的值或空字符串",'
            '"confidence":0到1之间的数字,'
            '"evidence_id":"资料ID或空字符串",'
            '"evidence_span":"资料原文引用或空字符串"}'
        )
        try:
            response = model.invoke(prompt)
            content = getattr(response, "content", response)
            payload = self._parse_json(str(content or ""))
        except Exception:
            return {}

        decision = str(payload.get("decision") or "").upper()
        if decision != "MATCH":
            return {}
        symptom_code = str(payload.get("symptom_code") or "")
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if symptom_code not in allowed_symptoms or confidence < self.min_confidence:
            return {}
        evidence_id = str(payload.get("evidence_id") or "")
        evidence_span = str(payload.get("evidence_span") or "").strip()
        evidence_by_id = {item.evidence_id: item for item in evidence}
        selected_evidence = evidence_by_id.get(evidence_id)
        if selected_evidence is None or not self._contains_span(
            selected_evidence.content,
            evidence_span,
        ):
            return {}
        return {
            "decision": decision,
            "symptom_code": symptom_code,
            "confidence": confidence,
            "evidence_id": evidence_id,
            "evidence_span": evidence_span,
        }

    @staticmethod
    def _contains_span(content: str, evidence_span: str) -> bool:
        if not evidence_span:
            return False
        normalized_content = " ".join(content.split())
        normalized_span = " ".join(evidence_span.split())
        return normalized_span in normalized_content

    @staticmethod
    def _load_model() -> Any:
        from model.factory import chat_model

        return chat_model

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        try:
            value = json.loads(content)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                return {}
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
            return value if isinstance(value, dict) else {}


class LLMObservationParser:
    """Interpret free-form step feedback without changing the state machine."""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def __call__(
        self,
        message: str,
        step: dict[str, Any],
    ) -> DiagnosisObservation:
        model = self.model or LLMKnowledgeResolver._load_model()
        prompt = (
            "你是客服故障排障反馈解析器。只能判断用户对当前步骤的反馈，禁止编造事实。\n"
            f"当前步骤: {step.get('instruction', '')}\n"
            f"成功标准: {', '.join(map(str, step.get('success_signals') or []))}\n"
            f"失败标准: {', '.join(map(str, step.get('failure_signals') or []))}\n"
            f"用户反馈: {message}\n\n"
            "只输出 JSON："
            '{"outcome":"SUCCESS、FAILURE 或 UNKNOWN",'
            '"confidence":0到1之间的数字,"evidence_span":"原文中的依据"}'
        )
        try:
            response = model.invoke(prompt)
            payload = LLMKnowledgeResolver._parse_json(
                str(getattr(response, "content", response) or "")
            )
        except Exception:
            return DiagnosisObservation(kind=ObservationKind.UNKNOWN, source="small_model")

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        outcome = str(payload.get("outcome") or "").upper()
        if confidence < 0.75:
            outcome = "UNKNOWN"
        kind = {
            "SUCCESS": ObservationKind.SUCCESS,
            "FAILURE": ObservationKind.FAILURE,
        }.get(outcome, ObservationKind.UNKNOWN)
        return DiagnosisObservation(
            kind=kind,
            evidence_span=str(payload.get("evidence_span") or ""),
            reason_code="small_model_observation",
            source="small_model",
        )
