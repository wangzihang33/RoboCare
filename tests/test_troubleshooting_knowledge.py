import json

import pytest

from agent.troubleshooting.knowledge import (
    DiagnosticEvidence,
    LLMKnowledgeResolver,
    build_diagnostic_retriever,
)


class FakeModel:
    def __init__(self, payload):
        self.payload = payload
        self.prompt = ""

    def invoke(self, prompt):
        self.prompt = prompt
        return type(
            "Response",
            (),
            {"content": json.dumps(self.payload, ensure_ascii=False)},
        )()


def _evidence():
    return [
        DiagnosticEvidence(
            evidence_id="KB-BRUSH-001",
            content="主刷被毛发或线材缠绕后可能停止转动，需要断电清理。",
            symptom_code="main_brush_jam",
        )
    ]


def test_diagnostic_evaluation_mode_uses_rrf_without_reranker():
    calls = []

    class FakeVectorService:
        def get_fusion_retriever(self, *, k):
            calls.append(("rrf", k))
            return "rrf-retriever"

        def get_fusion_rerank_retriever(self, *, k, candidate_k):
            calls.append(("rerank", k, candidate_k))
            return "reranked-retriever"

    retriever = build_diagnostic_retriever(
        FakeVectorService(),
        top_k=3,
        use_reranker=False,
    )

    assert retriever == "rrf-retriever"
    assert calls == [("rrf", 3)]


def test_resolver_accepts_allowlisted_match_with_verifiable_evidence_quote():
    model = FakeModel(
        {
            "decision": "MATCH",
            "symptom_code": "main_brush_jam",
            "confidence": 0.93,
            "evidence_id": "KB-BRUSH-001",
            "evidence_span": "主刷被毛发或线材缠绕后可能停止转动",
        }
    )
    resolver = LLMKnowledgeResolver(model=model)

    result = resolver(
        "主刷被毛发缠住后完全不转",
        _evidence(),
        ("main_brush_jam", "cannot_recharge"),
    )

    assert result == {
        "decision": "MATCH",
        "symptom_code": "main_brush_jam",
        "confidence": 0.93,
        "evidence_id": "KB-BRUSH-001",
        "evidence_span": "主刷被毛发或线材缠绕后可能停止转动",
    }
    assert "候选标签: main_brush_jam" in model.prompt


@pytest.mark.parametrize(
    "payload",
    [
        {
            "decision": "AMBIGUOUS",
            "symptom_code": "",
            "confidence": 0.48,
            "evidence_id": "",
            "evidence_span": "",
        },
        {
            "decision": "MATCH",
            "symptom_code": "lidar_failure",
            "confidence": 0.95,
            "evidence_id": "KB-BRUSH-001",
            "evidence_span": "主刷被毛发或线材缠绕后可能停止转动",
        },
        {
            "decision": "MATCH",
            "symptom_code": "main_brush_jam",
            "confidence": 0.95,
            "evidence_id": "KB-NOT-RETRIEVED",
            "evidence_span": "主刷被毛发或线材缠绕后可能停止转动",
        },
        {
            "decision": "MATCH",
            "symptom_code": "main_brush_jam",
            "confidence": 0.95,
            "evidence_id": "KB-BRUSH-001",
            "evidence_span": "这段文字并不存在于召回证据中",
        },
    ],
)
def test_resolver_rejects_unverifiable_or_non_match_decisions(payload):
    resolver = LLMKnowledgeResolver(model=FakeModel(payload))

    result = resolver(
        "主刷被毛发缠住后完全不转",
        _evidence(),
        ("main_brush_jam", "cannot_recharge"),
    )

    assert result == {}
