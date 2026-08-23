from agent.troubleshooting.observation import (
    DiagnosisObservation,
    ObservationKind,
    ObservationExtractor,
)
import agent.troubleshooting.knowledge as knowledge
import sys
import types


def test_step_contract_extracts_success_observation_with_evidence():
    extractor = ObservationExtractor.from_step(
        {
            "success_signals": ["声音完全消失", "不再有异响"],
            "failure_signals": ["还是有声音", "问题还在"],
        }
    )

    observation = extractor.extract("清理完以后声音完全消失了")

    assert observation == DiagnosisObservation(
        kind=ObservationKind.SUCCESS,
        evidence_span="声音完全消失",
        reason_code="step_success",
    )


def test_step_contract_extracts_failure_before_ambiguous_success_signal():
    extractor = ObservationExtractor.from_step(
        {
            "success_signals": ["正常"],
            "failure_signals": ["还是不正常"],
        }
    )

    observation = extractor.extract("清理后还是不正常")

    assert observation.kind is ObservationKind.FAILURE
    assert observation.evidence_span == "还是不正常"


def test_step_contract_keeps_unknown_observation_for_unverifiable_feedback():
    extractor = ObservationExtractor.from_step(
        {
            "success_signals": ["声音消失"],
            "failure_signals": ["仍然有声音"],
        }
    )

    observation = extractor.extract("我按照步骤操作了")

    assert observation.kind is ObservationKind.UNKNOWN
    assert observation.evidence_span == ""


def test_unknown_observation_can_use_injected_schema_constrained_fallback():
    calls = []

    def fallback(message):
        calls.append(message)
        return DiagnosisObservation(
            kind=ObservationKind.SUCCESS,
            evidence_span="已经能用了",
            reason_code="step_success",
            source="small_model",
        )

    extractor = ObservationExtractor.from_step(
        {"success_signals": [], "failure_signals": []},
        fallback=fallback,
    )

    observation = extractor.extract("现在已经能用了")

    assert observation.kind is ObservationKind.SUCCESS
    assert observation.source == "small_model"
    assert calls == ["现在已经能用了"]


def test_observer_model_uses_dedicated_lightweight_configuration(monkeypatch):
    captured = {}

    def fake_build_chat_model(model_name=None, **kwargs):
        captured["model_name"] = model_name
        captured.update(kwargs)
        return "observer-model"

    fake_factory = types.ModuleType("model.factory")
    fake_factory.build_chat_model = fake_build_chat_model
    monkeypatch.setitem(sys.modules, "model.factory", fake_factory)
    monkeypatch.setitem(knowledge.rag_conf, "diagnosis_observation_model", "deepseek-v4-flash")
    monkeypatch.setitem(knowledge.rag_conf, "diagnosis_observation_provider", "deepseek")
    monkeypatch.setitem(
        knowledge.rag_conf,
        "diagnosis_observation_api_key_env",
        "ROUTER_DEEPSEEK_API_KEY",
    )
    monkeypatch.setitem(
        knowledge.rag_conf,
        "diagnosis_observation_base_url",
        "https://api.deepseek.com",
    )

    assert knowledge.build_observation_model() == "observer-model"
    assert captured == {
        "model_name": "deepseek-v4-flash",
        "provider": "deepseek",
        "api_key_env": "ROUTER_DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
    }
