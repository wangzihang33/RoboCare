import sys

from agent.troubleshooting_evaluation import (
    DiagnosisEvalExample,
    DiagnosisEvalResult,
    aggregate_results,
    evaluate_examples,
    parse_args,
    save_ablation_report,
)
from agent.troubleshooting.engine import TroubleshootingEngine
from agent.troubleshooting.store import DiagnosisStore


def _evaluation_engine(db_path):
    def retrieve(query):
        if "回充" not in query:
            return []
        return [{"evidence_id": "TROUBLE-003", "content": "回充故障证据"}]

    def resolve(_query, _evidence, _allowed):
        return {"symptom_code": "cannot_recharge", "confidence": 0.95}

    return TroubleshootingEngine(
        DiagnosisStore(db_path),
        knowledge_retriever=retrieve,
        knowledge_resolver=resolve,
    )


def test_aggregates_multiturn_diagnosis_metrics():
    results = [
        DiagnosisEvalResult(
            session_id="s1",
            turn=1,
            query="无法回充",
            expected_action="give_step",
            predicted_action="give_step",
            expected_status="waiting_feedback",
            predicted_status="waiting_feedback",
            expected_symptom="cannot_recharge",
            predicted_symptom="cannot_recharge",
            expected_escalated=False,
            predicted_escalated=False,
            handoff_completeness=None,
        ),
        DiagnosisEvalResult(
            session_id="s2",
            turn=1,
            query="请转人工",
            expected_action="escalate",
            predicted_action="ask_symptom",
            expected_status="escalated",
            predicted_status="collecting",
            expected_symptom="",
            predicted_symptom="",
            expected_escalated=True,
            predicted_escalated=False,
            handoff_completeness=0.0,
        ),
    ]

    summary = aggregate_results(results)

    assert summary["case_count"] == 2
    assert summary["conversation_count"] == 2
    assert summary["action_accuracy"] == 0.5
    assert summary["state_accuracy"] == 0.5
    assert summary["symptom_accuracy"] == 1.0
    assert summary["escalation_accuracy"] == 0.5
    assert summary["handoff_completeness"] == 0.0


def test_conversation_metrics_use_final_terminal_and_followup_turns():
    results = [
        _result("s1", 1, "give_step", "give_step", "waiting_feedback", "waiting_feedback"),
        _result("s1", 2, "resolve", "resolve", "resolved", "resolved"),
        _result("s2", 1, "give_step", "give_step", "waiting_feedback", "waiting_feedback"),
        _result("s2", 2, "resolve", "ask_symptom", "resolved", "collecting"),
    ]

    summary = aggregate_results(results)

    assert summary["eligible_terminal_conversations"] == 2
    assert summary["task_completion_rate"] == 0.5
    assert summary["eligible_followup_turns"] == 2
    assert summary["unnecessary_repeat_question_rate"] == 0.5


def test_stateless_ablation_resets_diagnosis_state_each_turn(tmp_path):
    examples = [
        DiagnosisEvalExample(
            session_id="s1",
            turn=1,
            query="机器人无法回充",
            expected_action="give_step",
            expected_status="waiting_feedback",
            expected_symptom="cannot_recharge",
            expected_escalated=False,
        ),
        DiagnosisEvalExample(
            session_id="s1",
            turn=2,
            query="清理过了还是不行",
            expected_action="give_step",
            expected_status="waiting_feedback",
            expected_symptom="cannot_recharge",
            expected_escalated=False,
        ),
    ]
    stateful_engine = _evaluation_engine(tmp_path / "stateful.db")
    stateless_engine = _evaluation_engine(tmp_path / "stateless.db")

    stateful = evaluate_examples(
        stateful_engine,
        examples,
        preserve_session_state=True,
    )
    stateless = evaluate_examples(
        stateless_engine,
        examples,
        preserve_session_state=False,
    )

    assert stateful[1].predicted_action == "give_step"
    assert stateless[1].predicted_action == "ask_symptom"


def test_ablation_report_contains_both_configurations_and_delta(tmp_path):
    comparison = {
        "stateful": {
            "task_completion_rate": 0.9,
            "unnecessary_repeat_question_rate": 0.1,
            "avg_turns_to_terminal": 2.53,
        },
        "stateless": {
            "task_completion_rate": 0.4,
            "unnecessary_repeat_question_rate": 0.6,
            "avg_turns_to_terminal": 1.40,
        },
    }
    output = tmp_path / "ablation.md"

    save_ablation_report(
        comparison,
        output,
        tmp_path / "dataset.csv",
    )

    report = output.read_text(encoding="utf-8")
    assert "Stateful" in report
    assert "Stateless ablation" in report
    assert "task_completion_rate" in report
    assert "+50.00 pp" in report
    assert "-50.00 pp" in report
    assert "+1.13 turns" in report


def test_parse_args_supports_stateless_comparison(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["troubleshooting_evaluation", "--compare-stateless"],
    )

    args = parse_args()

    assert args.compare_stateless is True


def _result(
    session_id,
    turn,
    expected_action,
    predicted_action,
    expected_status,
    predicted_status,
):
    return DiagnosisEvalResult(
        session_id=session_id,
        turn=turn,
        query="test",
        expected_action=expected_action,
        predicted_action=predicted_action,
        expected_status=expected_status,
        predicted_status=predicted_status,
        expected_symptom="cannot_recharge",
        predicted_symptom="cannot_recharge",
        expected_escalated=False,
        predicted_escalated=False,
        handoff_completeness=None,
    )
