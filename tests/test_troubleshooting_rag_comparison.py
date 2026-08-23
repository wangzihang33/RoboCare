from agent.troubleshooting_evaluation import DiagnosisEvalExample
import sys

from agent.troubleshooting_rag_comparison import (
    BaselineJudgement,
    ConversationReply,
    LLMReplyJudge,
    aggregate_rag_baseline_results,
    build_local_rag_responder,
    evaluate_rag_baseline,
    evaluate_stateful_diagnosis,
    parse_baseline_judgement,
    parse_args,
    respond_with_local_rag,
    save_rag_comparison_details,
    save_rag_comparison_report,
    select_conversations,
)
from agent.troubleshooting.models import DiagnosisAction, DiagnosisState, DiagnosisStatus


def _example(
    session_id,
    turn,
    query,
    expected_action,
    expected_status,
):
    return DiagnosisEvalExample(
        session_id=session_id,
        turn=turn,
        query=query,
        expected_action=expected_action,
        expected_status=expected_status,
        expected_symptom="cannot_recharge",
        expected_escalated=expected_status == "escalated",
    )


def test_rag_baseline_sends_only_the_current_query_to_the_responder():
    examples = [
        _example("s1", 1, "机器人无法回充", "give_step", "waiting_feedback"),
        _example("s1", 2, "清理过了还是不行", "give_step", "waiting_feedback"),
    ]
    responder_calls = []
    judge_history_lengths = []

    def responder(query):
        responder_calls.append(query)
        return f"answer:{query}"

    def judge(query, response, history):
        judge_history_lengths.append(len(history))
        return BaselineJudgement(
            predicted_action="give_step",
            unnecessary_repeat=bool(history),
            reason="test",
        )

    results = evaluate_rag_baseline(examples, responder, judge)

    assert responder_calls == ["机器人无法回充", "清理过了还是不行"]
    assert judge_history_lengths == [0, 1]
    assert results[1].unnecessary_repeat is True


def test_rag_baseline_metrics_use_final_terminal_turn_and_followups():
    examples = [
        _example("s1", 1, "机器人无法回充", "give_step", "waiting_feedback"),
        _example("s1", 2, "现在恢复了", "resolve", "resolved"),
        _example("s2", 1, "设备持续异响", "give_step", "waiting_feedback"),
        _example("s2", 2, "还是没有改善", "escalate", "escalated"),
    ]
    predictions = iter(
        [
            BaselineJudgement("give_step", False, "给出建议"),
            BaselineJudgement("resolve", False, "确认恢复"),
            BaselineJudgement("give_step", False, "给出建议"),
            BaselineJudgement("give_step", True, "重复先前建议"),
        ]
    )

    results = evaluate_rag_baseline(
        examples,
        lambda query: f"answer:{query}",
        lambda query, response, history: next(predictions),
    )
    summary = aggregate_rag_baseline_results(results)

    assert summary["action_accuracy"] == 0.75
    assert summary["task_completion_rate"] == 0.5
    assert summary["eligible_followup_turns"] == 2
    assert summary["unnecessary_repeat_rate"] == 0.5


def test_rag_comparison_report_names_the_actual_baseline(tmp_path):
    comparison = {
        "stateful": {
            "task_completion_rate": 0.8,
            "unnecessary_repeat_rate": 0.1,
            "action_accuracy": 0.9,
        },
        "rag_baseline": {
            "task_completion_rate": 0.5,
            "unnecessary_repeat_rate": 0.4,
            "action_accuracy": 0.6,
        },
    }
    output = tmp_path / "comparison.md"

    save_rag_comparison_report(
        comparison,
        output,
        tmp_path / "dataset.csv",
        judge_model="judge-model",
    )

    report = output.read_text(encoding="utf-8")
    assert "Stateful diagnosis" in report
    assert "Single-turn Hybrid RAG" in report
    assert "同一知识库" in report
    assert "+30.00 pp" in report
    assert "-30.00 pp" in report
    assert "unnecessary_repeat_rate" in report
    assert "action_accuracy" not in report


def test_parses_constrained_judge_json_from_a_fenced_response():
    judgement = parse_baseline_judgement(
        """```json
        {
          "predicted_action": "escalate",
          "unnecessary_repeat": false,
          "reason": "建议停止操作并联系人工支持"
        }
        ```"""
    )

    assert judgement.predicted_action == "escalate"
    assert judgement.unnecessary_repeat is False


def test_stateful_comparison_uses_engine_action_but_shared_repeat_judge():
    examples = [
        _example("s1", 1, "机器人无法回充", "give_step", "waiting_feedback"),
        _example("s1", 2, "还是不行", "give_step", "waiting_feedback"),
    ]

    class FakeEngine:
        def handle(self, session_id, query):
            state = DiagnosisState.start(session_id)
            state.status = DiagnosisStatus.WAITING_FEEDBACK
            return type(
                "Turn",
                (),
                {
                    "action": DiagnosisAction.GIVE_STEP,
                    "response": f"next:{query}",
                    "state": state,
                },
            )()

    judgements = iter(
        [
            BaselineJudgement("unknown", False, "首次建议"),
            BaselineJudgement("unknown", True, "重复了已经失败的建议"),
        ]
    )

    results = evaluate_stateful_diagnosis(
        examples,
        FakeEngine(),
        lambda query, response, history: next(judgements),
    )

    assert results[0].predicted_action == "give_step"
    assert results[1].predicted_status == "waiting_feedback"
    assert results[1].unnecessary_repeat is True


def test_llm_judge_receives_prior_transcript_and_returns_constrained_result():
    prompts = []

    class FakeResponse:
        content = (
            '{"predicted_action":"give_step",'
            '"unnecessary_repeat":true,"reason":"重复了清理触点"}'
        )

    class FakeModel:
        def invoke(self, prompt):
            prompts.append(prompt)
            return FakeResponse()

    judge = LLMReplyJudge(FakeModel())
    result = judge(
        "还是不行",
        "请再次清理充电触点",
        (ConversationReply("无法回充", "请清理充电触点"),),
    )

    assert result.predicted_action == "give_step"
    assert result.unnecessary_repeat is True
    assert "请清理充电触点" in prompts[0]
    assert "还是不行" in prompts[0]


def test_llm_judge_retries_then_records_a_failure_without_aborting():
    class FailingModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            raise RuntimeError("temporary provider failure")

    model = FailingModel()
    judge = LLMReplyJudge(
        model,
        max_retries=2,
        retry_delay_seconds=0,
    )

    result = judge("无法回充", "请稍后重试", ())

    assert model.calls == 3
    assert result.predicted_action == "unknown"
    assert "temporary provider failure" in result.error


def test_local_rag_responder_extracts_answer_from_tool_contract():
    calls = []

    def rag_tool(query):
        calls.append(query)
        return {
            "ok": True,
            "data": {"answer": "请检查并清理主刷。"},
            "error": None,
        }

    response = respond_with_local_rag("主刷卡住了", rag_tool=rag_tool)

    assert response == "请检查并清理主刷。"
    assert calls == ["主刷卡住了"]


def test_evaluation_responder_can_disable_reranker(monkeypatch):
    import rag.rag_service as rag_service

    modes = []

    class FakeRagService:
        def __init__(self, *, use_reranker):
            modes.append(use_reranker)

        def rag_summarize(self, query):
            return f"answer:{query}"

    monkeypatch.setattr(rag_service, "RagSummarizeservice", FakeRagService)

    responder = build_local_rag_responder(use_reranker=False)

    assert responder("无法回充") == "answer:无法回充"
    assert modes == [False]


def test_conversation_limit_keeps_whole_sessions():
    examples = [
        _example("s1", 1, "a", "give_step", "waiting_feedback"),
        _example("s1", 2, "b", "resolve", "resolved"),
        _example("s2", 1, "c", "give_step", "waiting_feedback"),
    ]

    selected = select_conversations(examples, max_conversations=1)

    assert [(item.session_id, item.turn) for item in selected] == [
        ("s1", 1),
        ("s1", 2),
    ]


def test_comparison_details_preserve_both_system_outputs(tmp_path):
    examples = [
        _example("s1", 1, "无法回充", "give_step", "waiting_feedback"),
    ]
    stateful = evaluate_rag_baseline(
        examples,
        lambda query: "状态化回复",
        lambda query, response, history: BaselineJudgement(
            "give_step", False, "stateful"
        ),
    )
    baseline = evaluate_rag_baseline(
        examples,
        lambda query: "普通 RAG 回复",
        lambda query, response, history: BaselineJudgement(
            "give_step", False, "baseline"
        ),
    )
    output = tmp_path / "details.csv"

    save_rag_comparison_details(stateful, baseline, output)

    text = output.read_text(encoding="utf-8-sig")
    assert "stateful,s1,1,无法回充,状态化回复" in text
    assert "rag_baseline,s1,1,无法回充,普通 RAG 回复" in text


def test_parse_args_supports_full_conversation_trial(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "troubleshooting_rag_comparison",
            "--dataset",
            "data/test.csv",
            "--max-conversations",
            "2",
            "--judge-model",
            "deepseek-v4-flash",
        ],
    )

    args = parse_args()

    assert args.dataset == "data/test.csv"
    assert args.max_conversations == 2
    assert args.judge_model == "deepseek-v4-flash"
