from __future__ import annotations

import argparse
from collections.abc import Callable
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from agent.troubleshooting.engine import TroubleshootingEngine
from agent.troubleshooting.knowledge import (
    build_observation_model,
    LLMKnowledgeResolver,
    LLMObservationParser,
    LocalDiagnosticRetriever,
)
from agent.troubleshooting.models import DiagnosisStatus, HandoffTicket
from agent.troubleshooting.store import DiagnosisStore
from utils.path_tool import get_abs_path


DEFAULT_DATASET = "data/diagnosis_eval_dataset.csv"
DEFAULT_OUTPUT_DIR = "outputs/evaluations"


def build_evaluation_engine(
    db_path: Path,
    *,
    use_reranker: bool = True,
) -> TroubleshootingEngine:
    """Build the same RAG-backed diagnosis engine used by the application."""
    return TroubleshootingEngine(
        DiagnosisStore(db_path),
        observation_fallback=LLMObservationParser(model=build_observation_model()),
        knowledge_retriever=LocalDiagnosticRetriever(use_reranker=use_reranker),
        knowledge_resolver=LLMKnowledgeResolver(),
    )


@dataclass(frozen=True)
class DiagnosisEvalExample:
    session_id: str
    turn: int
    query: str
    expected_action: str
    expected_status: str
    expected_symptom: str
    expected_escalated: bool


@dataclass(frozen=True)
class DiagnosisEvalResult:
    session_id: str
    turn: int
    query: str
    expected_action: str
    predicted_action: str
    expected_status: str
    predicted_status: str
    expected_symptom: str
    predicted_symptom: str
    expected_escalated: bool
    predicted_escalated: bool
    handoff_completeness: float | None


def load_dataset(path: Path) -> list[DiagnosisEvalExample]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return [
            DiagnosisEvalExample(
                session_id=str(row["session_id"]).strip(),
                turn=int(row["turn"]),
                query=str(row["query"]).strip(),
                expected_action=str(row["expected_action"]).strip(),
                expected_status=str(row["expected_status"]).strip(),
                expected_symptom=str(row.get("expected_symptom") or "").strip(),
                expected_escalated=_parse_bool(row.get("expected_escalated")),
            )
            for row in rows
        ]


def evaluate_examples(
    engine: TroubleshootingEngine,
    examples: list[DiagnosisEvalExample],
    preserve_session_state: bool = True,
) -> list[DiagnosisEvalResult]:
    results: list[DiagnosisEvalResult] = []
    for example in examples:
        effective_session_id = example.session_id
        if not preserve_session_state:
            effective_session_id = f"{example.session_id}__turn_{example.turn}"
        turn = engine.handle(effective_session_id, example.query)
        predicted_escalated = turn.state.status is DiagnosisStatus.ESCALATED
        completeness = None
        if example.expected_escalated:
            completeness = _handoff_completeness(turn.handoff)
        results.append(
            DiagnosisEvalResult(
                session_id=example.session_id,
                turn=example.turn,
                query=example.query,
                expected_action=example.expected_action,
                predicted_action=turn.action.value,
                expected_status=example.expected_status,
                predicted_status=turn.state.status.value,
                expected_symptom=example.expected_symptom,
                predicted_symptom=turn.state.symptom_code,
                expected_escalated=example.expected_escalated,
                predicted_escalated=predicted_escalated,
                handoff_completeness=completeness,
            )
        )
    return results


def aggregate_results(results: list[DiagnosisEvalResult]) -> dict[str, float]:
    if not results:
        return {
            "case_count": 0,
            "conversation_count": 0,
            "action_accuracy": 0.0,
            "state_accuracy": 0.0,
            "symptom_accuracy": 0.0,
            "escalation_accuracy": 0.0,
            "handoff_completeness": 0.0,
            "avg_turns_to_terminal": 0.0,
            "eligible_terminal_conversations": 0,
            "task_completion_rate": 0.0,
            "eligible_followup_turns": 0,
            "unnecessary_repeat_question_rate": 0.0,
        }

    symptom_results = [item for item in results if item.expected_symptom]
    handoff_results = [
        item.handoff_completeness
        for item in results
        if item.handoff_completeness is not None
    ]
    terminal_turns: dict[str, int] = {}
    for item in results:
        if item.predicted_status in {
            DiagnosisStatus.RESOLVED.value,
            DiagnosisStatus.ESCALATED.value,
            DiagnosisStatus.CANCELLED.value,
        }:
            terminal_turns.setdefault(item.session_id, item.turn)

    final_results: dict[str, DiagnosisEvalResult] = {}
    for item in results:
        previous = final_results.get(item.session_id)
        if previous is None or item.turn > previous.turn:
            final_results[item.session_id] = item
    terminal_statuses = {
        DiagnosisStatus.RESOLVED.value,
        DiagnosisStatus.ESCALATED.value,
        DiagnosisStatus.CANCELLED.value,
    }
    eligible_terminal_results = [
        item
        for item in final_results.values()
        if item.expected_status in terminal_statuses
    ]
    followup_results = [item for item in results if item.turn > 1]
    unnecessary_repeat_questions = [
        item
        for item in followup_results
        if item.predicted_action in {
            "ask_symptom",
            "ask_feedback",
        }
        and item.predicted_action != item.expected_action
    ]

    return {
        "case_count": len(results),
        "conversation_count": len({item.session_id for item in results}),
        "action_accuracy": mean(
            item.expected_action == item.predicted_action for item in results
        ),
        "state_accuracy": mean(
            item.expected_status == item.predicted_status for item in results
        ),
        "symptom_accuracy": (
            mean(
                item.expected_symptom == item.predicted_symptom
                for item in symptom_results
            )
            if symptom_results
            else 0.0
        ),
        "escalation_accuracy": mean(
            item.expected_escalated == item.predicted_escalated for item in results
        ),
        "handoff_completeness": mean(handoff_results) if handoff_results else 0.0,
        "avg_turns_to_terminal": (
            mean(terminal_turns.values()) if terminal_turns else 0.0
        ),
        "eligible_terminal_conversations": len(eligible_terminal_results),
        "task_completion_rate": (
            mean(
                item.predicted_status == item.expected_status
                for item in eligible_terminal_results
            )
            if eligible_terminal_results
            else 0.0
        ),
        "eligible_followup_turns": len(followup_results),
        "unnecessary_repeat_question_rate": (
            len(unnecessary_repeat_questions) / len(followup_results)
            if followup_results
            else 0.0
        ),
    }


def evaluate_ablation(
    examples: list[DiagnosisEvalExample],
    engine_factory: Callable[[], TroubleshootingEngine],
) -> dict[str, dict[str, float]]:
    stateful_results = evaluate_examples(
        engine_factory(),
        examples,
        preserve_session_state=True,
    )
    stateless_results = evaluate_examples(
        engine_factory(),
        examples,
        preserve_session_state=False,
    )
    return {
        "stateful": aggregate_results(stateful_results),
        "stateless": aggregate_results(stateless_results),
    }


def save_report(
    results: list[DiagnosisEvalResult],
    summary: dict[str, float],
    output_path: Path,
    dataset_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 多轮故障诊断评测报告",
        "",
        f"- dataset: `{dataset_path}`",
        f"- total_turns: `{len(results)}`",
        f"- conversations: `{int(summary['conversation_count'])}`",
        "",
        "## 指标汇总",
        "",
        f"- action_accuracy: `{summary['action_accuracy']:.4f}`",
        f"- state_accuracy: `{summary['state_accuracy']:.4f}`",
        f"- symptom_accuracy: `{summary['symptom_accuracy']:.4f}`",
        f"- escalation_accuracy: `{summary['escalation_accuracy']:.4f}`",
        f"- handoff_completeness: `{summary['handoff_completeness']:.4f}`",
        f"- avg_turns_to_terminal: `{summary['avg_turns_to_terminal']:.4f}`",
        f"- task_completion_rate: `{summary['task_completion_rate']:.4f}`",
        "- unnecessary_repeat_question_rate: "
        f"`{summary['unnecessary_repeat_question_rate']:.4f}`",
        "",
        "## 单轮结果",
        "",
        "| session | turn | query | expected_action | predicted_action | expected_status | predicted_status | symptom | escalated |",
        "|---|---:|---|---|---|---|---|---|---:|",
    ]
    for item in results:
        query = item.query.replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {item.session_id} | {item.turn} | {query} | "
            f"{item.expected_action} | {item.predicted_action} | "
            f"{item.expected_status} | {item.predicted_status} | "
            f"{item.predicted_symptom} | {item.predicted_escalated} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_ablation_report(
    comparison: dict[str, dict[str, float]],
    output_path: Path,
    dataset_path: Path,
) -> None:
    stateful = comparison["stateful"]
    stateless = comparison["stateless"]
    ordered_metrics = (
        "task_completion_rate",
        "unnecessary_repeat_question_rate",
        "action_accuracy",
        "state_accuracy",
        "symptom_accuracy",
        "escalation_accuracy",
        "handoff_completeness",
        "avg_turns_to_terminal",
    )
    metrics = [
        metric
        for metric in ordered_metrics
        if metric in stateful and metric in stateless
    ]
    lines = [
        "# 多轮故障诊断状态消融评测",
        "",
        f"- dataset: `{dataset_path}`",
        f"- total_turns: `{int(stateful.get('case_count', 0))}`",
        f"- conversations: `{int(stateful.get('conversation_count', 0))}`",
        "- delta: `Stateful - Stateless ablation`",
        "",
        "## 指标定义",
        "",
        "- `task_completion_rate`: 最终应进入终态的会话中，预测终态正确的比例。",
        "- `unnecessary_repeat_question_rate`: 后续轮次中，不符合标准动作的症状或结果重复追问比例。",
        "- Stateless ablation 为每轮使用独立 session，不保留前序诊断状态的消融基线。",
        "",
        "## 对比结果",
        "",
        "| metric | Stateful | Stateless ablation | delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in metrics:
        stateful_value = float(stateful[metric])
        stateless_value = float(stateless[metric])
        if metric == "avg_turns_to_terminal":
            delta = f"{stateful_value - stateless_value:+.2f} turns"
        else:
            delta = f"{(stateful_value - stateless_value) * 100:+.2f} pp"
        lines.append(
            f"| {metric} | {stateful_value:.4f} | {stateless_value:.4f} | "
            f"{delta} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _handoff_completeness(ticket: HandoffTicket | None) -> float:
    if ticket is None:
        return 0.0
    required = (
        ticket.ticket_id,
        ticket.case_id,
        ticket.reason,
        ticket.issue_summary,
        ticket.device_model,
        ticket.error_code,
        ticket.attempted_steps is not None,
        ticket.risk_flags is not None,
        ticket.evidence_ids is not None,
    )
    return mean(bool(value) for value in required)


def _parse_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-turn troubleshooting.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compare-stateless", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(get_abs_path(args.dataset))
    examples = load_dataset(dataset_path)
    if not examples:
        raise ValueError(f"多轮诊断评测集为空: {dataset_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(get_abs_path(args.output_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.compare_stateless:
        with TemporaryDirectory(prefix="diagnosis_ablation_") as temp_dir:
            engine_index = 0

            def engine_factory() -> TroubleshootingEngine:
                nonlocal engine_index
                engine_index += 1
                return build_evaluation_engine(
                    Path(temp_dir) / f"diagnosis_eval_{engine_index}.db"
                )

            comparison = evaluate_ablation(examples, engine_factory)
        report_path = output_dir / f"diagnosis_state_ablation_{timestamp}.md"
        save_ablation_report(comparison, report_path, dataset_path)
        print(f"多轮故障诊断消融评测完成: {len(examples)} turns")
        print(f"Markdown: {report_path}")
        for metric in (
            "task_completion_rate",
            "unnecessary_repeat_question_rate",
        ):
            print(
                f"- {metric}: stateful={comparison['stateful'][metric]:.4f}, "
                f"stateless={comparison['stateless'][metric]:.4f}"
            )
        return

    with TemporaryDirectory(prefix="diagnosis_eval_") as temp_dir:
        engine = build_evaluation_engine(Path(temp_dir) / "diagnosis_eval.db")
        results = evaluate_examples(engine, examples)

    summary = aggregate_results(results)
    report_path = output_dir / f"diagnosis_eval_{timestamp}.md"
    save_report(results, summary, report_path, dataset_path)
    print(f"多轮故障诊断评测完成: {len(results)} turns")
    print(f"Markdown: {report_path}")
    for key in (
        "action_accuracy",
        "state_accuracy",
        "symptom_accuracy",
        "escalation_accuracy",
        "handoff_completeness",
        "avg_turns_to_terminal",
        "task_completion_rate",
        "unnecessary_repeat_question_rate",
    ):
        print(f"- {key}: {summary[key]:.4f}")


if __name__ == "__main__":
    main()
