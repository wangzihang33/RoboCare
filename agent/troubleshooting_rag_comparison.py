from __future__ import annotations

import argparse
from collections.abc import Callable
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from statistics import mean
from tempfile import TemporaryDirectory
import time
from typing import Any

from agent.troubleshooting.engine import TroubleshootingEngine
from agent.troubleshooting.models import DiagnosisStatus
from agent.troubleshooting_evaluation import (
    build_evaluation_engine,
    DiagnosisEvalExample,
    load_dataset,
)
from utils.config_handler import rag_conf
from utils.path_tool import get_abs_path


DEFAULT_DATASET = "data/diagnosis_post_contract_acceptance.csv"
DEFAULT_OUTPUT_DIR = "outputs/evaluations"


_ACTION_TO_STATUS = {
    "ask_symptom": DiagnosisStatus.COLLECTING.value,
    "ask_feedback": DiagnosisStatus.WAITING_FEEDBACK.value,
    "give_step": DiagnosisStatus.WAITING_FEEDBACK.value,
    "resolve": DiagnosisStatus.RESOLVED.value,
    "escalate": DiagnosisStatus.ESCALATED.value,
    "cancel": DiagnosisStatus.CANCELLED.value,
}


@dataclass(frozen=True)
class BaselineJudgement:
    predicted_action: str
    unnecessary_repeat: bool
    reason: str
    error: str = ""


@dataclass(frozen=True)
class ConversationReply:
    query: str
    response: str


@dataclass(frozen=True)
class RAGBaselineResult:
    session_id: str
    turn: int
    query: str
    response: str
    expected_action: str
    predicted_action: str
    expected_status: str
    predicted_status: str
    unnecessary_repeat: bool
    judge_reason: str
    judge_error: str = ""


RAGResponder = Callable[[str], str]
ReplyJudge = Callable[
    [str, str, tuple[ConversationReply, ...]],
    BaselineJudgement,
]


class LLMReplyJudge:
    def __init__(
        self,
        model: Any,
        *,
        max_history_turns: int = 4,
        max_retries: int = 2,
        retry_delay_seconds: float = 1.0,
    ):
        self.model = model
        self.max_history_turns = max(1, int(max_history_turns))
        self.max_retries = max(0, int(max_retries))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))

    def __call__(
        self,
        query: str,
        response: str,
        history: tuple[ConversationReply, ...],
    ) -> BaselineJudgement:
        prompt = _build_judge_prompt(
            query=query,
            response=response,
            history=history[-self.max_history_turns :],
        )
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                model_response = self.model.invoke(prompt)
                content = getattr(model_response, "content", model_response)
                if isinstance(content, list):
                    content = "".join(
                        str(item.get("text", ""))
                        if isinstance(item, dict)
                        else str(item)
                        for item in content
                    )
                return parse_baseline_judgement(str(content))
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries and self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds * (2**attempt))
        return BaselineJudgement(
            predicted_action="unknown",
            unnecessary_repeat=False,
            reason="Judge 调用失败，未进行动作归类",
            error=last_error,
        )


def respond_with_local_rag(
    query: str,
    *,
    rag_tool: Callable[[str], dict[str, Any]] | None = None,
) -> str:
    """Call the same single-turn RAG tool used by the LOCAL_RAG route."""
    if rag_tool is None:
        from agent.tools.core_tools import rag_summarize_core

        rag_tool = rag_summarize_core
    result = rag_tool(query)
    if not result.get("ok"):
        error = result.get("error") or {}
        raise RuntimeError(str(error.get("message") or "本地 RAG 调用失败"))
    answer = str((result.get("data") or {}).get("answer") or "").strip()
    if not answer:
        raise RuntimeError("本地 RAG 返回空答案")
    return answer


def build_local_rag_responder(*, use_reranker: bool = True) -> RAGResponder:
    """Build a direct LOCAL_RAG responder for controlled evaluation runs."""
    from rag.rag_service import RagSummarizeservice

    service = RagSummarizeservice(use_reranker=use_reranker)
    return service.rag_summarize


def select_conversations(
    examples: list[DiagnosisEvalExample],
    *,
    max_conversations: int | None,
) -> list[DiagnosisEvalExample]:
    if max_conversations is None:
        return list(examples)
    if max_conversations <= 0:
        raise ValueError("max_conversations 必须大于 0")
    selected_sessions: list[str] = []
    for example in examples:
        if example.session_id not in selected_sessions:
            if len(selected_sessions) >= max_conversations:
                break
            selected_sessions.append(example.session_id)
    allowed = set(selected_sessions)
    return [item for item in examples if item.session_id in allowed]


def evaluate_rag_baseline(
    examples: list[DiagnosisEvalExample],
    responder: RAGResponder,
    judge: ReplyJudge,
) -> list[RAGBaselineResult]:
    """Evaluate a strict single-turn RAG responder on a multi-turn dataset.

    The responder receives only the current user query. Conversation history is
    retained exclusively for the evaluator to identify repeated guidance.
    """
    histories: dict[str, list[ConversationReply]] = {}
    results: list[RAGBaselineResult] = []

    for example in examples:
        history = histories.setdefault(example.session_id, [])
        response = str(responder(example.query)).strip()
        judgement = judge(example.query, response, tuple(history))
        predicted_action = _normalize_action(judgement.predicted_action)
        results.append(
            RAGBaselineResult(
                session_id=example.session_id,
                turn=example.turn,
                query=example.query,
                response=response,
                expected_action=example.expected_action,
                predicted_action=predicted_action,
                expected_status=example.expected_status,
                predicted_status=_ACTION_TO_STATUS.get(predicted_action, "unknown"),
                unnecessary_repeat=bool(judgement.unnecessary_repeat),
                judge_reason=judgement.reason,
                judge_error=judgement.error,
            )
        )
        history.append(ConversationReply(query=example.query, response=response))

    return results


def evaluate_stateful_diagnosis(
    examples: list[DiagnosisEvalExample],
    engine: TroubleshootingEngine,
    judge: ReplyJudge,
) -> list[RAGBaselineResult]:
    """Evaluate engine actions while using the shared judge only for repetition."""
    histories: dict[str, list[ConversationReply]] = {}
    results: list[RAGBaselineResult] = []

    for example in examples:
        history = histories.setdefault(example.session_id, [])
        turn = engine.handle(example.session_id, example.query)
        response = str(turn.response).strip()
        judgement = judge(example.query, response, tuple(history))
        results.append(
            RAGBaselineResult(
                session_id=example.session_id,
                turn=example.turn,
                query=example.query,
                response=response,
                expected_action=example.expected_action,
                predicted_action=turn.action.value,
                expected_status=example.expected_status,
                predicted_status=turn.state.status.value,
                unnecessary_repeat=bool(judgement.unnecessary_repeat),
                judge_reason=judgement.reason,
                judge_error=judgement.error,
            )
        )
        history.append(ConversationReply(query=example.query, response=response))

    return results


def aggregate_rag_baseline_results(
    results: list[RAGBaselineResult],
) -> dict[str, float]:
    if not results:
        return {
            "case_count": 0,
            "conversation_count": 0,
            "action_accuracy": 0.0,
            "eligible_terminal_conversations": 0,
            "task_completion_rate": 0.0,
            "eligible_followup_turns": 0,
            "unnecessary_repeat_rate": 0.0,
            "judge_error_count": 0,
        }

    final_results: dict[str, RAGBaselineResult] = {}
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

    return {
        "case_count": len(results),
        "conversation_count": len({item.session_id for item in results}),
        "action_accuracy": mean(
            item.expected_action == item.predicted_action for item in results
        ),
        "eligible_terminal_conversations": len(eligible_terminal_results),
        "task_completion_rate": (
            mean(
                item.expected_status == item.predicted_status
                for item in eligible_terminal_results
            )
            if eligible_terminal_results
            else 0.0
        ),
        "eligible_followup_turns": len(followup_results),
        "unnecessary_repeat_rate": (
            mean(item.unnecessary_repeat for item in followup_results)
            if followup_results
            else 0.0
        ),
        "judge_error_count": sum(bool(item.judge_error) for item in results),
    }


def save_rag_comparison_report(
    comparison: dict[str, dict[str, float]],
    output_path: Path,
    dataset_path: Path,
    *,
    judge_model: str,
) -> None:
    stateful = comparison["stateful"]
    rag_baseline = comparison["rag_baseline"]
    metrics = (
        "task_completion_rate",
        "unnecessary_repeat_rate",
    )
    lines = [
        "# 状态化诊断与普通 RAG 对照评测",
        "",
        f"- dataset: `{dataset_path}`",
        f"- judge_model: `{judge_model}`",
        "- Stateful diagnosis 与 Single-turn Hybrid RAG 共享同一知识库和 Hybrid RAG 基础设施。",
        "- Baseline 每轮只接收当前用户问题，不读取结构化诊断状态。",
        "- Judge 只负责将自然语言回复归类并识别重复建议，不参与生成回复。",
        f"- Judge errors: stateful={int(stateful.get('judge_error_count', 0))}, "
        f"rag_baseline={int(rag_baseline.get('judge_error_count', 0))}",
        "",
        "| metric | Stateful diagnosis | Single-turn Hybrid RAG | delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in metrics:
        stateful_value = float(stateful[metric])
        baseline_value = float(rag_baseline[metric])
        delta = (stateful_value - baseline_value) * 100
        lines.append(
            f"| {metric} | {stateful_value:.4f} | {baseline_value:.4f} | "
            f"{delta:+.2f} pp |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_rag_comparison_details(
    stateful_results: list[RAGBaselineResult],
    baseline_results: list[RAGBaselineResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "system",
        "session_id",
        "turn",
        "query",
        "response",
        "expected_action",
        "predicted_action",
        "expected_status",
        "predicted_status",
        "unnecessary_repeat",
        "judge_reason",
        "judge_error",
    )
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for system_name, results in (
            ("stateful", stateful_results),
            ("rag_baseline", baseline_results),
        ):
            for item in results:
                writer.writerow(
                    {
                        "system": system_name,
                        "session_id": item.session_id,
                        "turn": item.turn,
                        "query": item.query,
                        "response": item.response,
                        "expected_action": item.expected_action,
                        "predicted_action": item.predicted_action,
                        "expected_status": item.expected_status,
                        "predicted_status": item.predicted_status,
                        "unnecessary_repeat": item.unnecessary_repeat,
                        "judge_reason": item.judge_reason,
                        "judge_error": item.judge_error,
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare stateful troubleshooting with single-turn Hybrid RAG."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-conversations", type=int)
    parser.add_argument(
        "--judge-provider",
        default=str(rag_conf.get("diagnosis_observation_provider", "deepseek")),
    )
    parser.add_argument(
        "--judge-model",
        default=str(rag_conf.get("diagnosis_observation_model", "deepseek-v4-flash")),
    )
    parser.add_argument(
        "--judge-api-key-env",
        default=str(
            rag_conf.get("diagnosis_observation_api_key_env", "MAIN_DEEPSEEK_API_KEY")
        ),
    )
    parser.add_argument(
        "--judge-base-url",
        default=str(rag_conf.get("diagnosis_observation_base_url", "")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(get_abs_path(args.dataset))
    examples = select_conversations(
        load_dataset(dataset_path),
        max_conversations=args.max_conversations,
    )
    if not examples:
        raise ValueError(f"诊断对照评测集为空: {dataset_path}")

    from model.factory import build_chat_model

    judge_model = build_chat_model(
        args.judge_model,
        provider=args.judge_provider,
        api_key_env=args.judge_api_key_env,
        base_url=args.judge_base_url or None,
    )
    judge = LLMReplyJudge(judge_model)

    with TemporaryDirectory(prefix="diagnosis_rag_comparison_") as temp_dir:
        engine = build_evaluation_engine(
            Path(temp_dir) / "stateful.db",
            use_reranker=False,
        )
        stateful_results = evaluate_stateful_diagnosis(examples, engine, judge)
        baseline_results = evaluate_rag_baseline(
            examples,
            build_local_rag_responder(use_reranker=False),
            judge,
        )

    comparison = {
        "stateful": aggregate_rag_baseline_results(stateful_results),
        "rag_baseline": aggregate_rag_baseline_results(baseline_results),
    }
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(get_abs_path(args.output_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"diagnosis_rag_comparison_{timestamp}.md"
    details_path = output_dir / f"diagnosis_rag_comparison_{timestamp}.csv"
    save_rag_comparison_report(
        comparison,
        report_path,
        dataset_path,
        judge_model=args.judge_model,
    )
    save_rag_comparison_details(stateful_results, baseline_results, details_path)

    print(
        "状态化诊断与普通 RAG 对照评测完成: "
        f"{len({item.session_id for item in examples})} conversations, "
        f"{len(examples)} turns"
    )
    print(f"Markdown: {report_path}")
    print(f"CSV: {details_path}")
    for metric in (
        "task_completion_rate",
        "unnecessary_repeat_rate",
    ):
        print(
            f"- {metric}: stateful={comparison['stateful'][metric]:.4f}, "
            f"rag_baseline={comparison['rag_baseline'][metric]:.4f}"
        )


def parse_baseline_judgement(raw_output: str) -> BaselineJudgement:
    payload = _extract_json_object(raw_output)
    expected_fields = {"predicted_action", "unnecessary_repeat", "reason"}
    if set(payload) != expected_fields:
        raise ValueError(
            "Judge 输出字段必须严格为 predicted_action、unnecessary_repeat、reason"
        )
    action = _normalize_action(payload["predicted_action"])
    if action == "unknown" and str(payload["predicted_action"]).strip() != "unknown":
        raise ValueError(f"Judge 输出了未知动作: {payload['predicted_action']}")
    if not isinstance(payload["unnecessary_repeat"], bool):
        raise ValueError("unnecessary_repeat 必须是 JSON 布尔值")
    reason = str(payload["reason"] or "").strip()
    if not reason:
        raise ValueError("reason 不能为空")
    return BaselineJudgement(
        predicted_action=action,
        unnecessary_repeat=payload["unnecessary_repeat"],
        reason=reason,
    )


def _build_judge_prompt(
    *,
    query: str,
    response: str,
    history: tuple[ConversationReply, ...],
) -> str:
    if history:
        history_text = "\n".join(
            f"用户：{item.query}\n客服：{item.response}"
            for item in history
        )
    else:
        history_text = "（无历史对话）"
    return f"""
你是扫地机器人客服回复评测器，只做行为归类，不回答用户问题。

请将 current_response 归为以下一个动作：
- ask_symptom：追问具体故障现象或必要设备信息；
- ask_feedback：要求用户说明刚才操作后的结果；
- give_step：给出一个或多个排障操作建议；
- resolve：确认故障已经恢复并结束排查；
- escalate：明确停止自动排查并建议或执行人工升级；
- cancel：响应用户要求停止或暂停诊断；
- unknown：无法归入以上类别。

unnecessary_repeat 仅在以下情况为 true：客服重复要求执行历史中已经明确完成、失败或无效的操作，且没有提供新的排障动作。仅复述故障背景、进行必要结果确认，或给出依赖于前一步的新动作，不算重复。

严格输出 JSON，不要输出 Markdown 或额外字段：
{{"predicted_action":"...","unnecessary_repeat":false,"reason":"一句话依据"}}

history:
<history>
{history_text}
</history>

current_user:
<query>
{query}
</query>

current_response:
<response>
{response}
</response>
""".strip()


def _normalize_action(value: str) -> str:
    action = str(value or "").strip().lower()
    return action if action in {*_ACTION_TO_STATUS, "unknown"} else "unknown"


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    cleaned = str(raw_output or "").strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
    if fenced_match:
        cleaned = fenced_match.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("Judge 输出中没有 JSON 对象")
        cleaned = cleaned[start : end + 1]
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Judge 输出必须是 JSON 对象")
    return payload


if __name__ == "__main__":
    main()
