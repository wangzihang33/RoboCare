from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
import uuid

from model.factory import build_chat_model, chat_model
from utils.prompt_loader import load_system_prompts
from utils.config_handler import agent_conf
from agent.tools.middleware import log_before_model, monitor_tool, report_prompt_switch
from agent.routing import (
    HybridRouter,
    RouteName,
    build_clarification_response,
    route_tool_names,
)
from agent.route_classifier import SmallLLMRouteClassifier
from agent.hooks.recorder import write_hook_event
from agent.troubleshooting.engine import TroubleshootingEngine
from agent.troubleshooting.knowledge import (
    build_observation_model,
    LLMKnowledgeResolver,
    LLMObservationParser,
    LocalDiagnosticRetriever,
)
from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_weather,
    rag_summarize,
    web_search,
)



_TOOL_REGISTRY = {
    "rag_summarize": rag_summarize,
    "get_weather": get_weather,
    "fetch_external_data": fetch_external_data,
    "fill_context_for_report": fill_context_for_report,
    "web_search": web_search,
}


class ReactAgent:
    def __init__(
        self,
        router: HybridRouter | None = None,
        troubleshooting_engine: TroubleshootingEngine | None = None,
    ):
        self.router = router or self._build_default_router()
        self.troubleshooting_engine = (
            troubleshooting_engine
            or TroubleshootingEngine(
                observation_fallback=LLMObservationParser(
                    model=build_observation_model()
                ),
                knowledge_retriever=LocalDiagnosticRetriever(),
                knowledge_resolver=LLMKnowledgeResolver(),
            )
        )
        self._default_session_id = f"sess_{uuid.uuid4().hex[:12]}"
        self._route_agents = {}
        # Keep the full-tool agent as a compatibility handle; requests use a
        # route-specific agent selected in _get_route_agent.
        self.agent = self._build_agent(tuple(_TOOL_REGISTRY))

    @staticmethod
    def _build_agent(tool_names: tuple[str, ...]):
        return create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(tool_names),
            tools=[_TOOL_REGISTRY[name] for name in tool_names],
            middleware=[log_before_model, monitor_tool, report_prompt_switch],
        )

    @staticmethod
    def _build_default_router() -> HybridRouter:
        if not agent_conf.get("router_llm_enabled", False):
            return HybridRouter()

        model_name = str(agent_conf.get("router_model_name") or "").strip()
        if not model_name:
            raise ValueError(
                "ROUTER_LLM_ENABLED=true 时必须配置 ROUTER_MODEL_NAME"
            )
        return HybridRouter(
            llm_classifier=SmallLLMRouteClassifier(
                build_chat_model(
                    model_name,
                    provider=str(agent_conf.get("router_provider", "deepseek")),
                    api_key_env=str(
                        agent_conf.get("router_api_key_env", "DEEPSEEK_API_KEY")
                    ),
                    base_url=str(agent_conf.get("router_base_url") or "") or None,
                )
            )
        )

    def execute_stream(
        self,
        query: str,
        history: list[dict] | None = None,
        session_id: str | None = None,
    ):
        resolved_session_id = session_id or self._default_session_id
        active_diagnosis = self.troubleshooting_engine.has_active_session(
            resolved_session_id
        )
        if self.troubleshooting_engine.requires_immediate_handoff(query) or (
            active_diagnosis
            and self.troubleshooting_engine.is_diagnostic_followup(query)
        ):
            yield self._run_troubleshooting(resolved_session_id, query, history)
            return

        route_decision = self.router.route(query)
        write_hook_event(
            {
                "stage": "route_decision",
                "route": route_decision.route.value if route_decision.route else None,
                "route_status": route_decision.status.value,
                "route_source": route_decision.source,
                "reason_code": route_decision.reason_code,
                "requires_clarification": route_decision.requires_clarification,
                "tool_candidates": list(route_decision.tool_candidates),
                "missing_slots": list(route_decision.missing_slots),
                "evidence_spans": list(route_decision.evidence_spans),
                "query_length": len(query),
            }
        )
        should_continue_diagnosis = active_diagnosis and (
            route_decision.route is None
        )
        if (
            route_decision.route is RouteName.TROUBLESHOOTING
            or should_continue_diagnosis
        ):
            yield self._run_troubleshooting(resolved_session_id, query, history)
            return

        clarification_response = build_clarification_response(route_decision)
        if clarification_response:
            yield clarification_response + "\n"
            return

        input_dict = {
            "messages": self._build_messages(query, history),
        }
        tool_names = route_tool_names(route_decision)
        agent = self._get_route_agent(route_decision)
        res = agent.stream(
            input_dict,
            stream_mode="values",
            context={
                "report": False,
                "route_decision": route_decision,
                "tool_names": tool_names,
                "session_id": resolved_session_id,
            },
        )
        for chunk in res:
            latest_message = chunk["messages"][-1]
            # ToolMessage contains internal failures and audit text. Keep it in
            # the agent context, but never stream it as user-visible content.
            if isinstance(latest_message, ToolMessage) and latest_message.status == "error":
                yield self._tool_failure_response(latest_message.name)
                return
            if isinstance(latest_message, AIMessage) and latest_message.content:
                yield latest_message.content.strip() + "\n"

    @staticmethod
    def _tool_failure_response(tool_name: str | None) -> str:
        if tool_name == "rag_summarize":
            return (
                "本地知识库暂时不可用，暂时无法完成这次资料检索。"
                "请稍后重试，或换一种方式描述你的问题。"
            )
        if tool_name in {"web_search", "get_weather", "fetch_external_data"}:
            return (
                "外部数据服务暂时不可用，暂时无法完成这次查询。"
                "请稍后重试，或补充更具体的信息。"
            )
        return "当前服务暂时不可用，请稍后重试。"

    def _run_troubleshooting(
        self,
        session_id: str,
        query: str,
        history: list[dict] | None = None,
    ) -> str:
        turn = self.troubleshooting_engine.handle(session_id, query, history=history)
        write_hook_event(
            {
                "stage": "diagnosis_turn",
                "session_id": session_id,
                "case_id": turn.state.case_id,
                "diagnosis_action": turn.action.value,
                "diagnosis_status": turn.state.status.value,
                "symptom_code": turn.state.symptom_code or None,
                "error_code": turn.state.error_code or None,
                "risk_flags": list(turn.state.risk_flags),
                "escalation_reason": turn.state.escalation_reason or None,
                "evidence_ids": list(turn.state.evidence_ids),
                "turn_count": turn.state.turn_count,
            }
        )
        return turn.response + "\n"

    def _get_route_agent(self, decision):
        tool_names = route_tool_names(decision)
        if tool_names == tuple(_TOOL_REGISTRY):
            return self.agent
        if tool_names not in self._route_agents:
            self._route_agents[tool_names] = self._build_agent(tool_names)
        return self._route_agents[tool_names]

    @staticmethod
    def _build_messages(query: str, history: list[dict] | None = None) -> list[dict]:
        max_history_messages = int(agent_conf.get("max_history_messages", 8))
        messages = []

        for message in (history or [])[-max_history_messages:]:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": query})
        return messages


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("我这个地区适合咱们智扫通品牌的扫地机器人吗？有什么优势？有什么劣势？"):
        print(chunk, end="", flush=True)
