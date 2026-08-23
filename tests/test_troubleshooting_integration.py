from unittest import TestCase
from unittest.mock import patch

from agent.react_agent import ReactAgent
from agent.routing import RouteDecision, RouteName, RouteStatus
from agent.troubleshooting.models import (
    DiagnosisAction,
    DiagnosisState,
    DiagnosisTurn,
)


class FakeRouter:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    def route(self, query):
        self.calls.append(query)
        return self.decision


class FakeTroubleshootingEngine:
    def __init__(self, active=False, immediate_handoff=False):
        self.active = active
        self.immediate_handoff = immediate_handoff
        self.calls = []

    def has_active_session(self, session_id):
        return self.active

    def is_diagnostic_followup(self, query):
        return query == "还是不行"

    def requires_immediate_handoff(self, query):
        return self.immediate_handoff

    def handle(self, session_id, query, history=None):
        self.calls.append((session_id, query, history))
        state = DiagnosisState.start(session_id)
        return DiagnosisTurn(
            action=DiagnosisAction.GIVE_STEP,
            response="请执行下一步排查。",
            state=state,
        )


class TroubleshootingIntegrationTests(TestCase):
    @patch.object(ReactAgent, "_build_agent", return_value=object())
    @patch("agent.react_agent.write_hook_event")
    def test_troubleshooting_route_delegates_to_state_engine(
        self,
        _write_event,
        _build_agent,
    ):
        decision = RouteDecision(
            status=RouteStatus.DECISIVE,
            route=RouteName.TROUBLESHOOTING,
            reason_code="device_fault",
            tool_candidates=("rag_summarize",),
        )
        engine = FakeTroubleshootingEngine()
        router = FakeRouter(decision)
        agent = ReactAgent(
            router=router,
            troubleshooting_engine=engine,
        )

        chunks = list(
            agent.execute_stream("机器人无法回充", session_id="session-1")
        )

        self.assertEqual(chunks, ["请执行下一步排查。\n"])
        self.assertEqual(engine.calls, [("session-1", "机器人无法回充", None)])

    @patch.object(ReactAgent, "_build_agent", return_value=object())
    @patch("agent.react_agent.write_hook_event")
    def test_ambiguous_followup_resumes_active_diagnosis(
        self,
        _write_event,
        _build_agent,
    ):
        decision = RouteDecision(
            status=RouteStatus.NO_MATCH,
            route=None,
            reason_code="no_rule_match",
        )
        engine = FakeTroubleshootingEngine(active=True)
        agent = ReactAgent(
            router=FakeRouter(decision),
            troubleshooting_engine=engine,
        )

        chunks = list(agent.execute_stream("还是不行", session_id="session-1"))

        self.assertEqual(chunks, ["请执行下一步排查。\n"])
        self.assertEqual(engine.calls, [("session-1", "还是不行", None)])

    @patch.object(ReactAgent, "_build_agent", return_value=object())
    @patch("agent.react_agent.write_hook_event")
    def test_first_turn_safety_signal_preempts_normal_routing(
        self,
        _write_event,
        _build_agent,
    ):
        decision = RouteDecision(
            status=RouteStatus.NO_MATCH,
            route=None,
            reason_code="no_rule_match",
        )
        engine = FakeTroubleshootingEngine(immediate_handoff=True)
        router = FakeRouter(decision)
        agent = ReactAgent(
            router=router,
            troubleshooting_engine=engine,
        )

        list(agent.execute_stream("机器突然冒烟", session_id="session-1"))

        self.assertEqual(engine.calls, [("session-1", "机器突然冒烟", None)])
        self.assertEqual(router.calls, [])

    @patch.object(ReactAgent, "_build_agent", return_value=object())
    @patch("agent.react_agent.write_hook_event")
    def test_troubleshooting_receives_recent_conversation_history(
        self,
        _write_event,
        _build_agent,
    ):
        decision = RouteDecision(
            status=RouteStatus.DECISIVE,
            route=RouteName.TROUBLESHOOTING,
            reason_code="device_fault",
            tool_candidates=("rag_summarize",),
        )
        engine = FakeTroubleshootingEngine()
        agent = ReactAgent(
            router=FakeRouter(decision),
            troubleshooting_engine=engine,
        )
        history = [
            {"role": "user", "content": "我的设备型号 X200"},
            {"role": "assistant", "content": "请描述故障现象"},
        ]

        list(
            agent.execute_stream(
                "机器人无法回充",
                history=history,
                session_id="session-1",
            )
        )

        self.assertEqual(
            engine.calls,
            [("session-1", "机器人无法回充", history)],
        )
