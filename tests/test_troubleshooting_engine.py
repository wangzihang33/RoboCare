import pytest

from agent.troubleshooting.models import DiagnosisAction, DiagnosisStatus
from agent.troubleshooting.store import DiagnosisStore
from agent.troubleshooting.engine import TroubleshootingEngine


_SEMANTIC_CASES = (
    ("无法回充", "cannot_recharge", "TROUBLE-003"),
    ("回冲失败", "cannot_recharge", "TROUBLE-003"),
    ("找不到底座", "cannot_recharge", "TROUBLE-003"),
    ("回座总失败", "cannot_recharge", "TROUBLE-003"),
    ("回不到充电座", "cannot_recharge", "TROUBLE-003"),
    ("异响", "abnormal_noise", "TROUBLE-006"),
    ("刺耳", "abnormal_noise", "TROUBLE-006"),
    ("吱吱响", "abnormal_noise", "TROUBLE-006"),
    ("吱吱", "abnormal_noise", "TROUBLE-006"),
    ("咔咔响", "abnormal_noise", "TROUBLE-006"),
    ("声音忽高忽低", "abnormal_noise", "TROUBLE-006"),
    ("原地打转", "spinning_in_place", "TROUBLE-009"),
    ("来回绕圈", "spinning_in_place", "TROUBLE-009"),
    ("打转", "spinning_in_place", "TROUBLE-009"),
    ("水箱", "water_leak", "TROUBLE-005"),
    ("漏水", "water_leak", "TROUBLE-005"),
    ("滴水", "water_leak", "TROUBLE-005"),
    ("往下滴", "water_leak", "TROUBLE-005"),
    ("不出水", "no_water", "TROUBLE-008"),
    ("一直没有水", "no_water", "TROUBLE-008"),
    ("拖地一直没有水", "no_water", "TROUBLE-008"),
    ("没有水痕", "no_water", "TROUBLE-008"),
    ("水痕", "no_water", "TROUBLE-008"),
    ("拖布", "no_water", "TROUBLE-008"),
    ("APP", "wifi_offline", "TROUBLE-001"),
    ("离线", "wifi_offline", "TROUBLE-001"),
    ("断开", "wifi_offline", "TROUBLE-001"),
    ("吸力", "low_suction", "TROUBLE-002"),
    ("吸不动", "low_suction", "TROUBLE-002"),
    ("清洁效果", "low_suction", "TROUBLE-002"),
    ("使不上力", "low_suction", "TROUBLE-002"),
)


def _semantic_retrieve(query):
    for phrase, _code, evidence_id in _SEMANTIC_CASES:
        if phrase in query:
            return [{"evidence_id": evidence_id, "content": f"{evidence_id} 相关故障资料"}]
    return []


def _semantic_resolve(_query, evidence, _allowed_symptoms):
    evidence_id = evidence[0].evidence_id
    return {
        "symptom_code": next(
            code for _phrase, code, item_id in _SEMANTIC_CASES
            if item_id == evidence_id
        ),
        "confidence": 0.95,
        "evidence_id": evidence_id,
    }


def _semantic_engine(db_path, **kwargs):
    return TroubleshootingEngine(
        DiagnosisStore(db_path),
        knowledge_retriever=kwargs.pop("knowledge_retriever", _semantic_retrieve),
        knowledge_resolver=kwargs.pop("knowledge_resolver", _semantic_resolve),
        **kwargs,
    )


@pytest.fixture
def engine(tmp_path):
    return _semantic_engine(tmp_path / "diagnosis.db")


def test_known_symptom_starts_first_diagnostic_step(engine):
    turn = engine.handle("session-1", "型号 X1 的机器人无法回充")

    assert turn.action is DiagnosisAction.GIVE_STEP
    assert turn.state.status is DiagnosisStatus.WAITING_FEEDBACK
    assert turn.state.device_model == "X1"
    assert turn.state.symptom_code == "cannot_recharge"
    assert turn.state.current_step_id
    assert turn.state.evidence_ids == ["TROUBLE-003"]


def test_all_troubleshooting_playbooks_use_action_groups(tmp_path):
    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "all-action-groups.db")
    )

    assert local_engine.playbooks
    assert all(
        isinstance(playbook.get("action_groups"), list)
        for playbook in local_engine.playbooks.values()
    )


def test_known_error_code_bypasses_semantic_fault_resolution(tmp_path):
    resolver_calls = []

    def retrieve(_query):
        return []

    def resolve(*_args):
        resolver_calls.append(True)
        raise AssertionError("known error codes must not call semantic resolution")

    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "error-code.db"),
        knowledge_retriever=retrieve,
        knowledge_resolver=resolve,
    )

    turn = local_engine.handle("error-code-session", "设备报 E01")

    assert turn.action is DiagnosisAction.GIVE_STEP
    assert turn.state.symptom_code == "main_brush_jam"
    assert resolver_calls == []


def test_e01_action_group_success_resolves(tmp_path):
    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "group-success.db")
    )

    first = local_engine.handle("e01-success", "设备报 E01")
    turn = local_engine.handle(
        "e01-success",
        "清理后主刷恢复转动，E01 消失了",
    )

    assert first.state.current_step_id == "brush-clean"
    assert "清理主刷两端" in first.response
    assert turn.action is DiagnosisAction.RESOLVE


def test_e01_action_group_failure_advances_to_next_group(tmp_path):
    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "group-failure.db")
    )

    local_engine.handle("e01-failure", "设备报 E01")
    turn = local_engine.handle(
        "e01-failure",
        "处理后还是卡住，仍然报 E01",
    )

    assert turn.action is DiagnosisAction.GIVE_STEP
    assert turn.state.current_step_id == "inspect-installation"
    assert turn.state.attempts[0].step_id == "brush-clean"
    assert turn.state.attempts[0].outcome == "failed"


def test_e01_action_groups_escalate_after_second_failure(tmp_path):
    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "group-escalate.db")
    )

    local_engine.handle("e01-escalate", "设备报 E01")
    local_engine.handle("e01-escalate", "清理后仍然卡住")
    turn = local_engine.handle(
        "e01-escalate",
        "检查安装后还是不能运行",
    )

    assert turn.action is DiagnosisAction.ESCALATE
    assert turn.state.escalation_reason == "action_groups_exhausted"


def test_e01_action_group_unknown_feedback_does_not_advance(tmp_path):
    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "group-unknown.db")
    )

    local_engine.handle("e01-unknown", "设备报 E01")
    turn = local_engine.handle("e01-unknown", "我按照要求弄了一下")

    assert turn.action is DiagnosisAction.ASK_FEEDBACK
    assert turn.state.current_step_id == "brush-clean"
    assert turn.state.attempts[0].outcome == "pending"


def test_unknown_error_code_asks_for_symptom_without_fault_resolution(tmp_path):
    resolver_calls = []

    def retrieve(_query):
        return [
            {
                "evidence_id": "KB-BRUSH",
                "content": "主刷被异物缠绕时可能停止转动",
                "symptom_code": "main_brush_jam",
            }
        ]

    def resolve(*_args):
        resolver_calls.append(True)
        return {"symptom_code": "main_brush_jam", "confidence": 0.95}

    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "unknown-error-code.db"),
        knowledge_retriever=retrieve,
        knowledge_resolver=resolve,
    )

    turn = local_engine.handle("unknown-code-session", "设备报 E99")

    assert turn.action is DiagnosisAction.ASK_SYMPTOM
    assert turn.state.error_code == "E99"
    assert turn.state.symptom_code == ""
    assert "暂未找到错误码 E99" in turn.response
    assert resolver_calls == []


def test_unknown_error_code_without_description_skips_retrieval(tmp_path):
    retriever_calls = []
    resolver_calls = []

    def retrieve(query):
        retriever_calls.append(query)
        return [
            {
                "evidence_id": "KB-BRUSH",
                "content": "主刷被异物缠绕时可能停止转动",
                "symptom_code": "main_brush_jam",
            }
        ]

    def resolve(*_args):
        resolver_calls.append(True)
        return {"symptom_code": "main_brush_jam", "confidence": 0.95}

    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "unknown-error-no-retrieval.db"),
        knowledge_retriever=retrieve,
        knowledge_resolver=resolve,
    )

    turn = local_engine.handle("unknown-code-no-retrieval", "设备报 E99")

    assert turn.action is DiagnosisAction.ASK_SYMPTOM
    assert turn.state.error_code == "E99"
    assert retriever_calls == []
    assert resolver_calls == []


def test_unknown_error_code_can_resolve_after_fault_description(tmp_path):
    resolver_calls = []

    def retrieve(query):
        if "主刷" not in query:
            return []
        return [{"evidence_id": "KB-BRUSH", "content": "主刷被异物缠绕时可能停止转动"}]

    def resolve(query, evidence, allowed_symptoms):
        resolver_calls.append(query)
        assert evidence[0].evidence_id == "KB-BRUSH"
        assert "main_brush_jam" in allowed_symptoms
        return {
            "symptom_code": "main_brush_jam",
            "confidence": 0.95,
            "evidence_id": "KB-BRUSH",
        }

    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "unknown-error-followup.db"),
        knowledge_retriever=retrieve,
        knowledge_resolver=resolve,
    )

    first = local_engine.handle("unknown-code-followup", "设备报 E99")
    second = local_engine.handle(
        "unknown-code-followup",
        "主刷被毛发缠住后完全不转",
    )

    assert first.action is DiagnosisAction.ASK_SYMPTOM
    assert second.action is DiagnosisAction.GIVE_STEP
    assert second.state.error_code == "E99"
    assert second.state.symptom_code == "main_brush_jam"
    assert resolver_calls == ["主刷被毛发缠住后完全不转"]


def test_without_error_code_uses_semantic_resolution_instead_of_signal_matching(tmp_path):
    calls = []

    def retrieve(_query):
        return [{"evidence_id": "KB-BRUSH", "content": "主刷故障证据"}]

    def resolve(query, evidence, allowed_symptoms):
        calls.append((query, evidence[0].evidence_id))
        assert "main_brush_jam" in allowed_symptoms
        return {"symptom_code": "main_brush_jam", "confidence": 0.93}

    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "semantic-only.db"),
        knowledge_retriever=retrieve,
        knowledge_resolver=resolve,
    )

    turn = local_engine.handle(
        "semantic-only-session",
        "清扫时滚刷被毛发缠住，后来完全不转了",
    )

    assert turn.action is DiagnosisAction.GIVE_STEP
    assert turn.state.symptom_code == "main_brush_jam"
    assert calls == [("清扫时滚刷被毛发缠住，后来完全不转了", "KB-BRUSH")]


def test_without_error_code_does_not_classify_from_removed_signal_list(tmp_path):
    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "no-signal-classification.db")
    )

    turn = local_engine.handle("no-signal-session", "主刷卡住了")

    assert turn.action is DiagnosisAction.ASK_SYMPTOM
    assert turn.state.symptom_code == ""


def test_failed_step_advances_without_repeating(engine):
    first = engine.handle("session-1", "机器人无法回充")
    second = engine.handle("session-1", "擦过了，还是不行")

    assert second.action is DiagnosisAction.GIVE_STEP
    assert second.state.current_step_id != first.state.current_step_id
    assert second.state.attempts[0].outcome == "failed"


def test_positive_feedback_resolves_and_closes_case(engine):
    engine.handle("session-1", "机器人无法回充")
    turn = engine.handle("session-1", "现在已经恢复正常了")

    assert turn.action is DiagnosisAction.RESOLVE
    assert turn.state.status is DiagnosisStatus.RESOLVED
    assert turn.state.attempts[-1].outcome == "success"
    assert not engine.has_active_session("session-1")


def test_unknown_symptom_is_collected_before_diagnosis(engine):
    first = engine.handle("session-1", "机器人有点问题")
    second = engine.handle("session-1", "它运行时一直原地打转")

    assert first.action is DiagnosisAction.ASK_SYMPTOM
    assert first.state.status is DiagnosisStatus.COLLECTING
    assert second.action is DiagnosisAction.GIVE_STEP
    assert second.state.symptom_code == "spinning_in_place"


def test_new_case_bootstraps_device_model_from_recent_user_history(engine):
    turn = engine.handle(
        "session-1",
        "现在一直无法回充",
        history=[{"role": "user", "content": "我的设备型号 X200"}],
    )

    assert turn.state.device_model == "X200"
    assert turn.state.symptom_code == "cannot_recharge"


def test_assistant_history_is_not_extracted_as_user_fact(engine):
    turn = engine.handle(
        "session-1",
        "机器人无法回充",
        history=[{"role": "assistant", "content": "设备型号 FAKE-1"}],
    )

    assert turn.state.device_model == ""


def test_high_risk_signal_immediately_creates_handoff(engine):
    turn = engine.handle("session-1", "机器突然冒烟，还有烧焦味")

    assert turn.action is DiagnosisAction.ESCALATE
    assert turn.state.status is DiagnosisStatus.ESCALATED
    assert turn.state.escalation_reason == "high_risk_signal"
    assert set(turn.state.risk_flags) == {"冒烟", "烧焦味"}
    assert turn.handoff is not None
    assert turn.handoff.reason == "high_risk_signal"
    assert not engine.has_active_session("session-1")


def test_explicit_human_request_creates_handoff(engine):
    turn = engine.handle("session-1", "这个问题请帮我转人工客服")

    assert turn.action is DiagnosisAction.ESCALATE
    assert turn.state.escalation_reason == "user_requested_handoff"
    assert turn.handoff is not None


def test_exhausted_playbook_escalates_with_attempt_summary(engine):
    turn = engine.handle("session-1", "机器人工作时持续异响")
    while turn.action is DiagnosisAction.GIVE_STEP:
        turn = engine.handle("session-1", "做过了，仍然没有解决")

    assert turn.action is DiagnosisAction.ESCALATE
    assert turn.state.escalation_reason == "action_groups_exhausted"
    assert turn.handoff is not None
    assert len(turn.handoff.attempted_steps) >= 2
    assert all(step["outcome"] == "failed" for step in turn.handoff.attempted_steps)


def test_identifies_diagnostic_feedback_without_swallowing_new_tasks(engine):
    assert engine.is_diagnostic_followup("还是不行")
    assert engine.is_diagnostic_followup("请帮我转人工客服")
    assert not engine.is_diagnostic_followup("北京今天天气怎么样")


def test_immediate_handoff_signals_can_preempt_normal_routing(engine):
    assert engine.requires_immediate_handoff("机器突然冒烟")
    assert engine.requires_immediate_handoff("请直接转人工客服")
    assert not engine.requires_immediate_handoff("机器人无法回充")


def test_labelled_knowledge_still_requires_semantic_resolution(tmp_path):
    resolver_calls = []

    def retrieve(_query):
        return [
            {
                "evidence_id": "KB-E01-001",
                "content": "E01 主刷卡住时，关闭设备并清理主刷两端异物。",
                "metadata": {"card_id": "TROUBLE-E01"},
                "symptom_code": "main_brush_jam",
            }
        ]

    def resolve(query, evidence, allowed_symptoms):
        resolver_calls.append((query, evidence[0].symptom_code))
        assert "main_brush_jam" in allowed_symptoms
        return {
            "symptom_code": "main_brush_jam",
            "confidence": 0.94,
            "evidence_id": "KB-E01-001",
            "evidence_span": "主刷卡住",
        }

    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "knowledge.db"),
        knowledge_retriever=retrieve,
        knowledge_resolver=resolve,
    )

    turn = local_engine.handle("knowledge-session", "滚刷被毛发缠住并停止转动")

    assert turn.action is DiagnosisAction.GIVE_STEP
    assert turn.state.symptom_code == "main_brush_jam"
    assert "KB-E01-001" in turn.state.evidence_ids
    assert resolver_calls == [
        ("滚刷被毛发缠住并停止转动", "main_brush_jam")
    ]


def test_unknown_retrieved_fault_does_not_invent_a_diagnostic_step(tmp_path):
    def retrieve(_query):
        return [
            {
                "evidence_id": "KB-UNKNOWN-001",
                "content": "资料提到一种未归类的异常现象，但没有经过审核的处理步骤。",
                "metadata": {"card_id": "UNKNOWN"},
            }
        ]

    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "unknown-knowledge.db"),
        knowledge_retriever=retrieve,
    )

    turn = local_engine.handle("unknown-session", "设备出现资料里提到的陌生异常")

    assert turn.action is DiagnosisAction.ASK_SYMPTOM
    assert turn.state.status is DiagnosisStatus.COLLECTING
    assert turn.state.current_step_id == ""
    assert "KB-UNKNOWN-001" in turn.state.evidence_ids


def test_retrieved_document_text_cannot_be_used_to_infer_an_unrelated_fault(tmp_path):
    def retrieve(_query):
        return [
            {
                "evidence_id": "KB-MIXED-001",
                "content": "资料同时提到原地转圈、主刷和其他设备现象，但没有确认当前故障类型。",
                "metadata": {"card_id": "MIXED"},
            }
        ]

    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "mixed-knowledge.db"),
        knowledge_retriever=retrieve,
    )

    turn = local_engine.handle("mixed-session", "滚刷出现一个未归类的新异常")

    assert turn.action is DiagnosisAction.ASK_SYMPTOM
    assert turn.state.symptom_code == ""


def test_semantic_knowledge_resolver_selects_only_an_allowed_playbook(tmp_path):
    def retrieve(_query):
        return [
            {
                "evidence_id": "KB-BRUSH-001",
                "content": "主刷被毛发缠绕后可能停止转动，建议断电清理。",
                "metadata": {"card_id": "BRUSH"},
            }
        ]

    def resolve(_query, _evidence, allowed_symptoms):
        assert "main_brush_jam" in allowed_symptoms
        return {"symptom_code": "main_brush_jam", "confidence": 0.94}

    local_engine = TroubleshootingEngine(
        DiagnosisStore(tmp_path / "resolver.db"),
        knowledge_retriever=retrieve,
        knowledge_resolver=resolve,
    )

    turn = local_engine.handle("resolver-session", "滚刷被毛发缠住后完全不转")

    assert turn.action is DiagnosisAction.GIVE_STEP
    assert turn.state.symptom_code == "main_brush_jam"
    assert "KB-BRUSH-001" in turn.state.evidence_ids


@pytest.mark.parametrize(
    ("query", "expected_symptom"),
    [
        ("它老是回不到充电座", "cannot_recharge"),
        ("清扫时一直咔咔响", "abnormal_noise"),
        ("拖布一直是干的", "no_water"),
        ("机器人一直回冲失败", "cannot_recharge"),
        ("水箱停机后还一直滴", "water_leak"),
        ("感觉机器吸不动灰了", "low_suction"),
        ("这台机器总找不到底座", "cannot_recharge"),
        ("工作声忽然变得很刺耳", "abnormal_noise"),
        ("机器走几步就开始打转", "spinning_in_place"),
        ("水箱边缘不断往外渗水", "water_leak"),
        ("拖地时拖布始终没有湿", "no_water"),
        ("手机里看设备状态一直断开", "wifi_offline"),
        ("最近吸尘像是使不上力", "low_suction"),
        ("停机后水还在一点点往下滴", "water_leak"),
        ("地面没有留下任何水痕", "no_water"),
        ("清扫过程中会有吱吱的声音", "abnormal_noise"),
        ("具体表现是走路时声音忽高忽低", "abnormal_noise"),
    ],
)
def test_understands_colloquial_fault_descriptions(tmp_path, query, expected_symptom):
    local_engine = _semantic_engine(tmp_path / f"{expected_symptom}.db")

    turn = local_engine.handle(expected_symptom, query)

    assert turn.action is DiagnosisAction.GIVE_STEP
    assert turn.state.symptom_code == expected_symptom


def test_understands_colloquial_failed_and_resolved_feedback(engine):
    engine.handle("session-1", "APP 一直显示离线")
    failed = engine.handle("session-1", "照做了但问题还在")
    resolved = engine.handle("session-1", "重新绑定以后在线了")

    assert failed.action is DiagnosisAction.GIVE_STEP
    assert resolved.action is DiagnosisAction.RESOLVE


def test_step_contract_resolves_natural_success_feedback(engine):
    engine.handle("session-1", "机器人工作时持续异响")

    turn = engine.handle("session-1", "清理完以后声音完全消失了")

    assert turn.action is DiagnosisAction.RESOLVE
    assert turn.state.attempts[-1].outcome == "success"
    assert turn.state.attempts[-1].evidence_span == "声音完全消失"
    assert turn.state.attempts[-1].observation_source == "rule"


def test_step_contract_advances_on_natural_failure_feedback(engine):
    first = engine.handle("session-1", "机器人工作时持续异响")

    turn = engine.handle("session-1", "清理主刷以后还是有声音")

    assert turn.action is DiagnosisAction.GIVE_STEP
    assert first.state.current_step_id != turn.state.current_step_id
    assert turn.state.attempts[0].outcome == "failed"
    assert turn.state.attempts[0].evidence_span == "还是有声音"


def test_natural_cancel_and_handoff_signals_remain_policy_events(engine):
    cancel = engine.handle("cancel", "机器人原地转圈")
    cancel_turn = engine.handle("cancel", "先暂停后面再查")
    human = engine.handle("human", "麻烦安排售后人员处理")
    risk = engine.handle("risk", "充电时发现电池外壳鼓起来了")

    assert cancel_turn.action is DiagnosisAction.CANCEL
    assert human.action is DiagnosisAction.ESCALATE
    assert human.state.escalation_reason == "user_requested_handoff"
    assert risk.action is DiagnosisAction.ESCALATE
    assert risk.state.escalation_reason == "high_risk_signal"


def test_engine_can_inject_observation_fallback_without_creating_a_router(tmp_path):
    from agent.troubleshooting.observation import DiagnosisObservation, ObservationKind

    def fallback(message, step):
        if "能用了" in message:
            return DiagnosisObservation(
                kind=ObservationKind.SUCCESS,
                evidence_span="能用了",
                reason_code="step_success",
                source="small_model",
            )
        return DiagnosisObservation(kind=ObservationKind.UNKNOWN)

    local_engine = _semantic_engine(
        tmp_path / "fallback.db",
        observation_fallback=fallback,
    )
    local_engine.handle("session-1", "机器人工作时持续异响")

    turn = local_engine.handle("session-1", "清理后能用了")

    assert turn.action is DiagnosisAction.RESOLVE
    assert turn.state.attempts[-1].observation_source == "small_model"


def test_observation_fallback_can_receive_the_current_step_context(tmp_path):
    from agent.troubleshooting.observation import DiagnosisObservation, ObservationKind

    def fallback(message, step):
        assert step["id"] == "noise-clean-brushes"
        return DiagnosisObservation(
            kind=ObservationKind.SUCCESS,
            evidence_span=message,
            reason_code="step_success",
            source="small_model",
        )

    local_engine = _semantic_engine(
        tmp_path / "contextual-fallback.db",
        observation_fallback=fallback,
    )
    local_engine.handle("contextual", "机器人工作时持续异响")

    turn = local_engine.handle("contextual", "清理后声音已经正常了")

    assert turn.action is DiagnosisAction.RESOLVE
    assert turn.state.attempts[-1].observation_source == "small_model"


def test_understands_implicit_risk_and_human_request(engine):
    risk = engine.handle("risk", "机器有股糊味而且明显发烫")
    human = engine.handle("human", "能不能找个真人处理")

    assert risk.action is DiagnosisAction.ESCALATE
    assert risk.state.escalation_reason == "high_risk_signal"
    assert human.action is DiagnosisAction.ESCALATE
    assert human.state.escalation_reason == "user_requested_handoff"


@pytest.mark.parametrize(
    ("query", "expected_action", "reason"),
    [
        ("电池仓看起来已经鼓胀变形", DiagnosisAction.ESCALATE, "high_risk_signal"),
        ("设备进水了却还连着电源", DiagnosisAction.ESCALATE, "high_risk_signal"),
        ("我想直接和真人沟通", DiagnosisAction.ESCALATE, "user_requested_handoff"),
        ("能不能让客服人员接手这个问题", DiagnosisAction.ESCALATE, "user_requested_handoff"),
    ],
)
def test_natural_policy_requests_preempt_diagnosis(engine, query, expected_action, reason):
    turn = engine.handle(query, query)

    assert turn.action is expected_action
    assert turn.state.escalation_reason == reason


def test_natural_cancel_request_closes_active_diagnosis(engine):
    engine.handle("cancel-natural", "机器人无法回充")

    turn = engine.handle("cancel-natural", "先把诊断关掉吧")

    assert turn.action is DiagnosisAction.CANCEL
    assert turn.state.status is DiagnosisStatus.CANCELLED


@pytest.mark.parametrize(
    ("initial", "feedback", "expected_action"),
    [
        ("工作声忽然变得很刺耳", "把刷子清理一遍后声音听不到了", DiagnosisAction.RESOLVE),
        ("这台机器总找不到底座", "后来它自己找到充电位置了", DiagnosisAction.RESOLVE),
    ],
)
def test_natural_success_feedback_resolves_current_step(
    tmp_path, initial, feedback, expected_action
):
    local_engine = _semantic_engine(tmp_path / "natural-feedback.db")
    local_engine.handle("natural-feedback", initial)

    turn = local_engine.handle("natural-feedback", feedback)

    assert turn.action is expected_action
    assert turn.state.status is DiagnosisStatus.RESOLVED


def test_natural_suction_failure_then_success_advances_and_resolves(tmp_path):
    local_engine = _semantic_engine(tmp_path / "natural-suction-feedback.db")
    local_engine.handle("natural-suction", "最近清洁效果比之前差了不少")

    failed = local_engine.handle(
        "natural-suction", "滤网和尘盒都处理过了吸力还是弱"
    )
    resolved = local_engine.handle("natural-suction", "把吸口清完后现在好多了")

    assert failed.action is DiagnosisAction.GIVE_STEP
    assert resolved.action is DiagnosisAction.RESOLVE


def test_advanced_failed_action_escalates_without_repeating_playbook_step(tmp_path):
    local_engine = _semantic_engine(tmp_path / "advanced-failure.db")
    local_engine.handle("advanced-failure", "它在地面上来回绕圈")
    local_engine.handle("advanced-failure", "擦完底部以后还是绕圈")

    turn = local_engine.handle("advanced-failure", "换到平地也没有变化")

    assert turn.action is DiagnosisAction.ESCALATE
    assert turn.state.escalation_reason == "advanced_step_failed"


def test_ambiguous_symptom_can_resolve_after_followup_description(tmp_path):
    local_engine = _semantic_engine(tmp_path / "ambiguous-followup.db")
    local_engine.handle("ambiguous-followup", "今天设备状态有些反常")
    local_engine.handle("ambiguous-followup", "具体表现是走路时声音忽高忽低")

    turn = local_engine.handle("ambiguous-followup", "重新清洁后运行稳定了")

    assert turn.action is DiagnosisAction.RESOLVE
    assert turn.state.status is DiagnosisStatus.RESOLVED


@pytest.mark.parametrize(
    ("initial", "feedback"),
    [
        ("这台机器总找不到底座", "把周围整理过了还是没能回去"),
        ("机器回座总失败", "清障之后故障依旧"),
        ("清扫过程中会有吱吱的声音", "刷子弄干净了但噪声未消失"),
        ("它拖地一直没有水", "重新装水箱不见好转"),
        ("它在地面上来回绕圈", "擦完底部以后还是绕圈"),
    ],
)
def test_natural_failed_feedback_advances_current_step(tmp_path, initial, feedback):
    local_engine = _semantic_engine(tmp_path / "natural-failure-feedback.db")
    first = local_engine.handle("natural-failure", initial)

    turn = local_engine.handle("natural-failure", feedback)

    assert first.action is DiagnosisAction.GIVE_STEP
    assert turn.action is DiagnosisAction.GIVE_STEP
    assert turn.state.attempts[0].outcome == "failed"


def test_three_failed_recharge_steps_escalate(tmp_path):
    local_engine = _semantic_engine(tmp_path / "recharge-exhaustion.db")
    local_engine.handle("recharge-exhaustion", "机器回座总失败")
    local_engine.handle("recharge-exhaustion", "清障之后故障依旧")
    local_engine.handle("recharge-exhaustion", "感应器擦过了也没变化")

    turn = local_engine.handle("recharge-exhaustion", "确认电源正常仍然回不去")

    assert turn.action is DiagnosisAction.ESCALATE
    assert turn.state.escalation_reason == "action_groups_exhausted"
