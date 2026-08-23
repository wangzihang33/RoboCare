from agent.troubleshooting.models import (
    DiagnosisState,
    DiagnosisStatus,
    HandoffTicket,
    StepAttempt,
)
from agent.troubleshooting.store import DiagnosisStore


def test_store_round_trips_active_diagnosis(tmp_path):
    store = DiagnosisStore(tmp_path / "diagnosis.db")
    state = DiagnosisState.start("session-1")
    state.symptom_code = "cannot_recharge"
    state.symptom_text = "机器人无法回充"
    state.status = DiagnosisStatus.WAITING_FEEDBACK
    state.current_step_id = "recharge-1"
    state.attempts.append(
        StepAttempt(
            step_id="recharge-1",
            instruction="擦拭充电触点",
            outcome="pending",
        )
    )

    store.save_state(state)
    loaded = store.get_active_state("session-1")

    assert loaded is not None
    assert loaded.case_id == state.case_id
    assert loaded.symptom_code == "cannot_recharge"
    assert loaded.current_step_id == "recharge-1"
    assert loaded.attempts[0].instruction == "擦拭充电触点"


def test_terminal_case_is_retained_but_not_returned_as_active(tmp_path):
    store = DiagnosisStore(tmp_path / "diagnosis.db")
    state = DiagnosisState.start("session-1")
    state.status = DiagnosisStatus.RESOLVED
    store.save_state(state)

    assert store.get_active_state("session-1") is None
    assert store.get_latest_state("session-1").status is DiagnosisStatus.RESOLVED


def test_store_round_trips_handoff_ticket(tmp_path):
    store = DiagnosisStore(tmp_path / "diagnosis.db")
    state = DiagnosisState.start("session-1")
    state.symptom_code = "abnormal_noise"
    state.symptom_text = "设备持续异响"
    state.escalation_reason = "steps_exhausted"
    ticket = HandoffTicket.from_state(state)

    store.save_state(state)
    store.save_handoff(ticket)
    loaded = store.get_handoff(ticket.ticket_id)

    assert loaded is not None
    assert loaded.case_id == state.case_id
    assert loaded.reason == "steps_exhausted"
    assert loaded.issue_summary == "设备持续异响"


def test_store_releases_database_file_after_operations(tmp_path):
    db_path = tmp_path / "diagnosis.db"
    moved_path = tmp_path / "diagnosis-moved.db"
    store = DiagnosisStore(db_path)
    state = DiagnosisState.start("session-1")

    store.save_state(state)
    assert store.get_active_state("session-1") is not None
    db_path.rename(moved_path)

    assert moved_path.exists()
