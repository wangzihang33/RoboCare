from __future__ import annotations

import time
import uuid
from html import escape

import streamlit as st

from agent.react_agent import ReactAgent
from agent.troubleshooting.models import DiagnosisStatus


st.set_page_config(
    page_title="RoboCare | 智能客服系统",
    page_icon="RC",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
:root {
    --ink: #1d252c;
    --muted: #69737b;
    --line: #dfe5e8;
    --paper: #ffffff;
    --canvas: #f4f6f5;
    --navy: #17324d;
    --teal: #167d78;
    --coral: #c86645;
    --soft-teal: #e6f3f1;
    --shadow: 0 14px 34px rgba(23, 50, 77, 0.08);
}

html, body, [class*="css"] {
    font-family: "DM Sans", "Noto Sans SC", sans-serif;
    color: var(--ink);
}

.stApp { background: var(--canvas); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #eef2f1; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding: 1.35rem 1.1rem 1.1rem; }

.brand-lockup { display: flex; align-items: center; gap: 0.7rem; padding: 0.25rem 0.15rem 1.5rem; }
.brand-mark {
    width: 38px; height: 38px; display: grid; place-items: center; border-radius: 10px;
    background: var(--navy); color: #fff; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.05em;
}
.brand-name { color: var(--navy); font-size: 1.02rem; font-weight: 700; line-height: 1.1; }
.brand-caption { color: var(--muted); font-size: 0.72rem; margin-top: 0.2rem; }
.sidebar-section-label { color: #879198; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; margin: 0.9rem 0 0.55rem; }
.sidebar-note { background: rgba(255,255,255,0.72); border: 1px solid var(--line); border-radius: 8px; padding: 0.78rem 0.82rem; }
.sidebar-note { color: var(--muted); font-size: 0.75rem; line-height: 1.55; margin-top: 1rem; }
.sidebar-note strong { color: var(--ink); display: block; margin-bottom: 0.22rem; }

.main-shell { max-width: 1180px; margin: 0 auto; padding: 1.2rem 1rem 4.5rem; }
.topbar { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; padding: 0.4rem 0 1.05rem; }
.eyebrow { color: var(--teal); font-size: 0.71rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.45rem; }
.page-title { color: var(--navy); font-size: clamp(1.42rem, 3vw, 2.05rem); font-weight: 700; line-height: 1.15; margin: 0; }
.page-subtitle { color: var(--muted); font-size: 0.86rem; margin-top: 0.42rem; }
.online-badge { align-items: center; background: var(--paper); border: 1px solid var(--line); border-radius: 999px; color: #46605d; display: inline-flex; font-size: 0.75rem; gap: 0.44rem; padding: 0.45rem 0.72rem; white-space: nowrap; }
.online-dot { background: #28a27a; border-radius: 50%; box-shadow: 0 0 0 4px #dff4eb; height: 7px; width: 7px; }

.metric-strip { background: var(--paper); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); display: grid; grid-template-columns: repeat(3, 1fr); margin: 0.35rem 0 1.1rem; overflow: hidden; }
.metric { border-right: 1px solid var(--line); padding: 0.78rem 1rem; }
.metric:last-child { border-right: 0; }
.metric-label { color: var(--muted); font-size: 0.71rem; }
.metric-value { color: var(--navy); font-size: 0.95rem; font-weight: 700; margin-top: 0.22rem; }

.diagnosis-strip { align-items: center; background: var(--soft-teal); border: 1px solid #bfe2dd; border-radius: 8px; display: flex; gap: 0.8rem; justify-content: space-between; margin: 0 0 1rem; padding: 0.72rem 0.9rem; }
.diagnosis-title { color: #176b68; font-size: 0.78rem; font-weight: 700; }
.diagnosis-meta { color: #47716e; font-size: 0.73rem; margin-top: 0.18rem; }
.diagnosis-status { background: #fff; border: 1px solid #b8d9d5; border-radius: 999px; color: #176b68; font-size: 0.71rem; padding: 0.34rem 0.6rem; white-space: nowrap; }

.welcome-panel { background: var(--paper); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); margin-bottom: 1rem; overflow: hidden; }
.welcome-band { background: var(--navy); color: #fff; padding: 1.25rem 1.35rem 1.1rem; }
.welcome-band h2 { font-size: 1.24rem; font-weight: 700; margin: 0; }
.welcome-band p { color: #d5e2e9; font-size: 0.82rem; line-height: 1.55; margin: 0.42rem 0 0; }
.prompt-label { color: var(--muted); font-size: 0.75rem; font-weight: 600; margin: 1rem 1.35rem 0.62rem; }
section.main [data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,0.52); border-color: var(--line); border-radius: 8px; min-height: 24rem; padding: 0.3rem 0.5rem; }
[data-testid="stChatMessage"] { border: 0; padding: 0.75rem 0.6rem; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { font-size: 0.9rem; line-height: 1.72; }
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] { background: var(--coral); }
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] { background: var(--teal); }
[data-testid="stChatInput"] { margin-top: 0.8rem; }
[data-testid="stChatInput"] textarea { background: var(--paper); border: 1px solid #cbd6d9; border-radius: 8px; color: var(--ink); font-size: 0.9rem; }
[data-testid="stChatInput"] textarea:focus { border-color: var(--teal); box-shadow: 0 0 0 3px rgba(22,125,120,0.12); }
.footer-note { color: #89939a; font-size: 0.7rem; margin-top: 1.1rem; text-align: center; }
button[kind="secondary"] { border-radius: 7px; font-weight: 600; }

@media (max-width: 720px) {
    .main-shell { padding: 0.75rem 0.6rem 3.8rem; }
    .topbar { align-items: flex-start; flex-direction: column; }
    .metric-strip { grid-template-columns: 1fr; }
    .metric { border-bottom: 1px solid var(--line); border-right: 0; }
    .metric:last-child { border-bottom: 0; }
    .diagnosis-strip { align-items: flex-start; flex-direction: column; }
}
</style>
""",
    unsafe_allow_html=True,
)


PROMPT_PRESETS = (
    ("滤网清洁周期", "扫地机器人滤网多久清洗一次？"),
    ("E01 主刷卡住", "主刷卡住同时出现 E01，应该怎么处理？"),
    ("深圳拖地建议", "深圳今天适合使用扫拖一体机器人的拖地功能吗？"),
)


def _init_session() -> None:
    if "agent" not in st.session_state:
        st.session_state["agent"] = ReactAgent()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"sess_{uuid.uuid4().hex[:12]}"
    if "pending_prompt" not in st.session_state:
        st.session_state["pending_prompt"] = None


def _reset_session() -> None:
    st.session_state["messages"] = []
    st.session_state["session_id"] = f"sess_{uuid.uuid4().hex[:12]}"
    st.session_state["pending_prompt"] = None


def _diagnosis_state():
    agent = st.session_state["agent"]
    return agent.troubleshooting_engine.store.get_active_state(
        st.session_state["session_id"]
    )


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-lockup">
                <div class="brand-mark">RC</div>
                <div>
                    <div class="brand-name">RoboCare</div>
                    <div class="brand-caption">智能客服系统</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("＋  新建会话", use_container_width=True, type="primary"):
            _reset_session()
            st.rerun()

        st.markdown('<div class="sidebar-section-label">示例问题</div>', unsafe_allow_html=True)
        for index, (label, prompt) in enumerate(PROMPT_PRESETS):
            if st.button(label, key=f"preset_{index}", use_container_width=True):
                st.session_state["pending_prompt"] = prompt
                st.rerun()

        st.markdown(
            """
            <div class="sidebar-note">
                <strong>工作状态</strong>
                路由、知识检索、故障诊断与工具调用日志均由同一会话上下文管理。
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_header() -> None:
    st.markdown(
        """
        <div class="topbar">
            <div>
                <div class="eyebrow">Customer support / 01</div>
                <h1 class="page-title">RoboCare 智能客服系统</h1>
                <div class="page-subtitle">让每一次咨询，都有清晰的依据和下一步。</div>
            </div>
            <div class="online-badge"><span class="online-dot"></span>系统在线</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metrics() -> None:
    messages = st.session_state["messages"]
    user_turns = sum(message["role"] == "user" for message in messages)
    diagnosis = _diagnosis_state()
    mode = "故障诊断" if diagnosis else "智能客服"
    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric"><div class="metric-label">当前模式</div><div class="metric-value">{mode}</div></div>
            <div class="metric"><div class="metric-label">会话轮次</div><div class="metric-value">{user_turns} turns</div></div>
            <div class="metric"><div class="metric-label">知识来源</div><div class="metric-value">本地知识库 · Web · 业务工具</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_diagnosis_strip() -> None:
    state = _diagnosis_state()
    if state is None:
        return
    status_labels = {
        DiagnosisStatus.COLLECTING: "收集现象",
        DiagnosisStatus.WAITING_FEEDBACK: "等待反馈",
        DiagnosisStatus.RESOLVED: "已解决",
        DiagnosisStatus.ESCALATED: "已转人工",
        DiagnosisStatus.CANCELLED: "已结束",
    }
    label = status_labels.get(state.status, state.status.value)
    symptom = state.symptom_code or "待识别故障"
    step = state.current_step_id or "等待下一步"
    st.markdown(
        f"""
        <div class="diagnosis-strip">
            <div>
                <div class="diagnosis-title">诊断会话 · {escape(symptom)}</div>
                <div class="diagnosis-meta">当前步骤：{escape(step)} · 已记录 {len(state.attempts)} 次尝试</div>
            </div>
            <div class="diagnosis-status">{escape(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_welcome() -> None:
    st.markdown(
        """
        <div class="welcome-panel">
            <div class="welcome-band">
                <h2>你好，我是 RoboCare 客服 Agent</h2>
                <p>从产品使用、维护保养到故障排查，直接描述你正在遇到的情况。</p>
            </div>
            <div class="prompt-label">从一个问题开始</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _capture_stream(generator, chunks: list[str]):
    for chunk in generator:
        chunks.append(chunk)
        for character in chunk:
            time.sleep(0.004)
            yield character


def _handle_prompt(prompt: str) -> None:
    prompt = prompt.strip()
    if not prompt:
        return

    history = list(st.session_state["messages"])
    st.session_state["messages"].append({"role": "user", "content": prompt})
    chunks: list[str] = []

    with st.chat_message("assistant", avatar=":material/support_agent:"):
        try:
            with st.status("正在检索并组织答案", expanded=False):
                stream = st.session_state["agent"].execute_stream(
                    prompt,
                    history=history,
                    session_id=st.session_state["session_id"],
                )
                st.write_stream(_capture_stream(stream, chunks))
            response = "".join(chunks).strip()
            if response:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response}
                )
        except Exception as exc:
            st.error("本次请求没有完成，请稍后重试。")
            st.caption(f"{type(exc).__name__}: {exc}")

    st.rerun()


_init_session()
_render_sidebar()

st.markdown('<main class="main-shell">', unsafe_allow_html=True)
_render_header()
_render_metrics()
_render_diagnosis_strip()

if not st.session_state["messages"]:
    _render_welcome()

with st.container(border=True):
    for message in st.session_state["messages"]:
        avatar = (
            ":material/support_agent:"
            if message["role"] == "assistant"
            else ":material/person:"
        )
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

pending_prompt = st.session_state.pop("pending_prompt", None)
prompt = st.chat_input("描述设备问题、产品使用或需要查询的内容")
if pending_prompt:
    _handle_prompt(pending_prompt)
elif prompt:
    _handle_prompt(prompt)

st.markdown(
    '<div class="footer-note">RoboCare · evidence-led support workspace</div></main>',
    unsafe_allow_html=True,
)
