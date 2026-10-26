"""세션 상태(현재 대화 thread_id, 대기 중인 질문)와 대화 전환 동작.

대화 목록·메시지는 DB에서 읽고, 세션에는 "지금 보고 있는 대화"만 둡니다.
thread_id를 URL(?t=...)에도 남겨 새로고침해도 같은 대화가 열리게 합니다.
"""

import uuid

import streamlit as st

from chat_app.db import delete_thread, rename_thread, save_message

THREAD_TITLE_MAX_LENGTH = 100


def init_session_state():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = st.query_params.get("t") or str(uuid.uuid4())
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "confirm_delete_thread" not in st.session_state:
        st.session_state.confirm_delete_thread = None


def start_new_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pending_prompt = None
    st.query_params.clear()


def select_thread(thread_id: str):
    st.session_state.thread_id = thread_id
    st.session_state.pending_prompt = None
    st.query_params["t"] = thread_id


def rename_thread_from_input(thread_id: str):
    """⋮ 메뉴의 이름 입력칸(key=rename_<thread_id>) 값으로 대화 이름을 바꿉니다. 빈 값은 무시합니다."""
    title = st.session_state.get(f"rename_{thread_id}", "").strip()[:THREAD_TITLE_MAX_LENGTH]
    if title:
        rename_thread(thread_id, title)


def request_delete_thread(thread_id: str):
    """삭제 확인 창을 띄우도록 표시합니다 (실제 삭제는 확인 창에서)."""
    st.session_state.confirm_delete_thread = thread_id


def confirm_delete_thread(thread_id: str):
    delete_thread(thread_id)
    st.session_state.confirm_delete_thread = None
    if st.session_state.thread_id == thread_id:
        start_new_chat()


def cancel_delete_thread():
    st.session_state.confirm_delete_thread = None


def submit_prompt(prompt: str):
    """질문을 DB에 저장하고, 다음 실행에서 대화 화면으로 전환해 답변을 스트리밍합니다."""
    save_message(st.session_state.thread_id, "user", prompt, title=prompt[:30])
    st.session_state.pending_prompt = prompt
    st.query_params["t"] = st.session_state.thread_id
    st.rerun()
