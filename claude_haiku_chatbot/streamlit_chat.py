"""Claude Haiku 5.5 - Streamlit Web Chatbot

GCP Vertex AI 환경의 Anthropic claude-haiku-5-5 모델을 브라우저에서 편리하게
테스트하고 대화할 수 있는 Streamlit 챗봇 애플리케이션입니다.

실행 방법:
    streamlit run claude_haiku_chatbot/streamlit_chat.py
"""

from __future__ import annotations

import os
import sys

# 모듈 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv/lib/python3.14/site-packages")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from claude_haiku_chatbot.chatbot import ClaudeHaikuChatbot
from claude_haiku_chatbot.claude_service import ClaudeHaikuVertexChatModel
from claude_haiku_chatbot.config import ClaudeHaikuConfig, default_config

st.set_page_config(
    page_title="Claude Haiku 5.5 Chatbot",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Anthropic Claude Haiku 5.5 Chatbot")
st.caption("Google Cloud Vertex AI & LangChain 기반 차세대 Claude Haiku 챗봇")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 서비스 설정")
    st.info("대상 모델: **claude-haiku-5-5** (Anthropic Vertex)")

    project_id = st.text_input(
        "GCP Project ID",
        value=os.getenv("PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "iceu-songpa05")),
    )
    region = st.text_input("GCP Region", value=os.getenv("REGION", "global"))
    model_name = st.text_input("Model ID", value="claude-haiku-5-5")

    st.subheader("추론 파라미터")
    max_tokens = st.number_input("Max Output Tokens", min_value=128, max_value=8192, value=1024, step=128)

    st.subheader("테스트 모드")
    mock_mode = st.toggle(
        "Mock(오프라인 테스트) 모드",
        value=False,
        help="외부 네트워크나 GCP 인증 없이 LangChain 파이프라인과 UI를 테스트합니다.",
    )

    system_prompt = st.text_area("System Prompt", value=default_config.default_system_prompt, height=120)

    if st.button("🧹 대화 내역 초기화", width="stretch"):
        st.session_state.messages = []
        if "chatbot" in st.session_state:
            st.session_state.chatbot.clear_history()
        st.rerun()

# 세션 상태 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

# 설정 객체 생성
config = ClaudeHaikuConfig(
    project_id=project_id,
    region=region,
    model_name=model_name,
    max_tokens=int(max_tokens),
    mock_mode=mock_mode,
    default_system_prompt=system_prompt,
)

service = ClaudeHaikuVertexChatModel(config=config)
chatbot = ClaudeHaikuChatbot(service=service, config=config, system_prompt=system_prompt)

# 기존 히스토리를 챗봇 엔진에 반영
chatbot.clear_history()
for msg in st.session_state.messages:
    if msg["role"] == "user":
        chatbot.history.add_user_message(msg["content"])
    elif msg["role"] == "assistant":
        chatbot.history.add_ai_message(msg["content"])

st.session_state.chatbot = chatbot

# 기존 메시지 화면 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
if user_prompt := st.chat_input("질문을 입력하세요..."):
    # 1. 사용자 메시지 기록 및 렌더링
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. 어시스턴트 스트리밍 응답
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            for chunk in chatbot.stream_ask(user_prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            err_msg = f"❌ 오류 발생: {e}"
            message_placeholder.error(err_msg)

