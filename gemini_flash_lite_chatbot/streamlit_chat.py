"""Gemini 3.5 Flash Lite - Streamlit Web Chatbot

GCP Vertex AI 환경의 gemini-3.5-flash-lite 모델을 브라우저에서 편리하게
테스트하고 대화할 수 있는 Streamlit 챗봇 애플리케이션입니다.

실행 방법:
    streamlit run gemini_flash_lite_chatbot/streamlit_chat.py
"""

import sys
import os

# 모듈 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv/lib/python3.14/site-packages")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from gemini_flash_lite_chatbot.config import GeminiConfig
from gemini_flash_lite_chatbot.gemini_service import GeminiFlashLiteService
from gemini_flash_lite_chatbot.chatbot import GeminiFlashLiteChatbot, DEFAULT_SYSTEM_PROMPT

st.set_page_config(
    page_title="Gemini 3.5 Flash Lite Chatbot",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Gemini 3.5 Flash Lite Chatbot")
st.caption("Google Cloud Vertex AI & LangChain 기반 경량 초고속 챗봇")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 서비스 설정")
    st.info("대상 모델: **gemini-3.5-flash-lite**")

    project_id = st.text_input("GCP Project ID", value=os.getenv("GOOGLE_CLOUD_PROJECT", "iceu-songpa05"))
    location = st.text_input("GCP Region", value=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    model_name = st.text_input("Model ID", value="gemini-3.5-flash-lite")

    st.subheader("추론 파라미터")
    reasoning_effort = st.selectbox(
        "Reasoning Effort (사고 수준)",
        options=["low", "medium", "high"],
        index=0,
        help="Gemini 3.x 모델의 추론(Thinking) 깊이를 지정합니다.",
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)
    max_output_tokens = st.number_input("Max Output Tokens", min_value=256, max_value=65536, value=4096, step=512)

    st.subheader("테스트 모드")
    mock_mode = st.toggle("Mock(오프라인 테스트) 모드", value=False, help="외부 네트워크나 GCP 인증 없이 LangChain 파이프라인을 테스트합니다.")

    system_prompt = st.text_area("System Prompt", value=DEFAULT_SYSTEM_PROMPT, height=120)

    if st.button("🧹 대화 내역 초기화", width="stretch"):
        st.session_state.messages = []
        if "chatbot" in st.session_state:
            st.session_state.chatbot.clear_history()
        st.rerun()

# 세션 상태 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

config = GeminiConfig(
    project_id=project_id,
    location=location,
    model_name=model_name,
    reasoning_effort=reasoning_effort,
    temperature=temperature,
    max_output_tokens=max_output_tokens,
    mock_mode=mock_mode,
)

service = GeminiFlashLiteService(config=config)
chatbot = GeminiFlashLiteChatbot(service=service, config=config, system_prompt=system_prompt)
st.session_state.chatbot = chatbot

# 기존 메시지 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
if user_prompt := st.chat_input("질문을 입력하세요..."):
    # 1. 사용자 메시지
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
            error_msg = f"⚠️ 오류 발생: {e}\n\n*GCP ADC 인증(`gcloud auth application-default login`) 확인 또는 사이드바의 Mock 모드를 켜서 테스트해 보세요.*"
            message_placeholder.error(error_msg)

