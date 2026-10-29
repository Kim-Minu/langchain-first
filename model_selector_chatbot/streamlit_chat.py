"""Model Selector Chatbot - Streamlit Web Application

GCP Vertex AI MaaS (Model-as-a-Service) 환경에서
DeepSeek V4.1 Flash 및 Gemma 4 26B A4B IT 모델을 선택하여
대화할 수 있는 LangChain 웹 챗봇입니다.

실행 방법:
    streamlit run model_selector_chatbot/streamlit_chat.py
"""

import sys
import os

# site-packages 및 모듈 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv/lib/python3.14/site-packages")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from model_selector_chatbot.config import SUPPORTED_MODELS, ModelConfig
from model_selector_chatbot.model_service import GcpMaasChatModel
from model_selector_chatbot.chatbot import ModelSelectorChatbot, DEFAULT_SYSTEM_PROMPT

st.set_page_config(
    page_title="GCP MaaS 모델 선택 챗봇",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 GCP Vertex AI MaaS 모델 선택 챗봇")
st.caption("DeepSeek V4.1 Flash & Google Gemma 4 26B A4B IT 모델 선택 및 LangChain 대화 테스트")

# 사이드바 설정
with st.sidebar:
    st.header("🎛️ 모델 선택 (Model Selection)")

    model_options = {
        "deepseek-flash": "⚡ DeepSeek V4.1 Flash (MaaS)",
        "gemma-4-26b": "🧠 Gemma 4 26B A4B IT (MaaS)",
    }

    selected_model_key = st.selectbox(
        "사용할 모델 선택",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
        index=0,
    )
    meta = SUPPORTED_MODELS[selected_model_key]
    st.info(f"**공식 명칭:** `{meta['service_name']}`\n\n**모델 ID:** `{meta['id']}`\n\n{meta['description']}")

    st.divider()
    st.subheader("🌐 엔드포인트 & GCP 설정")

    endpoint_host = st.text_input(
        "ENDPOINT (${ENDPOINT})",
        value=os.getenv("ENDPOINT", meta["default_endpoint"]),
    )
    region = st.text_input(
        "REGION (${REGION})",
        value=os.getenv("REGION", meta["default_region"]),
    )
    project_id = st.text_input(
        "PROJECT_ID (${PROJECT_ID})",
        value=os.getenv("PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "iceu-songpa05")),
    )

    assembled_url = (
        f"https://{endpoint_host.rstrip('/')}/v1/projects/{project_id}/"
        f"locations/{region}/endpoints/openapi/chat/completions"
    )
    st.caption(f"**호출 URL:**\n`{assembled_url}`")

    st.divider()
    st.subheader("⚙️ 추론 파라미터")

    # Gemma 4 전용 Thinking 옵션
    enable_thinking = False
    if meta.get("supports_thinking", False):
        enable_thinking = st.toggle(
            "🧠 Enable Thinking (사고 모드)",
            value=True,
            help='Gemma 4 사양: {"chat_template_kwargs": {"enable_thinking": true}} 파라미터 전달',
        )

    # 모델에 따른 Max Tokens 기본/최댓값
    default_max = meta["default_max_tokens"]
    max_limit = 128000 if selected_model_key == "gemma-4-26b" else 8192
    max_tokens = st.number_input(
        "Max Output Tokens",
        min_value=256,
        max_value=max_limit,
        value=default_max,
        step=512 if selected_model_key != "gemma-4-26b" else 2048,
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)

    st.subheader("테스트 모드")
    mock_mode = st.toggle("Mock(오프라인 테스트) 모드", value=False, help="외부 네트워크나 GCP 인증 없이 LangChain 파이프라인을 검증합니다.")

    # 모델별 기본 시스템 프롬프트 반영
    model_default_sys_prompt = meta.get("default_system_prompt", DEFAULT_SYSTEM_PROMPT)
    system_prompt = st.text_area(
        "System Prompt (시스템 프롬프트)",
        value=model_default_sys_prompt,
        height=100,
        key=f"sys_prompt_input_{selected_model_key}",
    )

    if st.button("🧹 대화 내역 초기화", width="stretch"):
        st.session_state.messages = []
        if "chatbot" in st.session_state:
            st.session_state.chatbot.clear_history()
        st.rerun()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 모델 설정 빌드
current_config = ModelConfig.from_model_key(
    model_key=selected_model_key,
    project_id=project_id,
    endpoint_host=endpoint_host,
    location=region,
    temperature=temperature,
    max_tokens=max_tokens,
    enable_thinking=enable_thinking,
    mock_mode=mock_mode,
)

service = GcpMaasChatModel(config=current_config)

# 챗봇 인스턴스 관리
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ModelSelectorChatbot(
        service=service,
        config=current_config,
        system_prompt=system_prompt,
    )
else:
    # 모델이나 파라미터가 변경된 경우 챗봇 백엔드 갱신 (히스토리는 유지)
    st.session_state.chatbot.service = service
    st.session_state.chatbot.config = current_config
    st.session_state.chatbot.system_prompt = system_prompt
    st.session_state.chatbot._rebuild_chain()

# 상단 상태 뱃지
badge_col1, badge_col2, badge_col3 = st.columns([2, 2, 2])
with badge_col1:
    st.markdown(f"**현재 활성 모델:** `{meta['id']}`")
with badge_col2:
    thinking_text = "🟢 켜짐" if enable_thinking else "⚪ 꺼짐"
    st.markdown(f"**Thinking 모드:** {thinking_text}")
with badge_col3:
    st.markdown(f"**엔드포인트:** `{endpoint_host} ({region})`")

# 기존 대화 내역 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
if user_prompt := st.chat_input("질문 또는 메시지를 입력하세요..."):
    # 1. 사용자 메시지 화면 출력
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. 어시스턴트 스트리밍 응답
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            for chunk in st.session_state.chatbot.stream_ask(user_prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            error_msg = (
                f"⚠️ **호출 오류 발생**: {e}\n\n"
                f"**확인 사항:**\n"
                f"- GCP ADC 인증: `gcloud auth application-default login` 실행 여부\n"
                f"- 프로젝트/엔드포인트 권한 확인\n"
                f"- 오프라인/로컬 테스트인 경우 사이드바에서 **Mock 모드**를 켜보세요."
            )
            message_placeholder.error(error_msg)
