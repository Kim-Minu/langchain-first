"""
[웹 브라우저 챗봇 서비스 - Streamlit & LangChain]

1. 모델 선택 지원:
   - Google Vertex AI Gemini (gemini-3.8-flash)
   - 로컬 Ollama Gemma (gemma4:e2b)
2. 모델 변경 시 에이전트 즉시 새로 생성 및 상태 동기화
3. 실시간 토큰 스트리밍(Streaming) 출력
4. 대화 문맥 기억: InMemorySaver
5. 응답별 토큰 사용량(입력/출력/총합) 및 예상 비용($ / 원화) 실시간 계산 및 표시

실행 방법:
    uv run streamlit run web_chat.py
"""

import os
import uuid

import google.auth
import streamlit as st
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

# ==============================================================================
# 1. 페이지 기본 설정 및 스타일
# ==============================================================================
st.set_page_config(
    page_title="LangChain Multi-Model Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 LangChain 멀티 모델 웹 챗봇")

# ==============================================================================
# 2. InMemorySaver (메모리 체크포인터) 싱글톤 인스턴스
# ==============================================================================
@st.cache_resource
def get_checkpointer():
    """
    모델을 변경하더라도 동일한 thread_id 내의 대화 기억이 유지되도록
    체크포인터는 단일 인스턴스로 관리합니다.
    """
    return InMemorySaver()


checkpointer = get_checkpointer()

# ==============================================================================
# 3. 모델 선택에 따라 새 에이전트를 직접 생성하는 함수
# ==============================================================================
def create_agent_for_model(model_name: str):
    """
    사용자가 선택한 모델에 맞춰 LLM 객체를 생성하고,
    해당 LLM을 탑재한 새로운 LangChain 에이전트를 생성하여 반환합니다.
    """
    # 1) Google Gemini 선택 시
    if "Gemini" in model_name:
        credentials, default_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or default_project
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        gemini_model_id = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

        llm = ChatGoogleGenerativeAI(
            model=gemini_model_id,
            vertexai=True,
            credentials=credentials,
            project=project_id,
            location=location,
            reasoning_effort="low",
            streaming=True,
            timeout=120,
            max_retries=0,
        )

    # 2) Ollama Gemma (gemma4:e2b) 선택 시
    else:
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            st.error("Ollama 모델을 사용하려면 터미널에서 `uv add langchain-ollama`를 먼저 실행해 주세요.")
            st.stop()

        llm = ChatOllama(
            model="gemma4:e2b",
            temperature=0,
        )

    # 도구(tools) 없이 순수 대화형 에이전트로 빌드 (InMemorySaver로 기억 유지)
    new_agent = create_agent(
        model=llm,
        tools=[],
        checkpointer=checkpointer,
        system_prompt=(
            "당신은 친절하고 유능한 AI 어시스턴트입니다. "
            "사용자의 이전 대화 맥락을 기억하여 일관성 있고 유익하게 답변하세요."
        ),
    )
    return new_agent


# ==============================================================================
# 4. 브라우저 세션 및 사이드바 설정 (모델 스위칭 및 누적 비용 통계)
# ==============================================================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [
        {
            "role": "assistant",
            "content": "안녕하세요! 무엇이든 물어보세요. 응답마다 토큰 사용량과 예상 비용을 함께 안내해 드립니다.",
            "model": "시스템",
        }
    ]

with st.sidebar:
    st.header("⚙️ 챗봇 설정")

    # 딱 2가지 모델 선택
    model_choice = st.selectbox(
        "🧠 사용할 모델 선택:",
        options=[
            "Google Gemini (Vertex AI)",
            "Ollama Gemma (gemma4:e2b)",
        ],
        index=0,
    )

    st.caption(f"**현재 세션 ID**: `{st.session_state.thread_id}`")

    # 새 대화 시작 버튼
    if st.button("🔄 새 대화 시작 (기억 초기화)", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.ui_messages = [
            {
                "role": "assistant",
                "content": "새로운 대화가 시작되었습니다. 이전 기억은 초기화되었습니다!",
                "model": "시스템",
            }
        ]
        st.session_state.agent = create_agent_for_model(model_choice)
        st.session_state.current_model = model_choice
        st.rerun()

    st.divider()

    # 모델 변경 감지 및 에이전트 즉시 재생성
    if ("current_model" not in st.session_state) or (st.session_state.current_model != model_choice):
        st.session_state.current_model = model_choice
        st.session_state.agent = create_agent_for_model(model_choice)
        st.toast(f"에이전트가 '{model_choice}' 모델로 새로 생성되었습니다!", icon="🤖")

    current_agent = st.session_state.agent

    # 세션 누적 사용량 통계
    st.subheader("📊 세션 누적 사용량")
    accum_tokens = sum(m.get("total_tokens", 0) for m in st.session_state.ui_messages)
    accum_usd = sum(m.get("cost_usd", 0.0) for m in st.session_state.ui_messages)
    st.metric("누적 사용 토큰", f"{accum_tokens:,} tokens")
    if accum_usd > 0:
        st.metric("누적 예상 비용", f"${accum_usd:.5f} (약 {accum_usd * 1380:.2f}원)")
    else:
        st.metric("누적 예상 비용", "$0.00 (무료)")

    st.divider()
    st.subheader("📦 InMemorySaver 상태")
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state = current_agent.get_state(config)
    saved_msgs = state.values.get("messages", [])
    st.write(f"메모리에 누적된 메시지 수: **{len(saved_msgs)}개**")

    with st.expander("저장된 LangChain 메시지 목록 보기"):
        for i, m in enumerate(saved_msgs):
            content_snippet = m.content if isinstance(m.content, str) else str(m.content)
            st.markdown(f"**[{i+1}] {m.type.upper()}**: {content_snippet[:50]}...")

# 메인 화면에 현재 활성화된 모델 뱃지 표시
st.caption(f"현재 활성화된 에이전트 모델: 🟢 **{st.session_state.current_model}**")

# ==============================================================================
# 5. 이전 대화 화면 렌더링 (모델명, 토큰 수, 예상 비용 표시)
# ==============================================================================
for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("model") and msg["model"] != "시스템":
            meta_text = f"🤖 **모델**: `{msg['model']}`"
            if "total_tokens" in msg and msg["total_tokens"] > 0:
                meta_text += (
                    f" | 📊 **토큰**: 입력 {msg['in_tokens']:,} + 출력 {msg['out_tokens']:,} = 총 {msg['total_tokens']:,}개"
                )
            if "cost_str" in msg:
                meta_text += f" | 💰 **예상 비용**: `{msg['cost_str']}`"
            st.caption(meta_text)

# ==============================================================================
# 6. 사용자 입력 처리, 실시간 스트리밍 및 토큰/비용 계산
# ==============================================================================
if user_input := st.chat_input("메시지를 입력하세요..."):
    # UI에 사용자 질문 표시
    st.session_state.ui_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        agent_config = {"configurable": {"thread_id": st.session_state.thread_id}}

        def generate_response_stream():
            for chunk, metadata in current_agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=agent_config,
                stream_mode="messages",
            ):
                content = chunk.content
                if content:
                    if isinstance(content, str):
                        yield content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, str):
                                yield item
                            elif isinstance(item, dict) and "text" in item:
                                yield item["text"]

        try:
            # 1) 실시간 스트리밍 렌더링
            response_text = st.write_stream(generate_response_stream())

            # 2) 토큰 사용량 추출 (InMemorySaver의 최신 AIMessage 메타데이터 확인)
            latest_state = current_agent.get_state(agent_config)
            saved_messages = latest_state.values.get("messages", [])
            last_ai_msg = saved_messages[-1] if saved_messages else None

            # 메타데이터에서 토큰 정보 가져오기
            usage = getattr(last_ai_msg, "usage_metadata", None)
            if not usage and hasattr(last_ai_msg, "response_metadata"):
                usage = last_ai_msg.response_metadata.get("usage_metadata") or last_ai_msg.response_metadata.get("token_usage")

            if usage and isinstance(usage, dict):
                in_tokens = usage.get("input_tokens", 0)
                out_tokens = usage.get("output_tokens", 0)
                total_tokens = usage.get("total_tokens", in_tokens + out_tokens)
            else:
                # 메타데이터가 없는 경우 한국어/영어 글자 수 기반 근사치 계산 (1토큰 ≈ 3글자)
                in_tokens = max(1, len(user_input) // 3)
                out_tokens = max(1, len(response_text) // 3)
                total_tokens = in_tokens + out_tokens

            # 3) 예상 비용 계산
            # Google Gemini Flash 공식 요율 (입력: $0.075 / 100만 토큰, 출력: $0.30 / 100만 토큰)
            if "Gemini" in st.session_state.current_model:
                cost_usd = (in_tokens * 0.075 / 1_000_000) + (out_tokens * 0.30 / 1_000_000)
                cost_krw = cost_usd * 1380  # 기준 환율 1,380원
                cost_str = f"${cost_usd:.6f} (약 {cost_krw:.3f}원)"
            else:
                # 로컬 Ollama 모델은 내 PC의 자원을 사용하므로 $0.00
                cost_usd = 0.0
                cost_str = "$0.00 (로컬 무료)"

            # 4) 응답 말풍선 하단에 토큰 및 비용 정보 출력
            st.caption(
                f"🤖 **모델**: `{st.session_state.current_model}` | "
                f"📊 **토큰**: 입력 {in_tokens:,} + 출력 {out_tokens:,} = 총 {total_tokens:,}개 | "
                f"💰 **예상 비용**: `{cost_str}`"
            )

            # 5) UI 세션 히스토리에 저장
            st.session_state.ui_messages.append({
                "role": "assistant",
                "content": response_text,
                "model": st.session_state.current_model,
                "in_tokens": in_tokens,
                "out_tokens": out_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "cost_str": cost_str,
            })
        except Exception as e:
            st.error(f"모델 응답 중 오류가 발생했습니다: {e}")
