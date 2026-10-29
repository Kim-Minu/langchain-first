"""
[Google Gemini 스타일 웹 브라우저 챗봇 - Streamlit & LangChain]

- 디자인: Google Gemini (https://gemini.google.com/app?hl=ko) 레이아웃 기준
    · 왼쪽 사이드바: 새 채팅 + 최근 대화 목록 + 설정/사용량
    · 상단 바: 모델 선택 메뉴(모델 · Google 검색 방식) + 새 채팅 아이콘
    · 첫 화면: "무엇을 도와드릴까요?" + 화면 중앙 입력창
    · 대화 화면: 사용자 말풍선(오른쪽) / AI 답변(왼쪽, 말풍선 없음) + 하단 입력창
- 기능:
    1) Google 검색 방식 선택: 끄기 / Vertex AI Grounding (+ LangGraph 캐시로 같은 맥락의 답변 재사용)
    2) 모델 선택 (Google Gemini Vertex AI & Ollama Gemma) + 사고 수준(reasoning_effort: 낮음 / 보통 / 높음)
    3) 실시간 토큰 스트리밍 (Streaming)
    4) 대화 내역 PostgreSQL 저장 (PostgresSaver + 대화 목록 테이블) + 여러 대화(thread) 전환
    5) 토큰 사용량 및 예상 비용 표시
    6) 구조화된 출력: 원샷 예시(JSON 또는 '라벨 : 값' 텍스트) → 같은 형식의 답변 (ProviderStrategy)
    7) 링크 읽기: 메시지 속 URL 페이지를 Gemini가 직접 읽음 (URL context)
       코드 실행: Gemini가 Python 코드를 Google 샌드박스에서 실행하고 결과 확인 (Code execution)
    8) 오른쪽 실행 설정 패널에서 모든 옵션 선택
    9) 날씨 카드: 날씨 질문에 Open-Meteo 예보로 현재 · 시간별(온도/강수량/바람) 카드 표시
    10) 토큰 사용량 대시보드 (pages/usage_dashboard.py): 일별 · 주별 · 월별 × 모델별 통계

이 파일은 실행 진입점으로, 화면 흐름만 담고 기능은 chat_app/ 모듈에 나뉘어 있습니다.

실행 방법:
    docker compose up -d          # 로컬 PostgreSQL 실행 (DATABASE_URL로 다른 DB 지정 가능)
    uv run streamlit run chat_app/web_chat.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (chat_app 패키지 임포트 지원)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from chat_app.agent import create_agent_for_model, create_llm
from chat_app.config import WEATHER_EXTRACT_MAX_TOKENS, is_search_active
from chat_app.db import get_checkpointer, get_graph_cache, load_messages
from chat_app.state import init_session_state
from chat_app.styles import inject_styles
from chat_app.ui import (
    render_conversation,
    render_hero,
    render_settings_panel,
    render_sidebar,
    render_top_bar,
    stream_assistant_reply,
)

# 1. 페이지 설정 및 Gemini 다크 테마
st.set_page_config(
    page_title="Gemini 스타일 챗봇",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_styles()

# 2. DB 연결 (PostgresSaver + 대화 목록 테이블)
try:
    checkpointer = get_checkpointer()
    graph_cache = get_graph_cache()  # LangGraph 캐시 (Grounding 답변 재사용)
except Exception as e:
    st.error(
        f"PostgreSQL 연결 또는 초기화에 실패했습니다: {e}\n\n"
        "프로젝트 폴더에서 `docker compose up -d`로 DB를 실행하거나 `DATABASE_URL` 환경변수를 확인하세요."
    )
    st.stop()

# 3. 세션 상태 및 현재 대화 불러오기
init_session_state()
messages = load_messages(st.session_state.thread_id)

# 4. 사이드바(대화 목록) · 상단 바 · 오른쪽 실행 설정 패널(모든 옵션)
render_sidebar()
render_top_bar()
settings = render_settings_panel()
search_active = is_search_active(settings.model_choice, settings.search_mode)

# 5. 실행 설정 변경 감지 ➔ 에이전트 새로 생성
if st.session_state.get("current_agent_key") != settings.agent_key():
    try:
        st.session_state.agent = create_agent_for_model(
            settings.model_choice,
            settings.search_mode,
            checkpointer,
            max_output_tokens=settings.max_output_tokens,
            cache=graph_cache,
            output_format=settings.output_format,
            url_context=settings.url_context,
            code_execution=settings.code_execution,
            reasoning_effort=settings.reasoning_effort,
        )
    except ImportError:
        st.error("Ollama 모델을 사용하려면 터미널에서 `uv add langchain-ollama`를 먼저 실행해 주세요.")
        st.stop()
    st.session_state.current_agent_key = settings.agent_key()

# 날씨 카드용 장소 추출 LLM (선택한 모델 · 낮은 사고 수준, 모델이 바뀔 때만 새로 생성)
if st.session_state.get("weather_llm_model") != settings.model_choice:
    try:
        st.session_state.weather_llm = create_llm(settings.model_choice, WEATHER_EXTRACT_MAX_TOKENS, "low")
    except ImportError:
        st.session_state.weather_llm = None
    st.session_state.weather_llm_model = settings.model_choice

# 6. 첫 화면(중앙 입력창) / 대화 화면(하단 입력창)
if not messages:
    render_hero()
    st.stop()

render_conversation(messages)

# 7. 대기 중인 질문이 있으면 답변 스트리밍
if user_input := st.session_state.pending_prompt:
    st.session_state.pending_prompt = None
    weather_llm = st.session_state.weather_llm if settings.weather_card else None
    stream_assistant_reply(st.session_state.agent, user_input, settings, search_active, weather_llm)
