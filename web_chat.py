"""
[Google Gemini 스타일 웹 브라우저 챗봇 - Streamlit & LangChain]

- 디자인: Google Gemini (https://gemini.google.com/app?hl=ko) 레이아웃 기준
    · 왼쪽 사이드바: 새 채팅 + 최근 대화 목록 + 설정/사용량
    · 상단 바: 모델 선택 메뉴(모델 · Google 검색 방식) + 새 채팅 아이콘
    · 첫 화면: "무엇을 도와드릴까요?" + 화면 중앙 입력창
    · 대화 화면: 사용자 말풍선(오른쪽) / AI 답변(왼쪽, 말풍선 없음) + 하단 입력창
- 기능:
    1) Google 검색 방식 선택: 끄기 / Vertex AI Grounding / Custom Search JSON API (CSE)
    2) 모델 선택 (Google Gemini Vertex AI & Ollama Gemma)
    3) 실시간 토큰 스트리밍 (Streaming)
    4) 대화 문맥 기억 (InMemorySaver) + 여러 대화(thread) 전환
    5) 토큰 사용량 및 예상 비용 표시

실행 방법:
    uv run streamlit run web_chat.py
    # CSE 검색을 사용하려면 GOOGLE_CSE_API_KEY, GOOGLE_CSE_ID 환경변수 필요
    uv run --env-file .env streamlit run web_chat.py
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid

import google.auth
import streamlit as st
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

# ==============================================================================
# 1. 페이지 기본 설정 및 Gemini 다크 모드 스타일 (CSS)
# ==============================================================================
st.set_page_config(
    page_title="Gemini 스타일 챗봇",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500&display=swap');

    :root {
        --gm-bg: #131314;
        --gm-surface: #1e1f20;
        --gm-surface-hover: #282a2c;
        --gm-user-bubble: #333537;
        --gm-text: #e3e3e3;
        --gm-text-muted: #9aa0a6;
        --gm-accent: #8ab4f8;
        --gm-selected: rgba(138, 180, 248, 0.16);
    }

    /* 전체 배경 및 폰트 */
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--gm-bg) !important;
        color: var(--gm-text) !important;
        font-family: 'Google Sans', 'Google Sans Text', 'Noto Sans KR', -apple-system, sans-serif !important;
    }

    /* 상단 헤더: 투명 처리 (사이드바 열기 버튼은 유지), 장식선/배포 버튼 숨김 */
    header[data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stDecoration"], [data-testid="stAppDeployButton"] { display: none !important; }

    /* 본문 폭: Gemini처럼 중앙 정렬된 좁은 대화 영역 */
    .block-container {
        max-width: 860px !important;
        padding: 1rem 1rem 2rem !important;
    }
    [data-testid="stBottomBlockContainer"] {
        max-width: 860px !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* 본문 아이콘 버튼 (새 채팅) — 사이드바 규칙이 뒤에 오므로 사이드바 버튼에는 덮어써지지 않음 */
    .block-container .stButton button {
        background: transparent !important;
        border: none !important;
        border-radius: 50% !important;
        width: 40px; height: 40px; min-height: 40px !important;
        padding: 0 !important;
        color: var(--gm-text) !important;
    }
    .block-container .stButton button:hover { background: var(--gm-surface-hover) !important; }

    /* ---------------- 사이드바 (최근 대화 목록) ---------------- */
    [data-testid="stSidebar"] {
        background: var(--gm-surface) !important;
        border-right: none !important;
    }
    [data-testid="stSidebar"] .stButton button {
        justify-content: flex-start !important;
        background: transparent !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 6px 14px !important;
        min-height: 36px !important;
        color: var(--gm-text) !important;
        font-weight: 400 !important;
    }
    [data-testid="stSidebar"] .stButton button > div { justify-content: flex-start !important; min-width: 0; }
    [data-testid="stSidebar"] .stButton button p {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] .stButton button:hover { background: var(--gm-surface-hover) !important; }
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: var(--gm-selected) !important;
        color: #d3e3fd !important;
    }
    .gm-section-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--gm-text);
        padding: 18px 14px 6px;
    }
    .gm-empty-history {
        font-size: 0.85rem;
        color: var(--gm-text-muted);
        padding: 4px 14px;
    }

    /* ---------------- 상단 바 (모델 선택 · 새 채팅) ---------------- */
    .gm-wordmark {
        font-size: 1.35rem;
        font-weight: 400;
        color: var(--gm-text);
        padding-left: 6px;
    }
    [data-testid="stPopover"] button {
        background: transparent !important;
        border: none !important;
        border-radius: 20px !important;
        color: var(--gm-text) !important;
    }
    [data-testid="stPopover"] button:hover { background: var(--gm-surface-hover) !important; }

    /* ---------------- 첫 화면 인사말 ---------------- */
    /* 첫 화면이 실제로 표시 중일 때만 Gemini 시그니처 블루 글로우 배경 적용 */
    .stApp:has([data-testid="stElementContainer"]:not([data-stale="true"]) .gm-hero) {
        background:
            radial-gradient(ellipse 55% 50% at 50% 50%, rgba(36, 62, 150, 0.45), transparent 72%),
            var(--gm-bg) !important;
    }
    .stApp:has([data-testid="stElementContainer"]:not([data-stale="true"]) .gm-hero) [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }
    /* 질문 전송 후 재실행(답변 스트리밍) 동안 Streamlit은 이전 실행의 요소를 data-stale로 흐리게 남겨 둠
       → 첫 화면 인사말 · 중앙 입력창이 대화 화면과 겹쳐 보이지 않도록 숨김 */
    [data-stale="true"]:has(.gm-hero), .st-key-hero_input[data-stale="true"] {
        display: none !important;
    }
    .gm-hero { padding-top: 24vh; }

    /* 답변 대기 중 "생각 중…" 표시: Gemini처럼 글자 위로 빛이 흐르는 shimmer 효과 */
    .gm-thinking {
        display: inline-block;
        font-size: 1rem;
        background: linear-gradient(90deg, var(--gm-text-muted) 0%, #ffffff 50%, var(--gm-text-muted) 100%);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gm-shimmer 1.6s linear infinite;
    }
    @keyframes gm-shimmer {
        from { background-position: 100% 0; }
        to { background-position: -100% 0; }
    }
    .gm-hero-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 400;
        color: var(--gm-text);
        margin: 0 0 1.6rem;
    }

    /* ---------------- 채팅 메시지 ---------------- */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 0.6rem 0 !important;
        gap: 1rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        line-height: 1.75;
        font-size: 1rem;
    }
    /* 사용자 메시지: 오른쪽 정렬 말풍선, 아바타 숨김 */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
    }
    [data-testid="stChatMessageAvatarUser"] { display: none !important; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        flex: 0 1 auto !important;
        margin: 0 0 0 auto !important;
        max-width: 75%;
        background: var(--gm-user-bubble);
        border-radius: 24px 4px 24px 24px;
        padding: 10px 20px;
    }
    /* Streamlit 마크다운의 음수 하단 여백(-1rem)이 말풍선 높이를 깎지 않도록 제거 */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
        margin-bottom: 0 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p { margin: 0; }
    /* AI 메시지: 말풍선 없이 스파클 아이콘 */
    [data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] {
        background: transparent !important;
        border: none !important;
        color: var(--gm-accent) !important;
    }
    [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {
        color: var(--gm-text-muted) !important;
        font-size: 0.78rem;
    }

    /* ---------------- 입력창 (둥근 pill) ---------------- */
    [data-testid="stBottom"], [data-testid="stBottom"] > div { background: var(--gm-bg) !important; }
    [data-testid="stChatInput"] {
        background: var(--gm-surface) !important;
        border: none !important;
        border-radius: 32px !important;
        padding: 6px 8px 6px 14px !important;
    }
    [data-testid="stChatInput"] > div {
        background: transparent !important;
        border-color: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--gm-text) !important;
        font-size: 1rem !important;
    }
    [data-testid="stChatInputSubmitButton"] { border-radius: 50% !important; }
    /* 하단 입력창 아래 안내 문구 */
    [data-testid="stBottomBlockContainer"]::after {
        content: "AI는 실수를 할 수 있으므로 중요한 정보는 다시 한번 확인하세요.";
        display: block;
        text-align: center;
        font-size: 0.75rem;
        color: var(--gm-text-muted);
        padding-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. InMemorySaver (메모리 체크포인터) 싱글톤 인스턴스
# ==============================================================================
@st.cache_resource
def get_checkpointer():
    return InMemorySaver()


checkpointer = get_checkpointer()

# ==============================================================================
# 3. 모델 생성 및 Google 검색 도구 연결 (Grounding / CSE)
# ==============================================================================
MODEL_OPTIONS = {
    "Google Gemini (Vertex AI)": "Gemini Flash",
    "Ollama Gemma (gemma4:e2b)": "Gemma 4 (로컬)",
}

SEARCH_OFF = "off"
SEARCH_GROUNDING = "grounding"
SEARCH_CSE = "cse"
SEARCH_OPTIONS = {
    SEARCH_OFF: "끄기",
    SEARCH_GROUNDING: "Vertex AI Grounding",
    SEARCH_CSE: "Custom Search (CSE)",
}

CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
CSE_ID = os.getenv("GOOGLE_CSE_ID", "").strip()
CSE_CONFIGURED = bool(CSE_API_KEY and CSE_ID)


@tool
def google_custom_search(query: str) -> str:
    """Google 검색으로 최신 웹 정보를 찾습니다. 최신 뉴스, 실시간 데이터, 사실 확인이 필요할 때 검색어를 넣어 호출하세요."""
    params = urllib.parse.urlencode({"cx": CSE_ID, "q": query, "num": 5, "hl": "ko"})
    # API 키는 URL 쿼리 대신 헤더로 전달 (로그·프록시에 키가 남지 않도록)
    req = urllib.request.Request(
        f"https://www.googleapis.com/customsearch/v1?{params}",
        headers={"X-Goog-Api-Key": CSE_API_KEY},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return f"Google Custom Search 요청 실패 (HTTP {e.code}): {e.read().decode('utf-8', errors='replace')[:300]}"
    except urllib.error.URLError as e:
        return f"Google Custom Search 요청 중 오류가 발생했습니다: {e}"

    items = data.get("items", [])
    if not items:
        return f"'{query}'에 대한 검색 결과가 없습니다."
    return "\n\n".join(
        f"[{i}] {item.get('title', '')}\n{item.get('link', '')}\n{item.get('snippet', '')}"
        for i, item in enumerate(items, start=1)
    )


def create_agent_for_model(model_name: str, search_mode: str = SEARCH_OFF):
    """
    선택한 모델에 맞춰 LLM 객체를 생성하고, 검색 방식에 따라 도구를 연결한 에이전트를 빌드합니다.
    - Grounding: Gemini 내장 google_search 도구 (Google 서버에서 검색까지 처리, Gemini 전용)
    - CSE: google_custom_search LangChain 도구 (에이전트가 직접 호출, 도구 호출 지원 모델이면 사용 가능)
    """
    use_grounding = search_mode == SEARCH_GROUNDING and "Gemini" in model_name
    use_cse = search_mode == SEARCH_CSE and CSE_CONFIGURED

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

        # [핵심] Google Search Grounding 도구 연결
        if use_grounding:
            # Vertex AI Gemini의 공식 native 구글 검색 도구 바인딩
            llm = llm.bind_tools([{"google_search": {}}])

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

    # 시스템 프롬프트 구성
    system_prompt = (
        "당신은 Google Gemini 스타일의 유능하고 지적이며 친절한 AI 어시스턴트입니다. "
        "사용자의 이전 대화 맥락을 기억하여 일관성 있고 풍부하게 답변하세요."
    )
    if use_grounding:
        system_prompt += (
            " 실시간 구글 검색(Google Search) 기능이 활성화되어 있으므로, "
            "최신 뉴스, 실시간 데이터, 웹 정보가 필요한 질문에는 검색 결과를 충실히 반영하여 답변하세요."
        )
    elif use_cse:
        system_prompt += (
            " 최신 뉴스, 실시간 데이터, 웹 정보가 필요한 질문에는 google_custom_search 도구로 먼저 검색하고, "
            "검색 결과를 근거로 답변한 뒤 답변 끝에 참고한 출처 링크를 남기세요."
        )

    new_agent = create_agent(
        model=llm,
        # Grounding은 bind_tools로 모델에 내장 검색을 붙이므로 에이전트 도구는 CSE일 때만 등록
        tools=[google_custom_search] if use_cse else [],
        checkpointer=checkpointer,
        system_prompt=system_prompt,
    )
    return new_agent


# ==============================================================================
# 4. 세션 상태 관리 (대화 목록 threads, 현재 thread_id, 대기 중인 질문)
# ==============================================================================
# threads: {thread_id: {"title": 첫 질문 요약, "messages": [UI 메시지...]}}
# 대화 기록 자체는 InMemorySaver가 thread_id별로 보관하므로, thread_id만 바꾸면 대화가 전환됩니다.
if "threads" not in st.session_state:
    st.session_state.threads = {}

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


def start_new_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pending_prompt = None


def select_thread(thread_id: str):
    st.session_state.thread_id = thread_id
    st.session_state.pending_prompt = None


def submit_prompt(prompt: str):
    """질문을 현재 대화에 기록하고, 다음 실행에서 대화 화면으로 전환해 답변을 스트리밍합니다."""
    thread = st.session_state.threads.setdefault(
        st.session_state.thread_id, {"title": prompt[:30], "messages": []}
    )
    thread["messages"].append({"role": "user", "content": prompt})
    st.session_state.pending_prompt = prompt
    st.rerun()


messages = st.session_state.threads.get(st.session_state.thread_id, {}).get("messages", [])
is_empty_chat = len(messages) == 0

# ==============================================================================
# 5. 사이드바: 새 채팅 + 최근 대화 목록 + 설정 및 사용량
# ==============================================================================
with st.sidebar:
    st.button("새 채팅", icon=":material/edit_square:", on_click=start_new_chat, width="stretch")

    st.markdown('<div class="gm-section-label">최근</div>', unsafe_allow_html=True)
    if st.session_state.threads:
        for tid, thread in reversed(list(st.session_state.threads.items())):
            st.button(
                thread["title"],
                key=f"thread_{tid}",
                type="primary" if tid == st.session_state.thread_id else "secondary",
                on_click=select_thread,
                args=(tid,),
                width="stretch",
            )
    else:
        st.markdown('<div class="gm-empty-history">아직 대화가 없습니다</div>', unsafe_allow_html=True)

    st.divider()

    # 세션 누적 사용량 통계 (모든 대화 합산)
    with st.expander("설정 및 사용량", icon=":material/settings:"):
        all_messages = [m for t in st.session_state.threads.values() for m in t["messages"]]
        accum_tokens = sum(m.get("total_tokens", 0) for m in all_messages)
        accum_usd = sum(m.get("cost_usd", 0.0) for m in all_messages)
        st.metric("누적 사용 토큰", f"{accum_tokens:,} tokens")
        if accum_usd > 0:
            st.metric("누적 예상 비용", f"${accum_usd:.5f} (약 {accum_usd * 1380:.2f}원)")
        else:
            st.metric("누적 예상 비용", "$0.00 (무료)")
        st.caption(f"대화 세션 ID: `{st.session_state.thread_id[:8]}...`")

# ==============================================================================
# 6. 상단 바: 워드마크 + 모델 선택 메뉴(모델 · Google 검색) + 새 채팅 아이콘
# ==============================================================================
col_brand, col_model, col_new = st.columns([5, 2, 0.6], vertical_alignment="center")
with col_brand:
    st.markdown('<div class="gm-wordmark">Gemini</div>', unsafe_allow_html=True)

with col_model:
    current_label = MODEL_OPTIONS[st.session_state.get("model_choice", next(iter(MODEL_OPTIONS)))]
    with st.popover(current_label, width="stretch"):
        model_choice = st.radio(
            "모델",
            options=list(MODEL_OPTIONS),
            format_func=lambda key: MODEL_OPTIONS[key],
            captions=["Vertex AI · 빠른 응답", "Ollama · 로컬 실행, 무료"],
            key="model_choice",
        )
        search_mode = st.radio(
            "Google 검색",
            options=list(SEARCH_OPTIONS),
            index=1,
            format_func=lambda key: SEARCH_OPTIONS[key],
            captions=[
                "검색 없이 모델 지식으로만 답변",
                "Gemini 내장 검색 · Google 서버에서 처리",
                "Custom Search JSON API · 에이전트 도구로 호출",
            ],
            key="search_mode",
        )
        if search_mode == SEARCH_GROUNDING and "Gemma" in model_choice:
            st.caption("ℹ️ Vertex AI Grounding은 Gemini 모델에서만 동작합니다. Gemma는 CSE 방식을 사용하세요.")
        if search_mode == SEARCH_CSE and not CSE_CONFIGURED:
            st.caption("⚠️ `GOOGLE_CSE_API_KEY`, `GOOGLE_CSE_ID` 환경변수가 없어 검색 없이 답변합니다.")

# 실제로 검색이 연결되는 조건 (모델 · 환경변수 고려)
search_active = (search_mode == SEARCH_GROUNDING and "Gemini" in model_choice) or (
    search_mode == SEARCH_CSE and CSE_CONFIGURED
)

with col_new:
    st.button(":material/edit_square:", help="새 채팅", on_click=start_new_chat, key="top_new_chat")

# 모델 또는 검색 방식 변경 감지 ➔ 에이전트 새로 생성
agent_state_key = f"{model_choice}_search_{search_mode}"
if st.session_state.get("current_agent_key") != agent_state_key:
    st.session_state.current_agent_key = agent_state_key
    st.session_state.agent = create_agent_for_model(model_choice, search_mode=search_mode)

current_agent = st.session_state.agent

# ==============================================================================
# 7. 화면 렌더링: 첫 화면(중앙 입력창) / 대화 화면(하단 입력창)
# ==============================================================================
ASSISTANT_AVATAR = ":material/auto_awesome:"


def render_meta(msg: dict):
    """AI 답변 하단에 모델 · 검색 여부 · 토큰 · 비용 정보를 작게 표시합니다."""
    searched = msg.get("searched")
    search_info = f" · {searched} 반영" if isinstance(searched, str) and searched else ""
    meta_text = f"{MODEL_OPTIONS.get(msg['model'], msg['model'])}{search_info}"
    if msg.get("total_tokens", 0) > 0:
        meta_text += f" · {msg['total_tokens']:,} 토큰 (입력 {msg['in_tokens']:,} / 출력 {msg['out_tokens']:,})"
    if "cost_str" in msg:
        meta_text += f" · {msg['cost_str']}"
    st.caption(meta_text)


if is_empty_chat:
    # Gemini 첫 화면: 화면 중앙의 인사말 + 입력창 (컨테이너 안의 chat_input은 하단 고정 대신 인라인으로 렌더링됨)
    st.markdown(
        '<div class="gm-hero"><div class="gm-hero-title">무엇을 도와드릴까요?</div></div>',
        unsafe_allow_html=True,
    )
    with st.container():
        prompt = st.chat_input("Gemini에게 물어보기", key="hero_input")
    if prompt:
        submit_prompt(prompt)
    st.stop()

for msg in messages:
    with st.chat_message(msg["role"], avatar=ASSISTANT_AVATAR if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg.get("model"):
            render_meta(msg)

prompt = st.chat_input("Gemini에게 물어보기", key="bottom_input")
if prompt:
    submit_prompt(prompt)

# ==============================================================================
# 8. 대기 중인 질문 처리 및 agent.stream() 실시간 스트리밍
# ==============================================================================
if user_input := st.session_state.pending_prompt:
    st.session_state.pending_prompt = None

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        agent_config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # 첫 토큰이 도착하기 전까지 로딩 문구 표시 (검색 사용 시 문구 구분)
        thinking_label = "검색하는 중…" if search_active else "생각하는 중…"
        thinking = st.empty()
        thinking.markdown(f'<div class="gm-thinking">{thinking_label}</div>', unsafe_allow_html=True)

        def clear_thinking_on_first_token(stream):
            for i, text in enumerate(stream):
                if i == 0:
                    thinking.empty()
                yield text

        def generate_response_stream():
            for chunk, metadata in current_agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=agent_config,
                stream_mode="messages",
            ):
                # CSE 사용 시 검색 결과(ToolMessage)도 스트림에 섞여 나오므로 AI 응답 조각만 출력
                if not isinstance(chunk, AIMessageChunk):
                    continue
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
            # 실시간 스트리밍 렌더링
            response_text = st.write_stream(clear_thinking_on_first_token(generate_response_stream()))
            thinking.empty()  # 응답이 비어 있는 경우에도 로딩 문구 제거

            # 이번 턴(마지막 사용자 질문 이후)의 메시지만 추출
            latest_state = current_agent.get_state(agent_config)
            saved_messages = latest_state.values.get("messages", [])
            last_human_idx = max(
                (i for i, m in enumerate(saved_messages) if isinstance(m, HumanMessage)), default=-1
            )
            turn_messages = saved_messages[last_human_idx + 1:]

            # 토큰 사용량: CSE는 도구 호출 → 최종 답변으로 모델을 여러 번 부르므로 이번 턴의 AI 메시지를 합산
            usages = []
            for m in turn_messages:
                if not isinstance(m, AIMessage):
                    continue
                usage = m.usage_metadata
                if not usage:
                    usage = m.response_metadata.get("usage_metadata") or m.response_metadata.get("token_usage")
                if usage and isinstance(usage, dict):
                    usages.append(usage)

            if usages:
                in_tokens = sum(u.get("input_tokens", 0) for u in usages)
                out_tokens = sum(u.get("output_tokens", 0) for u in usages)
                total_tokens = sum(u.get("total_tokens", u.get("input_tokens", 0) + u.get("output_tokens", 0)) for u in usages)
            else:
                in_tokens = max(1, len(user_input) // 3)
                out_tokens = max(1, len(response_text) // 3)
                total_tokens = in_tokens + out_tokens

            # 예상 비용 계산 (Gemini Flash 공식 요율 vs 로컬 무료)
            if "Gemini" in model_choice:
                cost_usd = (in_tokens * 0.075 / 1_000_000) + (out_tokens * 0.30 / 1_000_000)
                cost_krw = cost_usd * 1380
                cost_str = f"${cost_usd:.6f} (약 {cost_krw:.3f}원)"
            else:
                cost_usd = 0.0
                cost_str = "$0.00 (로컬 무료)"

            # 검색 반영 표시: CSE는 실제 도구 호출 여부로, Grounding은 활성화 여부로 판단
            if search_mode == SEARCH_CSE:
                cse_called = any(
                    isinstance(m, ToolMessage) and m.name == google_custom_search.name for m in turn_messages
                )
                searched = "Custom Search" if cse_called else ""
            else:
                searched = "Google 검색" if search_active else ""

            assistant_msg = {
                "role": "assistant",
                "content": response_text,
                "model": model_choice,
                "searched": searched,
                "in_tokens": in_tokens,
                "out_tokens": out_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "cost_str": cost_str,
            }
            render_meta(assistant_msg)

            # 세션 메시지 저장
            st.session_state.threads[st.session_state.thread_id]["messages"].append(assistant_msg)
        except Exception as e:
            thinking.empty()
            st.error(f"모델 응답 중 오류가 발생했습니다: {e}")
