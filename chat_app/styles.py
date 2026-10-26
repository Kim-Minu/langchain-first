"""Google Gemini 다크 테마 CSS."""

import streamlit as st

GEMINI_CSS = """
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
/* 대화 항목의 ⋮ 메뉴: Gemini처럼 마우스를 올린 항목 · 현재 대화에서만 표시 */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) {
    gap: 0 !important;
    align-items: center;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stPopover"] {
    opacity: 0;
    transition: opacity 0.15s;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover [data-testid="stPopover"],
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button[kind="primary"]) [data-testid="stPopover"],
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stPopover"]:focus-within {
    opacity: 1;
}
[data-testid="stSidebar"] [data-testid="stPopover"] button {
    min-height: 32px !important;
    padding: 0 6px !important;
}
/* 팝오버 기본 펼침 화살표(expand_more) 숨김 → ⋮ 아이콘만 표시 */
[data-testid="stSidebar"] [data-testid="stPopover"] button > div > div:nth-child(2) { display: none; }
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

/* ---------------- 오른쪽 실행 설정 패널 (Google AI Studio의 Run settings 형태) ---------------- */
.st-key-settings_panel {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    width: 340px;
    max-width: 100vw;
    padding: 20px 20px 32px !important;
    background: var(--gm-surface);
    border-left: 1px solid rgba(255, 255, 255, 0.06);
    overflow-y: auto;
    z-index: 999991;  /* Streamlit 헤더(999990) 위 */
}
/* 닫힌 상태: 위젯 값 유지를 위해 그리기는 하되 숨김 */
.st-key-settings_panel:has(.gm-panel-closed) { display: none !important; }
.st-key-settings_panel [data-testid="stElementContainer"]:has(.gm-panel-state) { display: none; }
.gm-panel-title { font-size: 1.05rem; font-weight: 500; color: var(--gm-text); }
.st-key-settings_panel hr { margin: 0.4rem 0 !important; }
/* 패널이 열려 있으면 본문 · 하단 입력창을 패널 폭만큼 비켜 배치 (좁은 화면에서는 패널이 위에 겹침) */
@media (min-width: 900px) {
    /* 하단 입력창(stBottom)은 .stMain 안에 있어 함께 비켜나므로 따로 지정하지 않음 */
    .stApp:has(.gm-panel-open) .stMain { padding-right: 340px; }
}

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
"""


def inject_styles():
    st.markdown(GEMINI_CSS, unsafe_allow_html=True)
