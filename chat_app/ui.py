"""화면 렌더링: 사이드바 · 상단 바 · 첫 화면 · 메시지 · 스트리밍 답변."""

import re
from dataclasses import dataclass

import streamlit as st
from langchain_core.messages import AIMessage

from chat_app.config import (
    KRW_PER_USD,
    MAX_OUTPUT_TOKENS,
    MODEL_CAPTIONS,
    MODEL_OPTIONS,
    OUTPUT_TOKENS_MAX,
    OUTPUT_TOKENS_MIN,
    SEARCH_CACHE_TTL_SECONDS,
    SEARCH_CAPTIONS,
    SEARCH_GROUNDING,
    SEARCH_OPTIONS,
)
from chat_app.db import get_graph_cache, get_usage_totals, list_threads, save_message
from chat_app.state import (
    cancel_delete_thread,
    confirm_delete_thread,
    rename_thread_from_input,
    request_delete_thread,
    select_thread,
    start_new_chat,
    submit_prompt,
)
from chat_app.structured import (
    DEFAULT_STRUCTURED_EXAMPLE,
    OutputFormat,
    StructuredExampleError,
    parse_output_format,
)
from chat_app.usage import (
    current_turn_messages,
    estimate_cost,
    is_truncated,
    searched_label,
    sum_token_usage,
    used_code_execution,
)

ASSISTANT_AVATAR = ":material/auto_awesome:"
INPUT_PLACEHOLDER = "Gemini에게 물어보기"


# ------------------------------------------------------------------------------
# 사이드바: 새 채팅 + 최근 대화 목록 + 설정 및 사용량
# ------------------------------------------------------------------------------
@st.dialog("대화를 삭제할까요?", on_dismiss=cancel_delete_thread)  # ✕ · Esc로 닫으면 삭제 대기 해제
def _confirm_delete_dialog(thread_id: str, title: str):
    st.write(f"**{title}**")
    st.caption("대화 내용과 에이전트의 기억(체크포인트)이 모두 삭제되며 되돌릴 수 없습니다.")
    col_cancel, col_delete = st.columns(2)
    # st.dialog는 fragment라 안의 버튼은 dialog만 다시 그림 → 처리 후 st.rerun()으로 앱 전체(사이드바 · 본문)를 갱신
    # (on_click 콜백 안에서는 st.rerun()이 동작하지 않으므로 버튼 반환값으로 처리)
    if col_cancel.button("취소", width="stretch", key="delete_cancel"):
        cancel_delete_thread()
        st.rerun()
    if col_delete.button("삭제", type="primary", width="stretch", key="delete_ok"):
        confirm_delete_thread(thread_id)
        st.rerun()


def _render_thread_row(thread: dict):
    """대화 항목 1줄: 제목 버튼(선택) + ⋮ 메뉴(이름 바꾸기 · 삭제)."""
    tid = thread["thread_id"]
    is_current = tid == st.session_state.thread_id
    col_title, col_menu = st.columns([6, 1], vertical_alignment="center", gap=None)
    with col_title:
        st.button(
            thread["title"],
            key=f"thread_{tid}",
            type="primary" if is_current else "secondary",
            on_click=select_thread,
            args=(tid,),
            width="stretch",
        )
    with col_menu, st.popover("", icon=":material/more_vert:", help="대화 메뉴"):
        st.text_input("이름 바꾸기", value=thread["title"], key=f"rename_{tid}", max_chars=100)
        st.button("이름 저장", icon=":material/edit:", key=f"rename_btn_{tid}",
                  on_click=rename_thread_from_input, args=(tid,), width="stretch")
        st.button("삭제", icon=":material/delete:", key=f"delete_btn_{tid}",
                  on_click=request_delete_thread, args=(tid,), width="stretch")


def render_sidebar():
    with st.sidebar:
        st.button("새 채팅", icon=":material/edit_square:", on_click=start_new_chat, width="stretch")

        st.markdown('<div class="gm-section-label">최근</div>', unsafe_allow_html=True)
        threads = list_threads()
        if threads:
            for thread in threads:
                _render_thread_row(thread)
            # 삭제 확인 창 (⋮ 메뉴의 삭제 → 확인 후 실제 삭제)
            pending_delete = st.session_state.confirm_delete_thread
            if pending_delete:
                title = next((t["title"] for t in threads if t["thread_id"] == pending_delete), None)
                if title is None:
                    cancel_delete_thread()
                else:
                    _confirm_delete_dialog(pending_delete, title)
        else:
            st.markdown('<div class="gm-empty-history">아직 대화가 없습니다</div>', unsafe_allow_html=True)

        st.divider()

        # 전체 누적 사용량 통계 (DB에 저장된 모든 대화 합산)
        with st.expander("설정 및 사용량", icon=":material/settings:"):
            totals = get_usage_totals()
            st.metric("누적 사용 토큰", f"{totals['tokens']:,} tokens")
            if totals["cost_usd"] > 0:
                st.metric("누적 예상 비용", f"${totals['cost_usd']:.5f} (약 {totals['cost_usd'] * KRW_PER_USD:.2f}원)")
            else:
                st.metric("누적 예상 비용", "$0.00 (무료)")

            # LangGraph 캐시 (Vertex AI Grounding 답변)
            graph_cache = get_graph_cache()
            if SEARCH_CACHE_TTL_SECONDS > 0:
                st.metric(
                    "검색 캐시",
                    f"{graph_cache.count_valid():,}건",
                    help=f"Grounding 답변 캐시 · 유효 시간 {SEARCH_CACHE_TTL_SECONDS // 60}분 · 같은 맥락의 질문에 재사용",
                )
            else:
                st.metric("검색 캐시", "사용 안 함", help="SEARCH_CACHE_TTL_SECONDS=0")
            st.button("검색 캐시 비우기", icon=":material/delete_sweep:", on_click=graph_cache.clear, width="stretch")

            st.caption(f"대화 세션 ID: `{st.session_state.thread_id[:8]}...`")


# ------------------------------------------------------------------------------
# 상단 바: 워드마크 + 실행 설정 패널 열기/닫기 + 새 채팅 아이콘
# ------------------------------------------------------------------------------
def toggle_settings_panel():
    st.session_state.settings_panel_open = not st.session_state.get("settings_panel_open", True)


def render_top_bar():
    col_brand, col_settings, col_new = st.columns([8, 0.6, 0.6], vertical_alignment="center")
    with col_brand:
        st.markdown('<div class="gm-wordmark">Gemini</div>', unsafe_allow_html=True)
    with col_settings:
        st.button(":material/tune:", help="실행 설정 패널 열기/닫기", on_click=toggle_settings_panel, key="top_settings")
    with col_new:
        st.button(":material/edit_square:", help="새 채팅", on_click=start_new_chat, key="top_new_chat")


# ------------------------------------------------------------------------------
# 오른쪽 실행 설정 패널: 모델 · Google 검색 · 링크 읽기 · 출력 토큰 상한 · 구조화된 출력
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class RunSettings:
    model_choice: str
    search_mode: str
    url_context: bool
    code_execution: bool
    max_output_tokens: int
    output_format: OutputFormat | None

    def agent_key(self) -> str:
        """이 값이 바뀌면 에이전트를 새로 만듭니다."""
        fmt = self.output_format.canonical() if self.output_format else "none"
        return (
            f"{self.model_choice}|{self.search_mode}|url={self.url_context}|code={self.code_execution}"
            f"|max={self.max_output_tokens}|format={fmt}"
        )


def _render_structured_output_settings() -> OutputFormat | None:
    """구조화된 출력: 켜기/끄기 + 원샷 예시(JSON 또는 '라벨 : 값' 텍스트). 켜져 있고 예시가 올바르면 형식을 반환."""
    enabled = st.toggle(
        "구조화된 출력",
        key="structured_enabled",
        help="원샷 예시와 같은 형식으로만 답변합니다. JSON 예시를 넣으면 같은 구조의 JSON으로, "
        "'라벨 : 값' 텍스트를 넣으면 같은 항목의 텍스트로 답합니다. 예시에서 형식을 추론해 강제하고, "
        "예시 자체는 원샷 프롬프트로 함께 전달합니다.",
    )
    example_text = st.text_area(
        "원샷 예시 (텍스트 또는 JSON)",
        value=DEFAULT_STRUCTURED_EXAMPLE,
        height=180,
        key="structured_example_text",
        disabled=not enabled,
        help="텍스트: 한 줄에 하나씩 '라벨 : 예시 값' (값은 비워도 됨)\nJSON: { 로 시작하는 객체",
    )
    if not enabled:
        return None
    try:
        output_format = parse_output_format(example_text)
    except StructuredExampleError as e:
        st.error(f"예시를 확인해 주세요: {e}")
        st.caption("⚠️ 예시가 올바르지 않아 일반 답변으로 응답합니다.")
        return None
    st.caption(f"✅ {output_format.describe()}")
    with st.expander("모델에 강제하는 JSON Schema"):
        st.json(output_format.json_schema(), expanded=True)
    return output_format


def render_settings_panel() -> RunSettings:
    """
    화면 오른쪽에 고정된 실행 설정 패널을 그리고 선택값(RunSettings)을 반환합니다.
    패널을 닫아도 위젯은 그대로 그리고 CSS로만 숨깁니다 (그리지 않으면 Streamlit이 위젯 값을 지워 설정이 초기화됨).
    """
    is_open = st.session_state.get("settings_panel_open", True)
    with st.container(key="settings_panel"):
        st.markdown(
            f'<div class="gm-panel-state {"gm-panel-open" if is_open else "gm-panel-closed"}"></div>',
            unsafe_allow_html=True,
        )
        col_title, col_close = st.columns([5, 1], vertical_alignment="center")
        col_title.markdown('<div class="gm-panel-title">실행 설정</div>', unsafe_allow_html=True)
        col_close.button(":material/close:", help="패널 닫기", on_click=toggle_settings_panel, key="panel_close")

        model_choice = st.radio(
            "모델",
            options=list(MODEL_OPTIONS),
            format_func=lambda key: MODEL_OPTIONS[key],
            captions=MODEL_CAPTIONS,
            key="model_choice",
        )
        is_gemma = "Gemma" in model_choice

        st.divider()
        search_mode = st.radio(
            "Google 검색",
            options=list(SEARCH_OPTIONS),
            index=1,
            format_func=lambda key: SEARCH_OPTIONS[key],
            captions=SEARCH_CAPTIONS,
            key="search_mode",
        )
        url_context = st.toggle(
            "링크 읽기 (URL context)",
            value=True,
            key="url_context_enabled",
            help="메시지에 웹 페이지 링크가 있으면 Gemini가 그 페이지를 직접 읽고 답합니다. (Gemini 내장 도구)",
        )
        code_execution = st.toggle(
            "코드 실행 (Code execution)",
            value=True,
            key="code_execution_enabled",
            help="계산 · 데이터 처리 · 코드 작성 요청에 Gemini가 Python 코드를 직접 실행하고 결과를 확인한 뒤 답합니다. "
            "코드는 Google 샌드박스에서 실행되며 이 컴퓨터에서는 실행되지 않습니다. (Gemini 내장 도구)",
        )
        if is_gemma and (search_mode == SEARCH_GROUNDING or url_context or code_execution):
            st.caption(
                "ℹ️ Google 검색 · 링크 읽기 · 코드 실행은 Gemini 모델에서만 동작합니다. Gemma는 모델 지식으로만 답변합니다."
            )

        st.divider()
        max_output_tokens = st.number_input(
            "출력 토큰 상한",
            min_value=OUTPUT_TOKENS_MIN,
            max_value=OUTPUT_TOKENS_MAX,
            value=MAX_OUTPUT_TOKENS,
            step=512,
            key="max_output_tokens",
            help="답변 1회당 생성할 수 있는 최대 토큰 수입니다. Gemini는 추론(thinking) 토큰도 포함되므로 "
            "너무 낮으면 답변이 짧게 잘릴 수 있습니다.",
        )

        st.divider()
        output_format = _render_structured_output_settings()

    return RunSettings(
        model_choice=model_choice,
        search_mode=search_mode,
        url_context=url_context and not is_gemma,
        code_execution=code_execution and not is_gemma,
        max_output_tokens=max_output_tokens,
        output_format=output_format,
    )


# ------------------------------------------------------------------------------
# 첫 화면 / 대화 화면
# ------------------------------------------------------------------------------
def render_meta(msg: dict):
    """AI 답변 하단에 모델 · 검색 여부 · 토큰 · 비용 정보를 작게 표시합니다."""
    searched = msg.get("searched")
    search_info = f" · {searched} 반영" if isinstance(searched, str) and searched else ""
    meta_text = f"{MODEL_OPTIONS.get(msg['model'], msg['model'])}{search_info}"
    if msg.get("code_executed"):
        meta_text += " · ▶ 코드 실행"
    if msg.get("total_tokens", 0) > 0:
        meta_text += f" · {msg['total_tokens']:,} 토큰 (입력 {msg['in_tokens']:,} / 출력 {msg['out_tokens']:,})"
    if "cost_str" in msg:
        meta_text += f" · {msg['cost_str']}"
    if msg.get("structured"):
        meta_text += " · { } 구조화된 출력" if msg.get("structured_ok") else " · ⚠️ 구조화된 출력: 형식대로 해석되지 않음"
    if msg.get("truncated"):
        meta_text += " · ⚠️ 출력 토큰 상한에 도달해 답변이 잘렸습니다 (모델 메뉴에서 상한을 올릴 수 있음)"
    st.caption(meta_text)


def render_hero():
    """Gemini 첫 화면: 화면 중앙의 인사말 + 입력창 (컨테이너 안의 chat_input은 하단 고정 대신 인라인으로 렌더링됨)."""
    st.markdown(
        '<div class="gm-hero"><div class="gm-hero-title">무엇을 도와드릴까요?</div></div>',
        unsafe_allow_html=True,
    )
    with st.container():
        prompt = st.chat_input(INPUT_PLACEHOLDER, key="hero_input")
    if prompt:
        submit_prompt(prompt)


def render_conversation(messages: list[dict]):
    """저장된 대화를 그리고 하단 고정 입력창을 표시합니다."""
    for msg in messages:
        with st.chat_message(msg["role"], avatar=ASSISTANT_AVATAR if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])
            if msg.get("model"):
                render_meta(msg)

    prompt = st.chat_input(INPUT_PLACEHOLDER, key="bottom_input")
    if prompt:
        submit_prompt(prompt)


# ------------------------------------------------------------------------------
# 대기 중인 질문 처리: agent.stream() 실시간 스트리밍 → 사용량 계산 → DB 저장
# ------------------------------------------------------------------------------
def _code_language(block: dict) -> str:
    # "Language.PYTHON" · "PYTHON" · "python" → "python"
    return str(block.get("language") or "python").split(".")[-1].lower()


def _execution_outcome_label(outcome) -> str:
    # Gemini outcome: 1=OUTCOME_OK, 2=OUTCOME_FAILED, 3=OUTCOME_DEADLINE_EXCEEDED (숫자 또는 enum 이름)
    value = str(outcome).upper()
    if value == "1" or value.endswith("OK"):
        return "✅ 실행 결과"
    if value == "2" or "FAILED" in value:
        return "❌ 실행 오류"
    if value == "3" or "DEADLINE" in value:
        return "⏱ 실행 시간 초과"
    return "실행 결과"


def _code_execution_markdown(block: dict) -> str:
    """Gemini 코드 실행 블록(실행한 코드 / 실행 결과)을 화면용 마크다운으로 변환."""
    if block.get("type") == "executable_code":
        lang = _code_language(block)
        return f"\n\n**▶ 코드 실행** · {lang}\n```{lang}\n{str(block.get('executable_code', '')).rstrip()}\n```\n"
    output = str(block.get("code_execution_result", "")).rstrip() or "(출력 없음)"
    return f"**{_execution_outcome_label(block.get('outcome'))}**\n```text\n{output}\n```\n\n"


def _iter_ai_display(agent, user_input: str, config: dict, answer_parts: list[str]):
    """
    답변을 화면용 조각으로 내보냅니다. 본문 텍스트 + Gemini 코드 실행 블록(실행한 코드 · 실행 결과).
    answer_parts에는 본문 텍스트만 모아 둡니다 (구조화된 출력 해석 · 토큰 추정에 사용, 코드 블록 제외).
    """
    for chunk, _metadata in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode="messages",
    ):
        # AI 응답만 출력 (스트리밍 호출은 AIMessageChunk, 캐시 적중 · 비스트리밍 모델은 완성된 AIMessage를
        # 한 번에 내보냄 — AIMessageChunk는 AIMessage의 하위 클래스)
        if not isinstance(chunk, AIMessage) or not chunk.content:
            continue
        items = [chunk.content] if isinstance(chunk.content, str) else chunk.content
        for item in items:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict) and item.get("type") in ("executable_code", "code_execution_result"):
                yield _code_execution_markdown(item)
                continue
            elif isinstance(item, dict) and "text" in item:
                text = item["text"]
            else:
                continue
            if text:
                answer_parts.append(text)
                yield text


URL_PATTERN = re.compile(r"https?://\S+")


def stream_assistant_reply(agent, user_input: str, settings: RunSettings, search_active: bool):
    thread_id = st.session_state.thread_id
    output_format = settings.output_format
    model_choice = settings.model_choice
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        agent_config = {"configurable": {"thread_id": thread_id}}

        # 첫 토큰이 도착하기 전까지 로딩 문구 표시 (링크 읽기 · 검색 사용 시 문구 구분)
        reads_link = settings.url_context and bool(URL_PATTERN.search(user_input))
        if reads_link:
            thinking_label = "링크를 읽는 중…"
        elif search_active:
            thinking_label = "검색하는 중…"
        else:
            thinking_label = "생각하는 중…"
        thinking = st.empty()
        thinking.markdown(f'<div class="gm-thinking">{thinking_label}</div>', unsafe_allow_html=True)

        def clear_thinking_on_first_token(stream):
            for i, text in enumerate(stream):
                if i == 0:
                    thinking.empty()
                yield text

        try:
            # 답변 자리: 스트리밍 중에는 원문, 구조화된 출력이면 끝난 뒤 예시 형식(JSON 코드 블록 / 라벨 텍스트)으로 교체
            answer_parts: list[str] = []
            answer_slot = st.empty()
            with answer_slot.container():
                streamed = st.write_stream(
                    clear_thinking_on_first_token(_iter_ai_display(agent, user_input, agent_config, answer_parts))
                )
            thinking.empty()  # 응답이 비어 있는 경우에도 로딩 문구 제거
            display_text = streamed if isinstance(streamed, str) else "".join(map(str, streamed))
            response_text = "".join(answer_parts)  # 코드 실행 블록을 뺀 본문

            structured_ok = False
            if output_format is not None:
                # 코드 실행 블록은 그대로 두고, 본문만 예시 형식으로 변환
                rendered, structured_ok = output_format.render(response_text)
                display_text = display_text.removesuffix(response_text) + rendered if structured_ok else display_text
                answer_slot.markdown(display_text)

            turn_messages = current_turn_messages(agent.get_state(agent_config).values.get("messages", []))
            in_tokens, out_tokens, total_tokens = sum_token_usage(turn_messages, user_input, response_text)
            cost_usd, cost_str = estimate_cost(model_choice, in_tokens, out_tokens)

            meta = {
                "model": model_choice,
                "searched": searched_label(search_active, turn_messages, url_read=reads_link),
                "code_executed": used_code_execution(turn_messages),
                "in_tokens": in_tokens,
                "out_tokens": out_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "cost_str": cost_str,
                "truncated": is_truncated(turn_messages),
                "structured": output_format is not None,
                "structured_ok": structured_ok,
            }
            render_meta(meta)

            # 화면용 메시지 DB 저장 (모델 대화 기억은 PostgresSaver가 이미 저장함)
            save_message(thread_id, "assistant", display_text, meta)
        except Exception as e:
            thinking.empty()
            st.error(f"모델 응답 중 오류가 발생했습니다: {e}")
