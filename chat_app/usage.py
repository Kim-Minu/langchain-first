"""턴별 토큰 합산, 예상 비용, 검색 반영 여부 판단 (Streamlit에 의존하지 않는 순수 로직)."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from chat_app.config import (
    GEMINI_INPUT_USD_PER_1M,
    GEMINI_OUTPUT_USD_PER_1M,
    KRW_PER_USD,
    is_gemini,
)


def current_turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """마지막 사용자 질문 이후(이번 턴)의 메시지만 추출합니다."""
    last_human_idx = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    return messages[last_human_idx + 1:]


def sum_token_usage(turn_messages: list[BaseMessage], user_input: str, response_text: str) -> tuple[int, int, int]:
    """
    (입력, 출력, 총) 토큰 수를 반환합니다.
    한 턴에 모델이 여러 번 호출될 수 있으므로 이번 턴의 AI 메시지를 모두 합산합니다.
    Grounding 캐시 적중 답변은 사용량 0으로 기록되어 있어 비용도 0이 됩니다.
    사용량 정보가 없으면 글자 수 기준으로 대략 추정합니다.
    """
    usages = []
    for m in turn_messages:
        if not isinstance(m, AIMessage):
            continue
        usage = m.usage_metadata
        if not usage:
            usage = m.response_metadata.get("usage_metadata") or m.response_metadata.get("token_usage")
        if usage and isinstance(usage, dict):
            usages.append(usage)

    if not usages:
        in_tokens = max(1, len(user_input) // 3)
        out_tokens = max(1, len(response_text) // 3)
        return in_tokens, out_tokens, in_tokens + out_tokens

    in_tokens = sum(u.get("input_tokens", 0) for u in usages)
    out_tokens = sum(u.get("output_tokens", 0) for u in usages)
    total_tokens = sum(u.get("total_tokens", u.get("input_tokens", 0) + u.get("output_tokens", 0)) for u in usages)
    return in_tokens, out_tokens, total_tokens


def estimate_cost(model_name: str, in_tokens: int, out_tokens: int) -> tuple[float, str]:
    """(USD 비용, 화면 표시 문자열)을 반환합니다. Gemini Flash 요율 기준, 로컬 모델은 무료."""
    if not is_gemini(model_name):
        return 0.0, "$0.00 (로컬 무료)"
    cost_usd = (in_tokens * GEMINI_INPUT_USD_PER_1M + out_tokens * GEMINI_OUTPUT_USD_PER_1M) / 1_000_000
    return cost_usd, f"${cost_usd:.6f} (약 {cost_usd * KRW_PER_USD:.3f}원)"


def is_truncated(turn_messages: list[BaseMessage]) -> bool:
    """최종 답변이 출력 토큰 상한에 걸려 잘렸는지 여부 (Gemini: finish_reason, Ollama: done_reason)."""
    final = next((m for m in reversed(turn_messages) if isinstance(m, AIMessage)), None)
    if final is None:
        return False
    meta = final.response_metadata
    return meta.get("finish_reason") == "MAX_TOKENS" or meta.get("done_reason") == "length"


def used_code_execution(turn_messages: list[BaseMessage]) -> bool:
    """이번 턴에 Gemini 코드 실행(executable_code 블록)이 있었는지 여부."""
    return any(
        isinstance(m, AIMessage)
        and isinstance(m.content, list)
        and any(isinstance(b, dict) and b.get("type") == "executable_code" for b in m.content)
        for m in turn_messages
    )


def searched_label(search_active: bool, turn_messages: list[BaseMessage], url_read: bool = False) -> str:
    """
    내장 도구 반영 표시: 링크 읽기(URL context, 질문에 링크가 있을 때) · Google 검색(Grounding 활성화 시).
    캐시 적중 시 "(캐시)"를 덧붙입니다.
    """
    parts = (["링크 읽기"] if url_read else []) + (["Google 검색"] if search_active else [])
    if not parts:
        return ""
    label = " + ".join(parts)
    final = next((m for m in reversed(turn_messages) if isinstance(m, AIMessage)), None)
    if final is not None and final.response_metadata.get("grounding_cache_hit"):
        return f"{label} (캐시)"
    return label
