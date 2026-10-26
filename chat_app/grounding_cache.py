"""Vertex AI Grounding 답변 캐시 미들웨어 (LangGraph BaseCache 사용).

Grounding은 Google 서버에서 검색 + 답변 생성이 한 번에 끝나므로 검색 결과만 따로 캐시할 수 없습니다.
그래서 "Grounding이 적용된 모델 호출"을 캐시 단위로 삼아, wrap_model_call 미들웨어에서
LangGraph 그래프 캐시(BaseCache → PostgresCache)를 조회/저장합니다.

- 캐시 키: 모델 · 출력 토큰 상한 · 시스템 프롬프트 · 지금까지의 대화 전체
  (맥락이 완전히 같을 때만 재사용 → 다른 대화의 "거기 날씨는?" 같은 후속 질문에 엉뚱한 답이 나가지 않음)
- 캐시 미스: 모델 노드 안에서 그대로 호출하므로 토큰 스트리밍이 유지됨
  (@task 안에서 모델을 호출하면 스트리밍 콜백이 이어지지 않아 답변이 한 번에 출력됨)
- 저장 제외: 잘린 답변(MAX_TOKENS), 빈 답변, 도구 호출 메시지, 오류(예외)
"""

import hashlib
import json
import time
from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage
from langgraph.cache.base import BaseCache, Namespace

CACHE_NAMESPACE = Namespace(("grounding_answer",))


class GroundingCacheMiddleware(AgentMiddleware):
    def __init__(self, cache: BaseCache, ttl_seconds: int, model_id: str, max_output_tokens: int):
        super().__init__()
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens

    def _cache_key(self, request: ModelRequest) -> str:
        payload = {
            "model": self.model_id,
            "max_output_tokens": self.max_output_tokens,
            "system": request.system_message.text if request.system_message else "",
            "messages": [(m.type, m.text) for m in request.messages],
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()

    @staticmethod
    def _cacheable(message: AIMessage) -> bool:
        return (
            bool(message.text)
            and not message.tool_calls
            and message.response_metadata.get("finish_reason") != "MAX_TOKENS"
        )

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse | AIMessage:
        full_key = (CACHE_NAMESPACE, self._cache_key(request))

        if cached := self.cache.get([full_key]).get(full_key):
            message: AIMessage = cached["message"]
            # 캐시 적중: 모델을 호출하지 않으므로 이번 사용량은 0, 메시지 id는 새로 발급
            # (같은 id가 다른 대화 상태에 들어가 add_messages가 덮어쓰는 일을 방지)
            return message.model_copy(
                update={
                    "id": None,
                    "usage_metadata": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "response_metadata": {
                        **message.response_metadata,
                        "grounding_cache_hit": True,
                        "grounding_cached_at": cached["fetched_at"],
                    },
                }
            )

        response = handler(request)
        final = response.result[-1] if response.result else None
        if isinstance(final, AIMessage) and self._cacheable(final):
            self.cache.set({full_key: ({"message": final, "fetched_at": time.time()}, self.ttl_seconds)})
        return response
