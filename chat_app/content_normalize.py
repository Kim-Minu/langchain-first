"""모델 응답 content 블록의 SDK enum 값을 문자열로 바꾸는 미들웨어.

Gemini 코드 실행 블록은 language(Language.PYTHON) · outcome 같은 google.genai enum 객체를 그대로 담고 있습니다.
이대로 체크포인트 · 캐시에 저장하면 LangGraph가 "등록되지 않은 타입" 경고를 내고, 이후 버전에서는
역직렬화가 차단되어 코드 실행이 들어간 대화를 다시 불러올 수 없게 됩니다.
저장되기 전에 enum을 이름 문자열("PYTHON", "OUTCOME_OK")로 바꿔 둡니다.
"""

from collections.abc import Callable
from enum import Enum

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage


def _normalize_value(value):
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    return value


def normalize_message(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, AIMessage) or not isinstance(message.content, list):
        return message
    normalized = _normalize_value(message.content)
    if normalized == message.content and not any(
        isinstance(v, Enum) for block in message.content if isinstance(block, dict) for v in block.values()
    ):
        return message
    return message.model_copy(update={"content": normalized})


class NormalizeContentMiddleware(AgentMiddleware):
    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        response = handler(request)
        response.result = [normalize_message(m) for m in response.result]
        return response
