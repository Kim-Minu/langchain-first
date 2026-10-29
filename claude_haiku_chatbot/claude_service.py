"""Claude Haiku 5.5 Chatbot - LangChain Chat Model Service

AnthropicVertex SDK(GCP Vertex AI)를 활용하여 claude-haiku-5-5 모델을 호출하는
LangChain 호환 BaseChatModel 서비스 클래스입니다.
"""

from __future__ import annotations

import logging
import time
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from pydantic import Field

from .config import ClaudeHaikuConfig, default_config

logger = logging.getLogger("ClaudeHaikuVertexChatModel")


def _split_system_and_messages(
    messages: Sequence[BaseMessage],
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """SystemMessage를 추출하고 일반 대화 메시지를 Anthropic 포맷으로 변환합니다."""
    system_parts: List[str] = []
    anthropic_messages: List[Dict[str, str]] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(str(msg.content))
        elif isinstance(msg, HumanMessage):
            anthropic_messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            anthropic_messages.append({"role": "assistant", "content": str(msg.content)})
        else:
            # 기타 메시지 타입은 user 역할로 폴백
            role = "user" if getattr(msg, "role", "user") == "user" else "assistant"
            anthropic_messages.append({"role": role, "content": str(msg.content)})

    system_text = "\n\n".join(system_parts) if system_parts else None

    # Anthropic API는 첫 메시지가 반드시 user여야 함
    if anthropic_messages and anthropic_messages[0]["role"] != "user":
        anthropic_messages.insert(0, {"role": "user", "content": "..."})

    return system_text, anthropic_messages


class ClaudeHaikuVertexChatModel(BaseChatModel):
    """GCP Vertex AI Anthropic Claude Haiku 5.5 전용 LangChain 챗 모델."""

    model_name: str = Field(default="claude-haiku-5-5", description="Anthropic 모델명")
    project_id: str = Field(default="iceu-songpa05", description="GCP 프로젝트 ID")
    region: str = Field(default="global", description="GCP Vertex AI 리전")
    max_tokens: int = Field(default=1024, description="최대 출력 토큰")
    timeout: float = Field(default=60.0, description="타임아웃(초)")
    mock_mode: bool = Field(default=False, description="오프라인 테스트용 모의 응답 모드")

    _client: Any = None

    def __init__(
        self,
        config: Optional[ClaudeHaikuConfig] = None,
        **kwargs: Any,
    ) -> None:
        cfg = config or default_config

        init_kwargs = {
            "model_name": kwargs.get("model_name", cfg.model_name),
            "project_id": kwargs.get("project_id", cfg.project_id),
            "region": kwargs.get("region", cfg.region),
            "max_tokens": kwargs.get("max_tokens", cfg.max_tokens),
            "timeout": kwargs.get("timeout", cfg.timeout),
            "mock_mode": kwargs.get("mock_mode", cfg.mock_mode),
        }
        super().__init__(**init_kwargs)
        self._init_anthropic_client()

    def _init_anthropic_client(self) -> None:
        """AnthropicVertex 클라이언트를 초기화합니다."""
        if self.mock_mode:
            return

        try:
            from anthropic import AnthropicVertex

            self._client = AnthropicVertex(
                region=self.region,
                project_id=self.project_id,
                timeout=self.timeout,
            )
        except Exception as e:
            logger.warning("AnthropicVertex 클라이언트 초기화 실패: %s (mock_mode 사용 권장)", e)
            self._client = None

    @property
    def _llm_type(self) -> str:
        return "claude-haiku-5-5-vertex"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "project_id": self.project_id,
            "region": self.region,
            "max_tokens": self.max_tokens,
            "mock_mode": self.mock_mode,
        }

    def _mock_response_text(self, messages: Sequence[BaseMessage]) -> str:
        last_msg = messages[-1].content if messages else "Hello"
        return (
            f"[Claude Haiku 5.5 Vertex API Service (Mock Mode)]\n"
            f"GCP Vertex AI ({self.region}, {self.project_id})\n"
            f"모델: {self.model_name}\n"
            f"질문하신 내용: '{last_msg}'에 대한 응답입니다. LangChain과 성공적으로 연동되었습니다."
        )

    # ------------------------------------------------------------------
    # 동기 호출 (_generate)
    # ------------------------------------------------------------------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.mock_mode or not self._client:
            text = self._mock_response_text(messages)
            ai_msg = AIMessage(
                content=text,
                response_metadata={"model": self.model_name, "service": "Claude Haiku 5.5 Vertex"},
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        system_text, anthropic_messages = _split_system_and_messages(messages)
        call_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if system_text:
            call_kwargs["system"] = system_text
        if stop:
            call_kwargs["stop_sequences"] = stop

        try:
            response = self._client.messages.create(**call_kwargs)
            # Anthropic response.content: list of TextBlock
            text = "".join(block.text for block in response.content if hasattr(block, "text"))

            ai_msg = AIMessage(
                content=text,
                response_metadata={
                    "model": response.model,
                    "id": response.id,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    "stop_reason": response.stop_reason,
                    "service": "Claude Haiku 5.5 Vertex",
                },
                usage_metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])
        except Exception as e:
            logger.error("Claude Haiku 5.5 API 호출 실패: %s", e)
            raise RuntimeError(f"Claude Haiku 5.5 Vertex API 호출 에러: {e}") from e

    # ------------------------------------------------------------------
    # 스트리밍 호출 (_stream)
    # ------------------------------------------------------------------
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.mock_mode or not self._client:
            mock_text = self._mock_response_text(messages)
            words = mock_text.split(" ")
            for idx, word in enumerate(words):
                chunk_text = word + (" " if idx < len(words) - 1 else "")
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
                if run_manager:
                    run_manager.on_llm_new_token(chunk_text)
                yield chunk
                time.sleep(0.02)
            return

        system_text, anthropic_messages = _split_system_and_messages(messages)
        call_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if system_text:
            call_kwargs["system"] = system_text
        if stop:
            call_kwargs["stop_sequences"] = stop

        with self._client.messages.stream(**call_kwargs) as stream:
            for text_chunk in stream.text_stream:
                if text_chunk:
                    chunk = ChatGenerationChunk(message=AIMessageChunk(content=text_chunk))
                    if run_manager:
                        run_manager.on_llm_new_token(text_chunk)
                    yield chunk

