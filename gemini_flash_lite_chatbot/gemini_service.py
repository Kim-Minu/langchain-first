"""Gemini 3.5 Flash Lite Chatbot - Model Service Module

GCP Vertex AI 인프라 상의 gemini-3.5-flash-lite 모델을 LangChain 환경에서 호출합니다.
"""

from __future__ import annotations

import logging
import time
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
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

from .config import GeminiConfig, default_config

logger = logging.getLogger("GeminiFlashLiteService")


class GeminiFlashLiteService(BaseChatModel):
    """GCP Vertex AI gemini-3.5-flash-lite 전용 LangChain 챗 모델 서비스."""

    model_name: str = Field(default="gemini-3.5-flash-lite", description="모델 식별자")
    project_id: str = Field(default="iceu-songpa05", description="GCP 프로젝트 ID")
    location: str = Field(default="global", description="GCP 리전")
    vertexai: bool = Field(default=True, description="Vertex AI 사용 여부")
    api_key: Optional[str] = Field(default=None, description="Google API Key (선택)")
    temperature: Optional[float] = Field(default=None, description="생성 다양성 (Lite 모델은 고정 샘플링)")
    max_output_tokens: int = Field(default=4096, description="최대 출력 토큰")
    top_p: Optional[float] = Field(default=None, description="Top-P (Lite 모델은 고정 샘플링)")
    reasoning_effort: str = Field(default="low", description="추론 수준 (low, medium, high)")
    timeout: float = Field(default=60.0, description="타임아웃(초)")
    mock_mode: bool = Field(default=False, description="테스트용 오프라인 모드")

    _client: Any = None

    def __init__(
        self,
        config: Optional[GeminiConfig] = None,
        **kwargs: Any,
    ) -> None:
        cfg = config or default_config

        init_kwargs = {
            "model_name": kwargs.get("model_name", cfg.model_name),
            "project_id": kwargs.get("project_id", cfg.project_id),
            "location": kwargs.get("location", cfg.location),
            "vertexai": kwargs.get("vertexai", cfg.vertexai),
            "api_key": kwargs.get("api_key", cfg.api_key),
            "temperature": kwargs.get("temperature", cfg.temperature),
            "max_output_tokens": kwargs.get("max_output_tokens", cfg.max_output_tokens),
            "top_p": kwargs.get("top_p", cfg.top_p),
            "reasoning_effort": kwargs.get("reasoning_effort", cfg.reasoning_effort),
            "timeout": kwargs.get("timeout", cfg.timeout),
            "mock_mode": kwargs.get("mock_mode", cfg.mock_mode),
        }
        super().__init__(**init_kwargs)
        self._init_inner_client()

    def _init_inner_client(self) -> None:
        """ChatGoogleGenerativeAI 내부 클라이언트를 초기화합니다."""
        if self.mock_mode:
            return

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            import google.auth

            client_kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "max_output_tokens": self.max_output_tokens,
                "timeout": self.timeout,
                "streaming": True,
            }

            # Flash Lite 모델은 고정 샘플링(fixed sampling defaults)을 강제하므로
            # temperature와 top_p를 전달하면 UserWarning이 발생합니다.
            # "lite"가 포함되지 않은 일반 모델이거나 명시적으로 설정된 경우에만 추가합니다.
            if "lite" not in self.model_name.lower():
                if self.temperature is not None:
                    client_kwargs["temperature"] = self.temperature
                if self.top_p is not None:
                    client_kwargs["top_p"] = self.top_p

            if self.api_key:
                client_kwargs["google_api_key"] = self.api_key
                client_kwargs["vertexai"] = False
            elif self.vertexai:
                credentials, default_project = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                client_kwargs["vertexai"] = True
                client_kwargs["credentials"] = credentials
                client_kwargs["project"] = self.project_id or default_project
                client_kwargs["location"] = self.location
                if self.reasoning_effort:
                    client_kwargs["reasoning_effort"] = self.reasoning_effort

            self._client = ChatGoogleGenerativeAI(**client_kwargs)
        except Exception as e:
            logger.warning("ChatGoogleGenerativeAI 초기화 실패: %s (mock_mode 사용 가능)", e)
            self._client = None

    @property
    def _llm_type(self) -> str:
        return "gemini-3.5-flash-lite-service"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "project_id": self.project_id,
            "location": self.location,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "mock_mode": self.mock_mode,
        }

    def _mock_response_text(self, messages: Sequence[BaseMessage]) -> str:
        last_msg = messages[-1].content if messages else "안녕하세요"
        return (
            f"[Gemini 3.5 Flash Lite Service (Mock Mode)]\n"
            f"모델: {self.model_name} (GCP Vertex AI {self.location})\n"
            f"질문: '{last_msg}'에 대한 응답입니다. LangChain과 성공적으로 연동되어 동작 중입니다."
        )

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
                response_metadata={"model": self.model_name, "service": "Gemini 3.5 Flash Lite"},
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        res = self._client.invoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=res)])

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

        for chunk in self._client.stream(messages, stop=stop, **kwargs):
            if run_manager and chunk.content:
                run_manager.on_llm_new_token(chunk.content)
            yield ChatGenerationChunk(message=chunk)

