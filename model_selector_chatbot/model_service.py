"""Model Selector Chatbot - GCP MaaS Chat Model Service

GCP Vertex AI MaaS (Model-as-a-Service) 환경에서
다양한 모델(DeepSeek V4.1 Flash, Gemma 4 26B A4B IT 등)을 선택적으로 호출할 수 있는
LangChain BaseChatModel 기반 서비스 클래스입니다.

지원 사양:
1. DeepSeek V4.1 Flash:
   Model: deepseek-ai/deepseek-v4.1-flash-maas
2. Gemma 4 26B A4B IT:
   Model: google/gemma-4-26b-a4b-it-maas
   chat_template_kwargs: {"enable_thinking": true}
"""

from __future__ import annotations

import json
import logging
import os
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

import httpx
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from pydantic import Field

from .config import ModelConfig, default_config, SUPPORTED_MODELS

logger = logging.getLogger("GcpMaasChatModel")


def _convert_message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """LangChain BaseMessage 객체를 OpenAI / GCP Vertex AI 규격의 dict로 변환합니다."""
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        msg_dict: Dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.additional_kwargs.get("tool_calls"):
            msg_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
        return msg_dict
    elif isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    elif isinstance(message, FunctionMessage):
        return {"role": "function", "name": message.name, "content": message.content}
    elif isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    else:
        return {"role": "user", "content": str(message.content)}


class GcpMaasChatModel(BaseChatModel):
    """GCP Vertex AI MaaS 모델 선택 가능 LangChain 챗 모델.
    
    DeepSeek V4.1 Flash, Gemma 4 26B A4B IT 등 다양한 MaaS 모델을 지원합니다.
    """

    model_name: str = Field(
        default="deepseek-ai/deepseek-v4.1-flash-maas",
        description="GCP MaaS 모델 식별자 (예: deepseek-ai/deepseek-v4.1-flash-maas, google/gemma-4-26b-a4b-it-maas)",
    )
    service_name: str = Field(
        default="GCP MaaS Model Service",
        description="모델 서비스 표시 명칭",
    )
    endpoint_host: str = Field(
        default="aiplatform.googleapis.com",
        description="GCP 엔드포인트 호스트 (${ENDPOINT})",
    )
    project_id: str = Field(default="", description="GCP 프로젝트 ID (${PROJECT_ID})")
    location: str = Field(default="global", description="GCP 리전 (${REGION})")
    api_key: Optional[str] = Field(default=None, description="선택적 직접 API 키")
    endpoint_url: Optional[str] = Field(default=None, description="직접 지정한 전체 엔드포인트 URL")
    temperature: float = Field(default=0.7, description="생성 다양성 온도")
    max_tokens: int = Field(default=4096, description="최대 생성 토큰 수")
    top_p: float = Field(default=0.95, description="Top-P 샘플링")
    timeout: float = Field(default=120.0, description="요청 타임아웃(초)")
    enable_thinking: bool = Field(default=False, description="추론(Thinking) 템플릿 활성화 여부 (Gemma 4 등)")
    mock_mode: bool = Field(default=False, description="테스트/오프라인 모의 응답 모드")

    _cached_token: Optional[str] = None
    _token_expiry: float = 0.0

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """모델 서비스 초기화."""
        cfg = config or default_config

        init_kwargs = {
            "model_name": kwargs.get("model_name", kwargs.get("model", cfg.model_id)),
            "service_name": kwargs.get("service_name", cfg.service_name),
            "endpoint_host": kwargs.get("endpoint_host", cfg.endpoint_host),
            "project_id": kwargs.get("project_id", cfg.project_id),
            "location": kwargs.get("location", cfg.location),
            "api_key": kwargs.get("api_key", cfg.api_key),
            "endpoint_url": kwargs.get("endpoint_url", cfg.custom_endpoint_url),
            "temperature": kwargs.get("temperature", cfg.temperature),
            "max_tokens": kwargs.get("max_tokens", cfg.max_tokens),
            "top_p": kwargs.get("top_p", cfg.top_p),
            "timeout": kwargs.get("timeout", cfg.timeout),
            "enable_thinking": kwargs.get("enable_thinking", cfg.enable_thinking),
            "mock_mode": kwargs.get("mock_mode", cfg.mock_mode),
        }
        super().__init__(**init_kwargs)

    @property
    def _llm_type(self) -> str:
        """LangChain LLM 유형 식별자."""
        return "gcp-maas-chat-model"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """모델 파라미터 매핑 정보."""
        return {
            "service_name": self.service_name,
            "model_name": self.model_name,
            "endpoint_host": self.endpoint_host,
            "project_id": self.project_id,
            "location": self.location,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "enable_thinking": self.enable_thinking,
            "mock_mode": self.mock_mode,
        }

    def _get_target_endpoint(self) -> str:
        """최종 호출 대상 Chat Completions URL을 계산합니다.
        
        규격: https://${ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/openapi/chat/completions
        """
        if self.endpoint_url:
            return self.endpoint_url

        host = self.endpoint_host.strip()
        if host.startswith("https://"):
            host = host[8:]
        elif host.startswith("http://"):
            host = host[7:]
        host = host.rstrip("/")

        return (
            f"https://{host}/v1/"
            f"projects/{self.project_id}/locations/{self.location}/endpoints/openapi/chat/completions"
        )

    def _get_auth_headers(self) -> Dict[str, str]:
        """GCP ADC Bearer 토큰 ($(gcloud auth print-access-token)) 기반 인증 헤더를 반환합니다."""
        headers = {"Content-Type": "application/json"}

        # 1. 수동 지정된 API 키가 있을 경우
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            return headers

        # 2. GCP ADC(Application Default Credentials)를 통해 Access Token 취득
        try:
            import google.auth
            from google.auth.transport.requests import Request

            now = time.time()
            if self._cached_token and now < self._token_expiry - 60:
                headers["Authorization"] = f"Bearer {self._cached_token}"
                return headers

            credentials, default_project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            credentials.refresh(Request())
            token = credentials.token

            if token:
                self._cached_token = token
                self._token_expiry = now + 3000
                headers["Authorization"] = f"Bearer {token}"
                return headers
        except Exception as e:
            logger.warning(
                "GCP 인증 토큰을 자동으로 획득하지 못했습니다: %s. "
                "(gcloud auth application-default login 확인 또는 mock_mode 사용 권장)",
                e,
            )

        return headers

    def _build_payload(
        self,
        messages: Sequence[BaseMessage],
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """OpenAI/Vertex AI 규격 요청 JSON 페이로드를 생성합니다.
        
        Gemma 4 등 enable_thinking이 요청된 경우 chat_template_kwargs를 추가합니다.
        """
        openai_messages = [_convert_message_to_dict(m) for m in messages]
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": stream,
        }

        # Thinking(추론) 파라미터 처리 (Gemma 4 사양: {"chat_template_kwargs": {"enable_thinking": true}})
        enable_think = kwargs.get("enable_thinking", self.enable_thinking)
        if enable_think:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        elif "chat_template_kwargs" in kwargs:
            payload["chat_template_kwargs"] = kwargs["chat_template_kwargs"]

        if "stop" in kwargs and kwargs["stop"]:
            payload["stop"] = kwargs["stop"]

        return payload

    def _mock_response_text(self, messages: Sequence[BaseMessage]) -> str:
        """모의(Mock) 모드에서 생성할 테스트 응답."""
        last_msg = messages[-1].content if messages else "안녕하세요"
        thinking_badge = " [Thinking 모드 활성화됨]" if self.enable_thinking else ""
        return (
            f"[{self.service_name} (Mock Mode){thinking_badge}]\n"
            f"GCP Vertex AI MaaS ({self.endpoint_host}, {self.location})\n"
            f"선택된 모델: {self.model_name}\n"
            f"질문하신 내용: '{last_msg}'에 대한 응답이 성공적으로 LangChain을 통해 처리되었습니다."
        )

    # ------------------------------------------------------------------
    # 동기 실행 (_generate)
    # ------------------------------------------------------------------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.mock_mode:
            text = self._mock_response_text(messages)
            ai_message = AIMessage(
                content=text,
                response_metadata={"model": self.model_name, "service": self.service_name},
            )
            return ChatResult(generations=[ChatGeneration(message=ai_message)])

        url = self._get_target_endpoint()
        headers = self._get_auth_headers()
        payload = self._build_payload(messages, stream=False, stop=stop, **kwargs)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            choice = data.get("choices", [{}])[0]
            message_data = choice.get("message", {})
            content = message_data.get("content", "")

            usage = data.get("usage", {})
            ai_message = AIMessage(
                content=content,
                response_metadata={
                    "model": data.get("model", self.model_name),
                    "usage": usage,
                    "finish_reason": choice.get("finish_reason"),
                    "service": self.service_name,
                },
                usage_metadata={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            )
            return ChatResult(generations=[ChatGeneration(message=ai_message)])
        except Exception as e:
            logger.error("GCP MaaS API 호출 실패 (URL: %s, Model: %s): %s", url, self.model_name, e)
            raise RuntimeError(f"GCP MaaS API 호출 에러 (Model: {self.model_name}, URL: {url}): {e}") from e

    # ------------------------------------------------------------------
    # 스트리밍 실행 (_stream)
    # ------------------------------------------------------------------
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.mock_mode:
            mock_text = self._mock_response_text(messages)
            words = mock_text.split(" ")
            for idx, word in enumerate(words):
                chunk_text = word + (" " if idx < len(words) - 1 else "")
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=chunk_text),
                )
                if run_manager:
                    run_manager.on_llm_new_token(chunk_text)
                yield chunk
                time.sleep(0.02)
            return

        url = self._get_target_endpoint()
        headers = self._get_auth_headers()
        payload = self._build_payload(messages, stream=True, stop=stop, **kwargs)

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line_str = line.strip()
                    if line_str.startswith("data: "):
                        data_part = line_str[6:]
                        if data_part == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data_part)
                            choices = parsed.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content_delta = delta.get("content", "")
                            if content_delta:
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(content=content_delta)
                                )
                                if run_manager:
                                    run_manager.on_llm_new_token(content_delta)
                                yield chunk
                        except json.JSONDecodeError:
                            continue

    # ------------------------------------------------------------------
    # 비동기 실행 (_agenerate)
    # ------------------------------------------------------------------
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.mock_mode:
            return self._generate(messages, stop=stop, **kwargs)

        url = self._get_target_endpoint()
        headers = self._get_auth_headers()
        payload = self._build_payload(messages, stream=False, stop=stop, **kwargs)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        ai_message = AIMessage(
            content=content,
            response_metadata={
                "model": data.get("model", self.model_name),
                "usage": usage,
                "finish_reason": choice.get("finish_reason"),
                "service": self.service_name,
            },
            usage_metadata={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    # ------------------------------------------------------------------
    # 비동기 스트리밍 (_astream)
    # ------------------------------------------------------------------
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.mock_mode:
            for chunk in self._stream(messages, stop=stop, **kwargs):
                yield chunk
            return

        url = self._get_target_endpoint()
        headers = self._get_auth_headers()
        payload = self._build_payload(messages, stream=True, stop=stop, **kwargs)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    line_str = line.strip()
                    if line_str.startswith("data: "):
                        data_part = line_str[6:]
                        if data_part == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data_part)
                            choices = parsed.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content_delta = delta.get("content", "")
                            if content_delta:
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(content=content_delta)
                                )
                                if run_manager:
                                    await run_manager.on_llm_new_token(content_delta)
                                yield chunk
                        except json.JSONDecodeError:
                            continue


# 공식 명칭 하위 호환성을 위한 alias
class DeepSeekV4FlashApiService(GcpMaasChatModel):
    """DeepSeek V4.1 Flash API Service (공식 명칭 호환 클래스)."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model_name", "deepseek-ai/deepseek-v4.1-flash-maas")
        kwargs.setdefault("service_name", "DeepSeek V4.1 Flash API Service")
        super().__init__(**kwargs)

