"""Model Selector Chatbot - Multi-Model LangChain Chatbot Engine

다양한 모델(DeepSeek V4.1 Flash, Gemma 4 26B 등)을 실시간으로 선택 및 교체할 수 있고,
대화 히스토리를 보존하면서 모델별 맞춤 옵션(Thinking 모드 등)을 지원하는 챗봇 엔진입니다.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import (
    ModelConfig,
    default_config,
    SUPPORTED_MODELS,
    get_default_system_prompt,
)
from .model_service import GcpMaasChatModel

DEFAULT_SYSTEM_PROMPT = get_default_system_prompt("deepseek-flash")


class SimpleConversationHistory:
    """메시지 내역을 보존하고 관리하는 대화 메모리."""

    def __init__(self) -> None:
        self.messages: List[BaseMessage] = []

    def add_user_message(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))

    def add_ai_message(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))

    def clear(self) -> None:
        self.messages.clear()


class ModelSelectorChatbot:
    """모델 선택 및 동적 전환이 가능한 LangChain 대화형 챗봇."""

    def __init__(
        self,
        service: Optional[GcpMaasChatModel] = None,
        config: Optional[ModelConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.config = config or default_config
        self.service = service or GcpMaasChatModel(config=self.config)
        self.system_prompt = system_prompt or self.config.system_prompt or get_default_system_prompt(self.config.model_key)
        self._custom_prompt_set = system_prompt is not None
        self.history = SimpleConversationHistory()

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        self.output_parser = StrOutputParser()
        self._rebuild_chain()

    def set_system_prompt(self, new_prompt: str) -> None:
        """시스템 프롬프트를 갱신합니다."""
        self.system_prompt = new_prompt.strip()
        self._custom_prompt_set = True

    def _rebuild_chain(self) -> None:
        """모델 서비스가 변경되었을 때 LCEL 체인을 재구성합니다."""
        self.chain = self.prompt_template | self.service | self.output_parser

    def switch_model(
        self,
        model_key_or_id: str,
        enable_thinking: Optional[bool] = None,
        **overrides,
    ) -> None:
        """대화 내역을 유지한 채 챗봇의 백엔드 모델을 동적으로 전환합니다."""
        if model_key_or_id in SUPPORTED_MODELS:
            key = model_key_or_id
        else:
            matched = [k for k, v in SUPPORTED_MODELS.items() if v["id"] == model_key_or_id]
            key = matched[0] if matched else "deepseek-flash"

        new_config = ModelConfig.from_model_key(
            key,
            project_id=self.config.project_id,
            endpoint_host=self.config.endpoint_host,
            location=self.config.location,
            mock_mode=self.config.mock_mode,
            **overrides,
        )

        if enable_thinking is not None:
            new_config.enable_thinking = enable_thinking

        self.config = new_config
        self.service = GcpMaasChatModel(config=new_config)

        # 사용자가 고유 커스텀 프롬프트를 직접 지정하지 않은 경우 모델에 맞게 자동 갱신
        if not self._custom_prompt_set:
            self.system_prompt = self.config.system_prompt or get_default_system_prompt(self.config.model_key)

        self._rebuild_chain()

    def ask(self, user_input: str) -> str:
        """사용자 입력을 동기식으로 호출하여 답변을 반환하고 대화 기록을 갱신합니다."""
        response_text = self.chain.invoke(
            {
                "system_prompt": self.system_prompt,
                "history": self.history.messages,
                "input": user_input,
            }
        )
        self.history.add_user_message(user_input)
        self.history.add_ai_message(response_text)
        return response_text

    def stream_ask(self, user_input: str) -> Iterator[str]:
        """사용자 입력을 받아 스트리밍 토큰을 실시간으로 산출합니다."""
        full_response: List[str] = []

        stream_iter = self.chain.stream(
            {
                "system_prompt": self.system_prompt,
                "history": self.history.messages,
                "input": user_input,
            }
        )

        for chunk in stream_iter:
            full_response.append(chunk)
            yield chunk

        complete_reply = "".join(full_response)
        self.history.add_user_message(user_input)
        self.history.add_ai_message(complete_reply)

    def clear_history(self) -> None:
        self.history.clear()

    def get_history_messages(self) -> List[BaseMessage]:
        return list(self.history.messages)


# 이전 호환성 alias
DeepSeekChatbot = ModelSelectorChatbot
