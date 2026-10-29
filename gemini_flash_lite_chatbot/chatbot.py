"""Gemini 3.5 Flash Lite Chatbot - Chatbot Engine

LangChain LCEL 체인과 대화 메모리를 결합한 멀티턴 대화형 챗봇 엔진입니다.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import GeminiConfig, default_config
from .gemini_service import GeminiFlashLiteService

DEFAULT_SYSTEM_PROMPT = (
    "당신은 Google Cloud Platform(GCP) Vertex AI에서 구동되는 "
    "Gemini 3.5 Flash Lite 기반의 지능형 AI 어시스턴트입니다. "
    "사용자의 질문에 대해 빠르고 정확하며 정중한 한국어로 답변해 주세요."
)


class SimpleConversationHistory:
    """대화 내역을 메모리에 안전하게 보관하는 클래스."""

    def __init__(self) -> None:
        self.messages: List[BaseMessage] = []

    def add_user_message(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))

    def add_ai_message(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))

    def clear(self) -> None:
        self.messages.clear()


class GeminiFlashLiteChatbot:
    """Gemini 3.5 Flash Lite 기반 LangChain 챗봇."""

    def __init__(
        self,
        service: Optional[GeminiFlashLiteService] = None,
        config: Optional[GeminiConfig] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.config = config or default_config
        self.service = service or GeminiFlashLiteService(config=self.config)
        self.system_prompt = system_prompt
        self.history = SimpleConversationHistory()

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        self.output_parser = StrOutputParser()
        self.chain = self.prompt_template | self.service | self.output_parser

    def ask(self, user_input: str) -> str:
        """사용자 입력을 동기식으로 호출하고 대화 기록을 갱신합니다."""
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

        for chunk in self.chain.stream(
            {
                "system_prompt": self.system_prompt,
                "history": self.history.messages,
                "input": user_input,
            }
        ):
            full_response.append(chunk)
            yield chunk

        complete_text = "".join(full_response)
        self.history.add_user_message(user_input)
        self.history.add_ai_message(complete_text)

    def clear_history(self) -> None:
        self.history.clear()

    def get_history_messages(self) -> List[BaseMessage]:
        return list(self.history.messages)

