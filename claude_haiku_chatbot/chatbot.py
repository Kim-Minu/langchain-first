"""Claude Haiku 5.5 Chatbot - Chatbot Engine

LangChain LCEL 체인과 대화 메모리를 결합한 Anthropic Claude Haiku 5.5 멀티턴 대화형 챗봇 엔진입니다.
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

from .claude_service import ClaudeHaikuVertexChatModel
from .config import ClaudeHaikuConfig, default_config


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


class ClaudeHaikuChatbot:
    """Claude Haiku 5.5 기반 LangChain 멀티턴 챗봇."""

    def __init__(
        self,
        service: Optional[ClaudeHaikuVertexChatModel] = None,
        config: Optional[ClaudeHaikuConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.config = config or default_config
        self.service = service or ClaudeHaikuVertexChatModel(config=self.config)
        self.system_prompt = system_prompt or self.config.default_system_prompt
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
        """대화 히스토리를 초기화합니다."""
        self.history.clear()

    def get_history_messages(self) -> List[BaseMessage]:
        """현재 저장된 대화 히스토리 목록을 반환합니다."""
        return list(self.history.messages)

    def set_system_prompt(self, new_prompt: str) -> None:
        """시스템 프롬프트를 갱신합니다."""
        self.system_prompt = new_prompt

