"""Claude Haiku 5.5 Chatbot - Unit & Integration Tests

Anthropic Vertex AI LangChain 모델 서비스 및 멀티턴 대화 챗봇의 동작을 검증합니다.
"""

from __future__ import annotations

import os
import sys
import unittest

# 프로젝트 및 의존성 경로 등록
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv/lib/python3.14/site-packages")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from claude_haiku_chatbot.config import ClaudeHaikuConfig
from claude_haiku_chatbot.claude_service import ClaudeHaikuVertexChatModel, _split_system_and_messages
from claude_haiku_chatbot.chatbot import ClaudeHaikuChatbot, SimpleConversationHistory


class TestClaudeHaikuVertexService(unittest.TestCase):
    """Claude Haiku 5.5 서비스 단위 테스트."""

    def setUp(self) -> None:
        self.config = ClaudeHaikuConfig(
            model_name="claude-haiku-5-5",
            project_id="test-project",
            region="global",
            max_tokens=512,
            mock_mode=True,
        )
        self.model = ClaudeHaikuVertexChatModel(config=self.config)

    def test_message_splitting(self) -> None:
        """SystemMessage와 대화 메시지가 Anthropic 규격에 맞게 올바르게 분리되는지 검증."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="Tell me about Paris."),
        ]
        system_text, anthropic_msgs = _split_system_and_messages(messages)

        self.assertEqual(system_text, "You are a helpful assistant.")
        self.assertEqual(len(anthropic_msgs), 3)
        self.assertEqual(anthropic_msgs[0], {"role": "user", "content": "Hello!"})
        self.assertEqual(anthropic_msgs[1], {"role": "assistant", "content": "Hi there!"})
        self.assertEqual(anthropic_msgs[2], {"role": "user", "content": "Tell me about Paris."})

    def test_model_invoke_mock(self) -> None:
        """Mock 모드에서 invoke 호출이 정상적으로 AIMessage를 반환하는지 검증."""
        messages = [
            SystemMessage(content="시스템 지침"),
            HumanMessage(content="파리 여행 일정을 추천해줘."),
        ]
        result = self.model.invoke(messages)

        self.assertIsInstance(result, AIMessage)
        self.assertIn("파리", str(result.content))
        self.assertIn("claude-haiku-5-5", str(result.content))

    def test_model_stream_mock(self) -> None:
        """Mock 모드에서 stream 호출 시 토큰 청크들이 순차적으로 산출되는지 검증."""
        messages = [HumanMessage(content="스트리밍 테스트")]
        chunks = list(self.model.stream(messages))

        self.assertGreater(len(chunks), 0)
        full_text = "".join([c.content for c in chunks])
        self.assertIn("claude-haiku-5-5", full_text)

    def test_chatbot_multiturn_flow(self) -> None:
        """Chatbot 엔진의 LCEL 체인 및 멀티턴 대화 메모리 유지 검증."""
        chatbot = ClaudeHaikuChatbot(service=self.model, config=self.config)

        # 첫 번째 턴
        res1 = chatbot.ask("첫 번째 질문입니다.")
        self.assertTrue(len(res1) > 0)
        history = chatbot.get_history_messages()
        self.assertEqual(len(history), 2)  # [Human, AI]

        # 두 번째 턴
        res2 = chatbot.ask("두 번째 질문입니다.")
        self.assertTrue(len(res2) > 0)
        history2 = chatbot.get_history_messages()
        self.assertEqual(len(history2), 4)  # [Human, AI, Human, AI]

        # 히스토리 초기화 검증
        chatbot.clear_history()
        self.assertEqual(len(chatbot.get_history_messages()), 0)

    def test_chatbot_stream_ask(self) -> None:
        """Chatbot 엔진의 stream_ask 스트리밍 출력 검증."""
        chatbot = ClaudeHaikuChatbot(service=self.model, config=self.config)

        streamed_chunks = list(chatbot.stream_ask("실시간 답변 요청"))
        self.assertGreater(len(streamed_chunks), 0)

        # 스트리밍 완료 후 히스토리에 온전히 합쳐져 저장되었는지 확인
        history = chatbot.get_history_messages()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1].content, "".join(streamed_chunks))


if __name__ == "__main__":
    unittest.main()

