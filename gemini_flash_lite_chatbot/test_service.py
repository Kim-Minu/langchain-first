"""Gemini 3.5 Flash Lite Chatbot - LangChain Test Suite

gemini-3.5-flash-lite 모델과 LangChain 프레임워크 간의 연동 기능을 검증합니다.
"""

import sys
import os

# 모듈 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv/lib/python3.14/site-packages")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from gemini_flash_lite_chatbot.config import GeminiConfig
from gemini_flash_lite_chatbot.gemini_service import GeminiFlashLiteService
from gemini_flash_lite_chatbot.chatbot import GeminiFlashLiteChatbot


def test_single_invoke(mock: bool = True):
    print("\n" + "=" * 50)
    print(" [1. 단일 모델 invoke 테스트]")
    print("=" * 50)

    cfg = GeminiConfig(mock_mode=mock)
    service = GeminiFlashLiteService(config=cfg)

    messages = [
        SystemMessage(content="당신은 Gemini 3.5 Flash Lite 도우미입니다."),
        HumanMessage(content="Flash Lite 모델의 핵심 특징은?"),
    ]

    response = service.invoke(messages)
    print("응답 내용:\n", response.content)
    assert response.content, "응답이 비어있습니다."
    print("✅ [성공] 단일 호출 테스트 통과")


def test_streaming(mock: bool = True):
    print("\n" + "=" * 50)
    print(" [2. 실시간 토큰 스트리밍(stream) 테스트]")
    print("=" * 50)

    cfg = GeminiConfig(mock_mode=mock)
    service = GeminiFlashLiteService(config=cfg)

    messages = [HumanMessage(content="Gemini 3.5 Flash Lite로 여행 일정 추천해줘.")]

    print("스트리밍 출력: ", end="", flush=True)
    chunks = list(service.stream(messages))
    for c in chunks:
        print(c.content, end="", flush=True)
    print()

    assert len(chunks) > 0, "스트리밍 청크를 수신하지 못했습니다."
    print(f"✅ [성공] 스트리밍 테스트 통과 (수신 청크 수: {len(chunks)})")


def test_lcel_chain(mock: bool = True):
    print("\n" + "=" * 50)
    print(" [3. LangChain LCEL 파이프라인 연동 테스트]")
    print("=" * 50)

    cfg = GeminiConfig(mock_mode=mock)
    service = GeminiFlashLiteService(config=cfg)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "공식 서비스 명칭: {service_name}"),
        ("human", "주제: {topic}"),
    ])
    parser = StrOutputParser()

    chain = prompt | service | parser

    result = chain.invoke({
        "service_name": cfg.SERVICE_NAME,
        "topic": "실시간 AI 에이전트 설계",
    })
    print("LCEL 결과:\n", result)
    assert result, "LCEL 체인 결과가 비어있습니다."
    print("✅ [성공] LCEL 체인 연동 테스트 통과")


def test_multiturn_chatbot(mock: bool = True):
    print("\n" + "=" * 50)
    print(" [4. GeminiFlashLiteChatbot 멀티턴 대화 및 메모리 테스트]")
    print("=" * 50)

    cfg = GeminiConfig(mock_mode=mock)
    bot = GeminiFlashLiteChatbot(config=cfg)

    print("1턴 질의: '내 취미는 사진 촬영이야.'")
    reply1 = bot.ask("내 취미는 사진 촬영이야.")
    print("1턴 응답:", reply1)

    print("\n2턴 질의: '내 취미가 뭐라고 했지?'")
    reply2 = bot.ask("내 취미가 뭐라고 했지?")
    print("2턴 응답:", reply2)

    history = bot.get_history_messages()
    print("\n저장된 대화 기록 수:", len(history))
    assert len(history) == 4, f"예상 메시지 수 4개, 실제: {len(history)}"
    print("✅ [성공] 멀티턴 대화 및 메모리 테스트 통과")


def main():
    print("🚀 Gemini 3.5 Flash Lite Chatbot - LangChain 전체 테스트 시작")
    test_single_invoke(mock=True)
    test_streaming(mock=True)
    test_lcel_chain(mock=True)
    test_multiturn_chatbot(mock=True)
    print("\n🎉 모든 Gemini 3.5 Flash Lite 연동 테스트가 성공적으로 통과되었습니다!")


if __name__ == "__main__":
    main()

