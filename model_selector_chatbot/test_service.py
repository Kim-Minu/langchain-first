"""Model Selector Chatbot - LangChain Multi-Model Test Suite

GCP curl 사양 검증:
1. DeepSeek V4.1 Flash:
   POST https://${ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/openapi/chat/completions
   Model: deepseek-ai/deepseek-v4.1-flash-maas

2. Gemma 4 26B A4B IT:
   POST https://${ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/openapi/chat/completions
   Model: google/gemma-4-26b-a4b-it-maas
   chat_template_kwargs: {"enable_thinking": true}
   max_tokens: 128000
"""

import sys
import os

# site-packages 및 모듈 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv/lib/python3.14/site-packages")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from model_selector_chatbot.config import ModelConfig, SUPPORTED_MODELS
from model_selector_chatbot.model_service import GcpMaasChatModel
from model_selector_chatbot.chatbot import ModelSelectorChatbot


def test_deepseek_specs():
    print("\n" + "=" * 55)
    print(" [1. DeepSeek V4.1 Flash 엔드포인트 & 페이로드 검증]")
    print("=" * 55)

    cfg = ModelConfig.from_model_key("deepseek-flash", project_id="iceu-songpa05", mock_mode=True)
    service = GcpMaasChatModel(config=cfg)

    url = service._get_target_endpoint()
    payload = service._build_payload([HumanMessage(content="Summer travel plan to Paris")])

    print("호출 URL   :", url)
    print("요청 모델  :", payload["model"])
    print("Max Tokens :", payload["max_tokens"])

    assert url == "https://aiplatform.googleapis.com/v1/projects/iceu-songpa05/locations/global/endpoints/openapi/chat/completions"
    assert payload["model"] == "deepseek-ai/deepseek-v4.1-flash-maas"
    assert payload["max_tokens"] == 4096

    # 스트리밍 실행 검증
    chunks = list(service.stream([HumanMessage(content="Summer travel plan to Paris")]))
    print(f"스트리밍 청크 수: {len(chunks)}")
    assert len(chunks) > 0
    print("✅ [성공] DeepSeek V4.1 Flash 테스트 통과")


def test_gemma4_thinking_specs():
    print("\n" + "=" * 55)
    print(" [2. Gemma 4 26B A4B IT (enable_thinking) 검증]")
    print("=" * 55)

    cfg = ModelConfig.from_model_key(
        "gemma-4-26b",
        project_id="iceu-songpa05",
        enable_thinking=True,
        mock_mode=True,
    )
    service = GcpMaasChatModel(config=cfg)

    url = service._get_target_endpoint()
    payload = service._build_payload([HumanMessage(content="Summer travel plan to Paris")])

    print("호출 URL      :", url)
    print("요청 모델     :", payload["model"])
    print("Max Tokens    :", payload["max_tokens"])
    print("chat_template :", payload.get("chat_template_kwargs"))

    assert payload["model"] == "google/gemma-4-26b-a4b-it-maas"
    assert payload["max_tokens"] == 128000
    assert payload.get("chat_template_kwargs") == {"enable_thinking": True}

    # 동기 호출 검증
    res = service.invoke([HumanMessage(content="Summer travel plan to Paris")])
    print("응답 내용     :\n", res.content)
    assert res.content
    print("✅ [성공] Gemma 4 26B (Thinking 포함) 테스트 통과")


def test_chatbot_model_switching():
    print("\n" + "=" * 55)
    print(" [3. 대화 도중 모델 동적 전환(DeepSeek <-> Gemma 4) 검증]")
    print("=" * 55)

    bot = ModelSelectorChatbot(
        config=ModelConfig.from_model_key("deepseek-flash", mock_mode=True)
    )

    print("턴 1 [DeepSeek]: '내 이름은 민우야'")
    reply1 = bot.ask("내 이름은 민우야")
    print("응답 1:", reply1)
    assert "DeepSeek" in reply1

    # 대화 기록을 유지한 채 모델을 Gemma 4로 전환
    print("\n👉 모델을 'gemma-4-26b'로 전환 (Thinking 활성화)")
    bot.switch_model("gemma-4-26b", enable_thinking=True)

    print("\n턴 2 [Gemma 4]: '내 이름이 뭐라고 했지?'")
    reply2 = bot.ask("내 이름이 뭐라고 했지?")
    print("응답 2:", reply2)
    assert "Gemma 4" in reply2

    history = bot.get_history_messages()
    print("\n유지된 대화 총 턴 수:", len(history))
    assert len(history) == 4, f"메모리 유지 실패: {len(history)} != 4"
    print("✅ [성공] 모델 동적 전환 및 대화 메모리 보존 테스트 통과")


def main():
    print("🚀 GCP Vertex AI MaaS 모델 선택 챗봇 - 통합 테스트 시작")
    test_deepseek_specs()
    test_gemma4_thinking_specs()
    test_chatbot_model_switching()
    print("\n🎉 모든 모델별 엔드포인트 및 옵션(Thinking 포함) 테스트 통과 완료!")


if __name__ == "__main__":
    main()
