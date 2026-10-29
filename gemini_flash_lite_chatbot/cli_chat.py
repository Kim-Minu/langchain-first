"""Gemini 3.5 Flash Lite - CLI Interactive Chatbot

터미널 환경에서 gemini-3.5-flash-lite 모델과 대화할 수 있는 인터랙티브 CLI 프로그램입니다.
"""

import argparse
import sys
import os

# 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv/lib/python3.14/site-packages")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemini_flash_lite_chatbot.config import GeminiConfig
from gemini_flash_lite_chatbot.gemini_service import GeminiFlashLiteService
from gemini_flash_lite_chatbot.chatbot import GeminiFlashLiteChatbot


def print_banner(cfg: GeminiConfig):
    print("=" * 68)
    print("       🤖 Gemini 3.5 Flash Lite Chatbot - LangChain CLI")
    print("=" * 68)
    print(f" • 모델 식별자 : {cfg.model_name}")
    print(f" • GCP 프로젝트: {cfg.project_id}")
    print(f" • GCP 리전   : {cfg.location}")
    print(f" • 추론 수준   : {cfg.reasoning_effort}")
    print(f" • 동작 모드   : {'[Mock 테스트 모드]' if cfg.mock_mode else '[GCP Vertex AI 실시간 API]'}")
    print(" • 명령어      : /reset (대화초기화), /history (기록), /info (설정), /mode (모드전환), /exit (종료)")
    print("=" * 68)
    print()


def main():
    parser = argparse.ArgumentParser(description="Gemini 3.5 Flash Lite CLI Chatbot")
    parser.add_argument("--mock", action="store_true", help="오프라인 모의 모드로 실행")
    parser.add_argument("--project", type=str, default=None, help="GCP 프로젝트 ID")
    parser.add_argument("--location", type=str, default=None, help="GCP 리전")
    parser.add_argument("--model", type=str, default=None, help="모델명")
    args = parser.parse_args()

    cfg = GeminiConfig()
    if args.mock:
        cfg.mock_mode = True
    if args.project:
        cfg.project_id = args.project
    if args.location:
        cfg.location = args.location
    if args.model:
        cfg.model_name = args.model

    service = GeminiFlashLiteService(config=cfg)
    chatbot = GeminiFlashLiteChatbot(service=service, config=cfg)

    print_banner(cfg)

    while True:
        try:
            user_input = input("\n👤 사용자: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                print("\n챗봇을 종료합니다. 감사합니다.")
                break

            elif user_input.lower() in ("/reset", "/clear"):
                chatbot.clear_history()
                print("🧹 [대화 기록이 초기화되었습니다.]")
                continue

            elif user_input.lower() == "/history":
                history_msgs = chatbot.get_history_messages()
                print("\n--- [대화 기록] ---")
                if not history_msgs:
                    print("(기록이 없습니다)")
                else:
                    for m in history_msgs:
                        sender = "사용자" if m.type == "human" else "Gemini"
                        print(f"[{sender}]: {m.content}")
                print("------------------")
                continue

            elif user_input.lower() == "/info":
                print("\n--- [서비스 설정 정보] ---")
                print(f"• 모델: {cfg.model_name}")
                print(f"• GCP Project: {cfg.project_id}")
                print(f"• GCP Location: {cfg.location}")
                print(f"• Reasoning Effort: {cfg.reasoning_effort}")
                print(f"• Mock Mode: {service.mock_mode}")
                print(f"• Max Output Tokens: {cfg.max_output_tokens}")
                print("--------------------------")
                continue

            elif user_input.lower() == "/mode":
                service.mock_mode = not service.mock_mode
                cfg.mock_mode = service.mock_mode
                service._init_inner_client()
                print(f"🔄 [모드가 전환되었습니다 -> {'Mock 모드' if service.mock_mode else 'GCP Vertex AI 실시간 모드'}]")
                continue

            print(f"🤖 Gemini Flash Lite: ", end="", flush=True)
            try:
                for chunk in chatbot.stream_ask(user_input):
                    print(chunk, end="", flush=True)
                print()
            except Exception as e:
                print(f"\n❌ [오류 발생]: {e}")
                print("💡 팁: 네트워크/자격증명 문제 시 '/mode' 명령어로 Mock 모드 전환이 가능합니다.")

        except (KeyboardInterrupt, EOFError):
            print("\n\n프로그램을 종료합니다.")
            break


if __name__ == "__main__":
    main()

