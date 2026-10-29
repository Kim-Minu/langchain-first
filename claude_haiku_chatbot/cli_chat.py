"""Claude Haiku 5.5 - CLI Interactive Chatbot

터미널 환경에서 Anthropic claude-haiku-5-5 모델과 대화할 수 있는 인터랙티브 CLI 프로그램입니다.
"""

from __future__ import annotations

import argparse
import os
import sys

# 프로젝트 및 의존성 경로 등록
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv/lib/python3.14/site-packages")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from claude_haiku_chatbot.chatbot import ClaudeHaikuChatbot
from claude_haiku_chatbot.claude_service import ClaudeHaikuVertexChatModel
from claude_haiku_chatbot.config import ClaudeHaikuConfig


def print_banner(cfg: ClaudeHaikuConfig) -> None:
    print("=" * 68)
    print("      🤖 Anthropic Claude Haiku 5.5 Chatbot - LangChain CLI")
    print("=" * 68)
    print(f" • 모델 식별자 : {cfg.model_name}")
    print(f" • GCP 프로젝트: {cfg.project_id}")
    print(f" • GCP 리전   : {cfg.region}")
    print(f" • Max Tokens : {cfg.max_tokens}")
    print(f" • 동작 모드   : {'[Mock 테스트 모드]' if cfg.mock_mode else '[GCP Vertex AI 실시간 API]'}")
    print(" • 명령어      : /reset, /history, /system, /info, /mode, /exit")
    print("=" * 68)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Haiku 5.5 CLI Chatbot")
    parser.add_argument("--mock", action="store_true", help="오프라인 모의 모드로 실행")
    parser.add_argument("--project", type=str, default=None, help="GCP 프로젝트 ID")
    parser.add_argument("--region", type=str, default=None, help="GCP 리전 (기본: global)")
    parser.add_argument("--model", type=str, default=None, help="모델명 (기본: claude-haiku-5-5)")
    parser.add_argument("--system", type=str, default=None, help="시스템 프롬프트")
    args = parser.parse_args()

    cfg = ClaudeHaikuConfig()
    if args.mock:
        cfg.mock_mode = True
    if args.project:
        cfg.project_id = args.project
    if args.region:
        cfg.region = args.region
    if args.model:
        cfg.model_name = args.model
    if args.system:
        cfg.default_system_prompt = args.system

    service = ClaudeHaikuVertexChatModel(config=cfg)
    chatbot = ClaudeHaikuChatbot(service=service, config=cfg)

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
                        sender = "사용자" if m.type == "human" else "Claude"
                        print(f"[{sender}]: {m.content}")
                print("------------------")
                continue

            elif user_input.lower() == "/info":
                print("\n--- [서비스 설정 정보] ---")
                print(f"• 모델: {cfg.model_name}")
                print(f"• GCP Project: {cfg.project_id}")
                print(f"• GCP Region: {cfg.region}")
                print(f"• Max Tokens: {cfg.max_tokens}")
                print(f"• Mock Mode: {service.mock_mode}")
                print(f"• System Prompt: {chatbot.system_prompt}")
                print("--------------------------")
                continue

            elif user_input.lower().startswith("/system"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    new_sys = parts[1].strip()
                    chatbot.set_system_prompt(new_sys)
                    print(f"⚙️ [시스템 프롬프트가 변경되었습니다]: {new_sys}")
                else:
                    print(f"⚙️ [현재 시스템 프롬프트]: {chatbot.system_prompt}")
                continue

            elif user_input.lower() == "/mode":
                service.mock_mode = not service.mock_mode
                cfg.mock_mode = service.mock_mode
                service._init_client()
                print(f"🔄 [모드가 전환되었습니다 -> {'Mock 모드' if service.mock_mode else 'GCP Vertex AI 실시간 모드'}]")
                continue

            # 모델 스트리밍 응답
            print("\n🤖 Claude Haiku: ", end="", flush=True)
            for token in chatbot.stream_ask(user_input):
                print(token, end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\n\n입력이 취소되었습니다. 종료하려면 /exit를 입력하세요.")
        except Exception as e:
            print(f"\n❌ [오류 발생]: {e}")


if __name__ == "__main__":
    main()

