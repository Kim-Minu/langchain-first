"""Model Selector Chatbot - CLI Interactive Chatbot

터미널 환경에서 LangChain 기반 DeepSeek V4.1 Flash 및 Gemma 4 26B A4B IT 모델을
선택하고 테스트할 수 있는 대화형 CLI 프로그램입니다.
"""

import argparse
import sys
import os

# site-packages 및 모듈 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv/lib/python3.14/site-packages")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_selector_chatbot.config import SUPPORTED_MODELS, ModelConfig
from model_selector_chatbot.model_service import GcpMaasChatModel
from model_selector_chatbot.chatbot import ModelSelectorChatbot, get_default_system_prompt


def print_banner(config: ModelConfig, service: GcpMaasChatModel, system_prompt: str = ""):
    meta = SUPPORTED_MODELS.get(config.model_key, {})
    print("=" * 72)
    print("       🤖 GCP Vertex AI MaaS 모델 선택 대화형 챗봇 (LangChain)")
    print("=" * 72)
    print(f" • 활성 모델   : {config.model_id} ({config.service_name})")
    print(f" • ENDPOINT    : {config.endpoint_host}")
    print(f" • REGION      : {config.location}")
    print(f" • PROJECT_ID  : {config.project_id}")
    print(f" • Thinking 모드: {'[ON]' if config.enable_thinking else '[OFF]'}")
    print(f" • Max Tokens  : {config.max_tokens}")
    print(f" • 작동 모드   : {'[Mock 테스트 모드]' if config.mock_mode else '[GCP Vertex AI 실시간 API]'}")
    if system_prompt:
        short_prompt = system_prompt if len(system_prompt) <= 60 else system_prompt[:57] + "..."
        print(f" • 시스템 프롬프트: {short_prompt}")
    print(" • 명령어      : /model, /thinking, /system, /reset, /history, /info, /exit")
    print("=" * 72)
    print()


def prompt_select_model() -> str:
    print("\n[사용 가능한 모델 목록]")
    keys = list(SUPPORTED_MODELS.keys())
    for idx, k in enumerate(keys, 1):
        m = SUPPORTED_MODELS[k]
        print(f" {idx}. {m['display_name']} ({m['id']})")
    choice = input("선택할 모델 번호를 입력하세요 (엔터 시 취소): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(keys):
        return keys[int(choice) - 1]
    return ""


def main():
    parser = argparse.ArgumentParser(description="GCP MaaS Model Selector CLI Chatbot")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-flash",
        choices=["deepseek-flash", "gemma-4-26b"],
        help="사용할 모델 (deepseek-flash | gemma-4-26b)",
    )
    parser.add_argument("--thinking", action="store_true", help="Thinking 모드 활성화 (Gemma 4 등)")
    parser.add_argument("--mock", action="store_true", help="오프라인 모의 응답 모드로 실행")
    parser.add_argument("--endpoint", type=str, default=None, help="GCP 엔드포인트 도메인 (${ENDPOINT})")
    parser.add_argument("--region", type=str, default=None, help="GCP 리전 (${REGION})")
    parser.add_argument("--project", type=str, default=None, help="GCP 프로젝트 ID (${PROJECT_ID})")
    parser.add_argument("--system", type=str, default=None, help="사용자 지정 시스템 프롬프트")
    args = parser.parse_args()

    cfg = ModelConfig.from_model_key(
        model_key=args.model,
        endpoint_host=args.endpoint,
        location=args.region,
        project_id=args.project,
        enable_thinking=args.thinking if args.thinking else None,
        mock_mode=args.mock,
    )

    service = GcpMaasChatModel(config=cfg)
    chatbot = ModelSelectorChatbot(service=service, config=cfg, system_prompt=args.system)

    print_banner(cfg, service, chatbot.system_prompt)

    while True:
        try:
            user_input = input(f"\n👤 [{cfg.model_key}] 질문: ").strip()
            if not user_input:
                continue

            # 종료 명령어
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                print("\n챗봇을 종료합니다. 감사합니다.")
                break

            # 대화 초기화
            elif user_input.lower() in ("/reset", "/clear"):
                chatbot.clear_history()
                print("🧹 [대화 기록이 초기화되었습니다.]")
                continue

            # 대화 기록 보기
            elif user_input.lower() == "/history":
                history_msgs = chatbot.get_history_messages()
                print("\n--- [대화 기록] ---")
                if not history_msgs:
                    print("(기록이 없습니다)")
                else:
                    for m in history_msgs:
                        sender = "사용자" if m.type == "human" else "AI"
                        print(f"[{sender}]: {m.content}")
                print("------------------")
                continue

            # 설정 정보 확인
            elif user_input.lower() == "/info":
                print("\n--- [현재 서비스 및 모델 설정 정보] ---")
                print(f"• 모델 키      : {cfg.model_key}")
                print(f"• 모델 ID      : {cfg.model_id}")
                print(f"• 서비스 명칭   : {cfg.service_name}")
                print(f"• ENDPOINT     : {cfg.endpoint_host}")
                print(f"• REGION       : {cfg.location}")
                print(f"• PROJECT_ID   : {cfg.project_id}")
                print(f"• 엔드포인트 URL: {service._get_target_endpoint()}")
                print(f"• Thinking 모드: {cfg.enable_thinking}")
                print(f"• Max Tokens   : {cfg.max_tokens}")
                print(f"• Mock Mode    : {cfg.mock_mode}")
                print(f"• 시스템 프롬프트: {chatbot.system_prompt}")
                print("---------------------------------------")
                continue

            # 시스템 프롬프트 확인 및 변경 명령어
            elif user_input.lower().startswith("/system"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    print(f"\n📝 [현재 시스템 프롬프트]:\n\"{chatbot.system_prompt}\"\n")
                    print("💡 팁: 변경하려면 '/system <새로운 프롬프트 내용>' 형식으로 입력하세요.")
                else:
                    new_sys_prompt = parts[1].strip()
                    chatbot.set_system_prompt(new_sys_prompt)
                    print(f"\n✅ [시스템 프롬프트가 변경되었습니다]:\n\"{chatbot.system_prompt}\"")
                continue

            # 모델 전환
            elif user_input.lower() == "/model":
                new_key = prompt_select_model()
                if new_key:
                    chatbot.switch_model(new_key)
                    cfg = chatbot.config
                    service = chatbot.service
                    print(f"\n🔄 [모델이 전환되었습니다: {cfg.model_id}]")
                    print_banner(cfg, service, chatbot.system_prompt)
                continue

            # Thinking 모드 토글
            elif user_input.lower() == "/thinking":
                cfg.enable_thinking = not cfg.enable_thinking
                service.enable_thinking = cfg.enable_thinking
                chatbot._rebuild_chain()
                print(f"💡 [Thinking 모드 상태 변경 -> {'ON (켜짐)' if cfg.enable_thinking else 'OFF (꺼짐)'}]")
                continue

            # 답변 스트리밍 출력
            print(f"🤖 [{cfg.service_name}]: ", end="", flush=True)
            try:
                for chunk in chatbot.stream_ask(user_input):
                    print(chunk, end="", flush=True)
                print()
            except Exception as e:
                print(f"\n❌ [오류 발생]: {e}")
                print("💡 팁: 네트워크/권한 이슈 시 사이드바/옵션에서 Mock 모드로 확인하실 수 있습니다.")

        except (KeyboardInterrupt, EOFError):
            print("\n\n프로그램을 종료합니다.")
            break


if __name__ == "__main__":
    main()
