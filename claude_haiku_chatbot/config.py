"""Claude Haiku 5.5 Chatbot - Configuration Module

GCP Vertex AI 환경의 Anthropic Claude Haiku 5.5 모델 설정을 관리합니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 루트 및 로컬 .env 로드
load_dotenv()
local_env = Path(__file__).parent / ".env"
if local_env.exists():
    load_dotenv(dotenv_path=local_env)


@dataclass
class ClaudeHaikuConfig:
    """Claude Haiku 5.5 Vertex AI 설정 클래스."""

    SERVICE_NAME: str = "Claude Haiku 5.5 Vertex API Service"

    # 모델명 (기본값: claude-haiku-5-5)
    model_name: str = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")

    # GCP 설정
    project_id: str = os.getenv(
        "PROJECT_ID",
        os.getenv("GOOGLE_CLOUD_PROJECT", "iceu-songpa05"),
    )
    region: str = os.getenv(
        "REGION",
        os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
    )

    # 추론 파라미터
    max_tokens: int = int(os.getenv("CLAUDE_MAX_TOKENS", "1024"))
    timeout: float = float(os.getenv("CLAUDE_TIMEOUT", "60.0"))

    # 기본 시스템 프롬프트
    default_system_prompt: str = (
        "당신은 GCP Vertex AI 상의 Anthropic Claude Haiku 5.5 모델 기반의 지능형 AI 어시스턴트입니다. "
        "사용자의 질문에 대해 빠르고 정확하며 정중한 한국어로 답변해 주세요."
    )

    # 모의(Mock) 오프라인 테스트 모드 여부
    mock_mode: bool = os.getenv("CLAUDE_MOCK_MODE", "false").lower() in ("true", "1", "yes")


default_config = ClaudeHaikuConfig()

