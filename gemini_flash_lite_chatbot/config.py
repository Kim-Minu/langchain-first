"""Gemini 3.5 Flash Lite Chatbot - Configuration Module

Google Cloud Vertex AI 및 Gemini 3.5 Flash Lite 모델 연동 설정을 관리합니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# 루트 및 로컬 .env 로드
load_dotenv()
local_env = Path(__file__).parent / ".env"
if local_env.exists():
    load_dotenv(dotenv_path=local_env)


@dataclass
class GeminiConfig:
    """Gemini 3.5 Flash Lite 챗봇 설정."""

    # 서비스 및 모델 식별자 (기본값: gemini-3.5-flash-lite)
    SERVICE_NAME: str = "Gemini 3.5 Flash Lite Chatbot Service"
    model_name: str = os.getenv("GEMINI_FLASH_LITE_MODEL", "gemini-3.5-flash-lite")

    # GCP 설정
    project_id: str = os.getenv(
        "GOOGLE_CLOUD_PROJECT",
        os.getenv("PROJECT_ID", "iceu-songpa05"),
    )
    location: str = os.getenv(
        "GOOGLE_CLOUD_LOCATION",
        os.getenv("REGION", "global"),
    )

    # Vertex AI 활성화 여부 (True: ADC/Vertex AI, False: Google AI Studio API Key)
    vertexai: bool = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() in ("true", "1", "yes")

    # 선택적 Google API Key
    api_key: str | None = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", None))

    # 추론 파라미터
    temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    max_output_tokens: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))
    top_p: float = float(os.getenv("GEMINI_TOP_P", "0.95"))
    reasoning_effort: str = os.getenv("GEMINI_REASONING_EFFORT", "low")  # low | medium | high
    timeout: float = float(os.getenv("GEMINI_TIMEOUT", "60.0"))

    # 모의(Mock) 오프라인 테스트 모드
    mock_mode: bool = os.getenv("GEMINI_MOCK_MODE", "false").lower() in ("true", "1", "yes")


default_config = GeminiConfig()
