"""Model Selector Chatbot - Configuration Module

GCP Vertex AI MaaS (Model-as-a-Service) 환경에서 다양한 오픈/파트너 모델
(DeepSeek V4.1 Flash, Gemma 4 26B 등)을 선택적으로 호출하기 위한 설정 모듈입니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# 루트 및 로컬 디렉토리의 .env 로드
load_dotenv()
local_env = Path(__file__).parent / ".env"
if local_env.exists():
    load_dotenv(dotenv_path=local_env)


# 지원하는 모델 카탈로그 및 모델별 기본 사양
SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
    "deepseek-flash": {
        "id": "deepseek-ai/deepseek-v4.1-flash-maas",
        "display_name": "DeepSeek V4.1 Flash (MaaS)",
        "service_name": "DeepSeek V4.1 Flash API Service",
        "default_endpoint": "aiplatform.googleapis.com",
        "default_region": "global",
        "default_max_tokens": 4096,
        "supports_thinking": False,
        "default_enable_thinking": False,
        "description": "초고속 응답 속도와 경제적인 추론 비용을 제공하는 Flash 모델",
        "default_system_prompt": (
            "당신은 DeepSeek V4.1 Flash API Service 기반의 지능형 AI 어시스턴트입니다. "
            "사용자의 질문에 대해 신속하고 핵심을 명확히 짚는 정중한 한국어로 성실히 답변해 주세요."
        ),
    },
    "gemma-4-26b": {
        "id": "google/gemma-4-26b-a4b-it-maas",
        "display_name": "Gemma 4 26B A4B IT (MaaS)",
        "service_name": "Gemma 4 26B IT API Service",
        "default_endpoint": "aiplatform.googleapis.com",
        "default_region": "global",
        "default_max_tokens": 128000,
        "supports_thinking": True,
        "default_enable_thinking": True,
        "description": "최대 128k 토큰 및 내장 Thinking(추론) 템플릿을 지원하는 고성능 26B 모델",
        "default_system_prompt": (
            "당신은 Google Gemma 4 26B A4B IT 기반의 심층 추론 AI 어시스턴트입니다. "
            "깊이 있는 논리적 사고(Thinking)를 바탕으로 체계적이고 구체적인 답변을 제공해 주세요."
        ),
    },
}

DEFAULT_MODEL_KEY = "deepseek-flash"


def get_default_system_prompt(model_key_or_id: str = DEFAULT_MODEL_KEY) -> str:
    """모델 키 또는 모델 식별자(ID)에 따른 모델 고유의 기본 시스템 프롬프트를 반환합니다."""
    # 1. key 직접 매칭 (예: 'deepseek-flash', 'gemma-4-26b')
    if model_key_or_id in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_key_or_id]["default_system_prompt"]

    # 2. model id 매칭 (예: 'deepseek-ai/deepseek-v4.1-flash-maas')
    for meta in SUPPORTED_MODELS.values():
        if meta["id"] == model_key_or_id:
            return meta["default_system_prompt"]

    # 3. 폴백
    return SUPPORTED_MODELS[DEFAULT_MODEL_KEY]["default_system_prompt"]


@dataclass
class ModelConfig:
    """개별 모델에 대한 호출 설정."""

    model_key: str = DEFAULT_MODEL_KEY
    model_id: str = "deepseek-ai/deepseek-v4.1-flash-maas"
    service_name: str = "DeepSeek V4.1 Flash API Service"
    endpoint_host: str = "aiplatform.googleapis.com"
    location: str = "global"
    project_id: str = "iceu-songpa05"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95
    timeout: float = 120.0
    enable_thinking: bool = False
    system_prompt: str = ""
    custom_endpoint_url: str | None = None
    api_key: str | None = None
    mock_mode: bool = False

    @classmethod
    def from_model_key(
        cls,
        model_key: str = DEFAULT_MODEL_KEY,
        project_id: str | None = None,
        endpoint_host: str | None = None,
        location: str | None = None,
        **overrides: Any,
    ) -> ModelConfig:
        """모델 키로부터 기본값을 채우고 오버라이드 설정을 반영한 ModelConfig 생성."""
        meta = SUPPORTED_MODELS.get(model_key, SUPPORTED_MODELS[DEFAULT_MODEL_KEY])

        proj = (
            project_id
            or os.getenv("PROJECT_ID")
            or os.getenv("GOOGLE_CLOUD_PROJECT", "iceu-songpa05")
        )
        host = (
            endpoint_host
            or os.getenv("ENDPOINT", meta["default_endpoint"])
        )
        loc = (
            location
            or os.getenv("REGION")
            or os.getenv("GOOGLE_CLOUD_LOCATION", meta["default_region"])
        )

        cfg = cls(
            model_key=model_key,
            model_id=meta["id"],
            service_name=meta["service_name"],
            endpoint_host=host,
            location=loc,
            project_id=proj,
            max_tokens=meta["default_max_tokens"],
            enable_thinking=meta.get("default_enable_thinking", False),
            system_prompt=meta.get("default_system_prompt", ""),
            mock_mode=os.getenv("DEEPSEEK_MOCK_MODE", "false").lower() in ("true", "1", "yes"),
        )

        for k, v in overrides.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)

        return cfg

    @property
    def effective_endpoint_url(self) -> str:
        """실제 호출할 최종 Chat Completions 엔드포인트 URL 반환.
        
        형식: https://${ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/openapi/chat/completions
        """
        if self.custom_endpoint_url:
            return self.custom_endpoint_url

        host = self.endpoint_host.strip()
        if host.startswith("https://"):
            host = host[8:]
        elif host.startswith("http://"):
            host = host[7:]
        host = host.rstrip("/")

        return (
            f"https://{host}/v1/"
            f"projects/{self.project_id}/locations/{self.location}/endpoints/openapi/chat/completions"
        )


# 이전 호환성을 위한 alias
DeepSeekConfig = ModelConfig
default_config = ModelConfig.from_model_key("deepseek-flash")
