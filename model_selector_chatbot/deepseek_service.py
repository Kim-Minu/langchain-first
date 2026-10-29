"""DeepSeek Service module (호환성 래퍼)

공식 명칭: DeepSeek V4.1 Flash API Service
실제 구현은 model_service.py의 GcpMaasChatModel 및 DeepSeekV4FlashApiService를 제공합니다.
"""

from .model_service import (
    DeepSeekV4FlashApiService,
    GcpMaasChatModel,
    _convert_message_to_dict,
)
from .config import ModelConfig as DeepSeekConfig

__all__ = [
    "DeepSeekV4FlashApiService",
    "GcpMaasChatModel",
    "DeepSeekConfig",
]
