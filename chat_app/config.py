"""환경변수와 화면 옵션 상수."""

import os

# PostgreSQL (docker-compose.yml 로컬 기본값)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://langchain:langchain@localhost:5433/langchain_chat",
)

# Vertex AI Gemini
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# 답변 1회당 출력 토큰 상한 (Gemini: max_output_tokens, Ollama: num_predict)
# Gemini는 추론(thinking) 토큰도 출력 토큰에 포함되므로 너무 낮게 잡으면 답변이 비거나 잘릴 수 있음
# 화면 입력칸의 허용 범위와 기본값 (환경변수 값은 범위 안으로 맞춤)
OUTPUT_TOKENS_MIN = 1
OUTPUT_TOKENS_MAX = 65536
MAX_OUTPUT_TOKENS = min(max(int(os.getenv("MAX_OUTPUT_TOKENS", "8192")), OUTPUT_TOKENS_MIN), OUTPUT_TOKENS_MAX)

# Vertex AI Grounding 답변 캐시 유효 시간(초). 0이면 캐시 사용 안 함
# 검색이 반영된 답변은 금방 낡으므로(날씨·뉴스 등) 짧게 유지
SEARCH_CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "600"))

# 모델 선택지: {내부 키: 화면 표시 이름}
MODEL_OPTIONS = {
    "Google Gemini (Vertex AI)": "Gemini Flash",
    "Ollama Gemma (gemma4:e2b)": "Gemma 4 (로컬)",
}
MODEL_CAPTIONS = ["Vertex AI · 빠른 응답", "Ollama · 로컬 실행, 무료"]

# 검색 방식
SEARCH_OFF = "off"
SEARCH_GROUNDING = "grounding"
SEARCH_OPTIONS = {
    SEARCH_OFF: "끄기",
    SEARCH_GROUNDING: "Vertex AI Grounding",
}
SEARCH_CAPTIONS = [
    "검색 없이 모델 지식으로만 답변",
    "Gemini 내장 검색 · Google 서버에서 처리",
]

# 예상 비용 계산 (Gemini Flash 요율, USD / 1M 토큰)
GEMINI_INPUT_USD_PER_1M = 0.075
GEMINI_OUTPUT_USD_PER_1M = 0.30
KRW_PER_USD = 1380


def is_gemini(model_name: str) -> bool:
    return "Gemini" in model_name


def is_search_active(model_name: str, search_mode: str) -> bool:
    """선택한 모델 기준으로 실제로 검색이 연결되는지 여부 (Grounding은 Gemini 전용)."""
    return search_mode == SEARCH_GROUNDING and is_gemini(model_name)
