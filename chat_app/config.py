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
GEMINI_ARGON_MODEL = os.getenv("GEMINI_ARGON_MODEL", "gemini-4-argon")

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
    "Google Gemini 4.0 Argon (Vertex AI)": "Gemini 4.0 Argon",
    "Ollama Gemma (gemma4:e2b)": "Gemma 4 (로컬)",
}
MODEL_CAPTIONS = ["Vertex AI · 빠른 응답", "Vertex AI · Gemini 4.0", "Ollama · 로컬 실행, 무료"]

# Gemini 선택지 키 → Vertex AI 모델 ID
GEMINI_MODEL_IDS = {
    "Google Gemini (Vertex AI)": GEMINI_MODEL,
    "Google Gemini 4.0 Argon (Vertex AI)": GEMINI_ARGON_MODEL,
}

# 사고 수준 (Gemini reasoning_effort): {값: 화면 표시 이름} — Gemini 앱의 "사고 수준" 메뉴와 동일한 3단계
REASONING_OPTIONS = {"low": "낮음", "medium": "보통", "high": "높음"}
REASONING_CAPTIONS = ["빠르고 효율적", "균형 잡힌 깊이", "매우 철저함"]
DEFAULT_REASONING_EFFORT = "low"

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

# 날씨 카드: 질문에 장소가 없을 때 쓰는 기본 장소
WEATHER_DEFAULT_PLACE = os.getenv("WEATHER_DEFAULT_PLACE", "서울특별시")
WEATHER_DEFAULT_LATITUDE = float(os.getenv("WEATHER_DEFAULT_LATITUDE", "37.5665"))
WEATHER_DEFAULT_LONGITUDE = float(os.getenv("WEATHER_DEFAULT_LONGITUDE", "126.9780"))
# 장소 추출 호출의 출력 토큰 상한 (Gemini는 추론 토큰 포함이라 너무 낮으면 JSON이 잘림)
WEATHER_EXTRACT_MAX_TOKENS = 2048

# 예상 비용 계산 (Gemini Flash 요율, USD / 1M 토큰)
GEMINI_INPUT_USD_PER_1M = 0.075
GEMINI_OUTPUT_USD_PER_1M = 0.30
KRW_PER_USD = 1380


def is_gemini(model_name: str) -> bool:
    return "Gemini" in model_name


def gemini_model_id(model_name: str) -> str:
    return GEMINI_MODEL_IDS.get(model_name, GEMINI_MODEL)


def is_search_active(model_name: str, search_mode: str) -> bool:
    """선택한 모델 기준으로 실제로 검색이 연결되는지 여부 (Grounding은 Gemini 전용)."""
    return search_mode == SEARCH_GROUNDING and is_gemini(model_name)
