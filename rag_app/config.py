"""RAG 실습의 기본 설정값. 화면 사이드바에서 바꿀 수 있는 값은 범위(min/max)도 함께 둔다."""

import os

# Vertex AI (chat_app과 같은 ADC 인증 · 프로젝트 사용)
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
# 엔드포인트 리전: gemini-embedding-2는 global 엔드포인트에서만 제공됨 (us-central1은 404)
LLM_LOCATION = os.getenv("RAG_LLM_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
EMBEDDING_LOCATION = os.getenv("RAG_EMBEDDING_LOCATION", "global")

# Gemini 모델 (LLM_MODEL은 Gemini 답변과 PDF 속 이미지 읽기에 함께 사용)
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-2")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))

# 답변 모델 선택지: {내부 키: 화면 표시 이름} (chat_app/config.py의 MODEL_OPTIONS와 같은 이름)
LLM_GEMINI = "Google Gemini (Vertex AI)"
LLM_OLLAMA_GEMMA = "Ollama Gemma (gemma4:e2b)"
LLM_OPTIONS = {
    LLM_GEMINI: "Gemini Flash",
    LLM_OLLAMA_GEMMA: "Gemma 4 (로컬)",
}
LLM_CAPTIONS = ["Vertex AI · 빠른 응답", "Ollama · 로컬 실행, 무료"]
OLLAMA_MODEL = "gemma4:e2b"

# 벡터 스토어: docker-compose.yml의 PostgreSQL(pgvector) — chat_app과 같은 DB를 쓰고 테이블만 분리
# (langchain-postgres가 langchain_pg_collection / langchain_pg_embedding 테이블을 만들어 관리)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://langchain:langchain@localhost:5433/langchain_chat",
)
# PGVector는 SQLAlchemy 연결 문자열을 쓰므로 psycopg3 드라이버를 명시
PGVECTOR_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "rag_pdf")

# PDF 속 이미지 처리 (스캔 PDF는 페이지 전체가 이미지라 이 옵션 없이는 텍스트가 0자)
# gemini: 이미지마다 Gemini가 글자를 받아 적고 도표 · 그림을 설명한 텍스트를 페이지 본문에 끼워 넣음
# (IMAGE_PROMPT는 기본값이며 화면 사이드바에서 수정 가능)
IMAGE_OFF = "off"
IMAGE_GEMINI = "gemini"
IMAGE_OPTIONS = {
    IMAGE_GEMINI: "Gemini로 글자 추출 · 설명",
    IMAGE_OFF: "끄기 (텍스트만)",
}
IMAGE_MODE = IMAGE_GEMINI
# 페이지 전체를 받아 적을 수 있도록 넉넉하게 (추론 토큰 포함)
IMAGE_MAX_OUTPUT_TOKENS = 8192
# 이미지를 동시에 몇 개씩 Gemini에 보낼지 (스캔 PDF는 페이지 수 = 이미지 수)
IMAGE_CONCURRENCY = int(os.getenv("RAG_IMAGE_CONCURRENCY", "4"))
# 가로 · 세로 중 짧은 쪽이 이보다 작은 이미지(아이콘 · 로고 · 구분선)는 건너뜀 (픽셀)
IMAGE_MIN_SIZE = 64
IMAGE_PROMPT = (
    "이 이미지는 PDF 문서에서 꺼낸 것입니다. 검색(RAG)에 쓸 수 있도록 다음을 마크다운으로 작성하세요.\n"
    "1. 이미지 안의 글자를 빠짐없이 원문 그대로 받아 적기 (표는 마크다운 표로)\n"
    "2. 도표 · 그래프 · 사진 · 다이어그램이 있으면 무엇을 보여 주는지 핵심 수치와 함께 짧게 설명\n"
    "설명 문구나 코드 블록 기호(```) 없이 내용만 출력하세요. 글자도 의미 있는 그림도 없으면 빈 문자열만 출력하세요."
)

# 청크 나누기 (RecursiveCharacterTextSplitter, 글자 수 기준)
CHUNK_SIZE = 1000
CHUNK_SIZE_MIN, CHUNK_SIZE_MAX = 100, 8000
CHUNK_OVERLAP = 200
CHUNK_OVERLAP_MIN, CHUNK_OVERLAP_MAX = 0, 2000

# 임베딩 차원 (Gemini 임베딩 기본 3072, 768 · 1536으로 줄여도 검색 품질 손실이 작음)
# 같은 컬렉션 안에서는 차원이 같아야 비교할 수 있으므로, 차원을 바꾸면 다시 인덱싱해야 함
EMBEDDING_DIMENSION_OPTIONS = [768, 1536, 3072]
EMBEDDING_DIMENSION = 768
# gemini-embedding-2(Vertex AI)는 요청 한 번에 텍스트 하나만 받으므로 청크를 동시에 몇 개씩 요청할지
EMBEDDING_CONCURRENCY = int(os.getenv("RAG_EMBEDDING_CONCURRENCY", "8"))

# 검색할 청크 개수 (top-k)
TOP_K = 4
TOP_K_MIN, TOP_K_MAX = 1, 20

# 답변 생성 (Gemini는 추론 토큰도 출력 토큰에 포함)
MAX_OUTPUT_TOKENS = 4096

# 프롬프트 구성의 "지시문" (사용자가 화면에서 수정 가능)
SYSTEM_PROMPT = (
    "당신은 주어진 문서 내용만 근거로 답하는 어시스턴트입니다.\n"
    "- 아래 [문서] 안의 내용만 사용해 한국어로 답하세요.\n"
    "- 문서에 답이 없으면 '문서에서 찾을 수 없습니다'라고 답하고 지어내지 마세요.\n"
    "- 근거로 쓴 청크 번호를 [1], [2]처럼 문장 끝에 표시하세요."
)
