"""RAG 실습의 기본 설정값. 화면 사이드바에서 바꿀 수 있는 값은 범위(min/max)도 함께 둔다."""

import os
from dataclasses import dataclass
from pathlib import Path

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
# 페이지를 이어 붙인 뒤 나누기 (페이지 경계에서 끊긴 문장을 한 청크로)
CHUNK_ACROSS_PAGES = False

# 임베딩 차원 (Gemini 임베딩 기본 3072, 768 · 1536으로 줄여도 검색 품질 손실이 작음)
# 같은 컬렉션 안에서는 차원이 같아야 비교할 수 있으므로, 차원을 바꾸면 다시 인덱싱해야 함
EMBEDDING_DIMENSION_OPTIONS = [768, 1536, 3072]
EMBEDDING_DIMENSION = 768
# 같은 텍스트의 벡터를 PostgreSQL embedding_cache 테이블에 저장해 재사용 (모델 · 차원 · 문서/질문용별)
EMBEDDING_CACHE = os.getenv("RAG_EMBEDDING_CACHE", "1") != "0"
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

# ---------------------------------------------------------------------------
# OCR · RAG 평가 화면 (pages/ocr_eval.py)
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent / "eval"
# 원본 PDF · OCR 캐시는 저작물이라 git에서 제외한 data/ 폴더에 둠
EVAL_DATA_DIR = EVAL_DIR / "data"


@dataclass(frozen=True)
class EvalDoc:
    label: str
    pdf: Path  # 평가할 스캔 PDF (git 제외 폴더)
    qa_file: Path  # 질문 · 정답 세트 (git에 포함)
    collection: str  # pgvector 컬렉션


# 평가 문서 목록: 화면에서 고르면 PDF · 질문 세트 · 컬렉션이 함께 바뀜
EVAL_DOCS = {
    "han-gang-20": EvalDoc(
        "한강 다리 1~20쪽", EVAL_DATA_DIR / "han-gang_1-20.pdf", EVAL_DIR / "han_gang_qa.json", "rag_eval_han_gang",
    ),
    "han-gang-100": EvalDoc(
        "한강 다리 1~100쪽", EVAL_DATA_DIR / "han-gang_1-100.pdf", EVAL_DIR / "han_gang_100_qa.json",
        "rag_eval_han_gang_100",
    ),
}
EVAL_DOC = "han-gang-20"
EVAL_QUESTION_COUNT = 20
# 너무 높이면 할당량(429)에 걸림
EVAL_CONCURRENCY = int(os.getenv("RAG_EVAL_CONCURRENCY", "2"))  # 문항마다 답변 · 채점 · 근거 충실도 3회 호출

# 스캔한 책 페이지 → 인쇄된 텍스트만 (사진 · 그림 · 지도 안의 글자와 그림 설명은 제외)
OCR_PROMPT = (
    "이 이미지는 책의 한 페이지를 스캔한 것입니다. 페이지에 인쇄된 텍스트만 원문 그대로 받아 적으세요.\n"
    "- 포함: 본문, 소제목, 인용문, 각주, 사진 · 그림 아래의 캡션 문장 (예: '0-1. 서울 북한산 한옥마을')\n"
    "- 제외: 사진 · 그림 · 지도 · 도표 안에 들어 있는 글자, 이미지에 대한 설명, 페이지 머리말(장 제목 반복)과 쪽 번호\n"
    "- 줄 끝에서 끊긴 문장은 이어 붙이고, 문단 사이는 빈 줄 하나로 구분\n"
    "- 소제목은 '## 소제목' 형태로 한 줄에 쓰기\n"
    "- 한자 병기는 원문대로 붙여 쓰기 (예: 대수帶水)\n"
    "설명 문구나 코드 블록 기호 없이 텍스트만 출력하세요. 인쇄된 텍스트가 없는 페이지는 빈 문자열만 출력하세요."
)

# 질문 · 정답 쌍 생성 (문서 내용만으로 답할 수 있는 사실형 질문)
QA_GENERATION_PROMPT = (
    "아래는 책의 페이지별 본문입니다. 이 내용만으로 답할 수 있는 질문과 정답 {count}쌍을 만드세요.\n"
    "- 사실 확인형(인물 · 연도 · 수치 · 명칭 · 장소 · 이유)을 고르게 섞고, 여러 페이지에 고르게 분포시키기\n"
    "- 정답은 문서 표현을 살려 한두 문장으로 짧게, 근거 페이지 번호를 함께\n"
    "- 문서 밖 지식이 필요하거나 '이 책' 같은 모호한 지시어가 필요한 질문은 제외\n\n{pages}"
)

# 채점 모델: 답변 모델과 다른 모델을 써서 자기 답에 후한 편향(self-preference)을 줄임
# (gemini-4-argon은 이 프로젝트에서 403, Pro 계열 중 호출되는 미리보기 모델을 기본값으로)
JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "gemini-3.1-pro-preview")

# 답이 없는 질문(문서에 없는 내용)의 정답 표기
UNANSWERABLE = "문서에서 찾을 수 없음"

# 채점: 문서 기반 정답과 비교해 RAG 답변을 판정 (LLM-as-judge)
JUDGE_PROMPT = (
    "질문, 정답(문서 기준), 평가할 답변이 주어집니다. 답변이 정답의 핵심 사실을 맞혔는지 판정하세요.\n"
    "- correct: 핵심 사실을 모두 맞힘 (표현이 달라도 됨, 정답에 없는 내용이 더 있어도 틀리지 않았다면 정답)\n"
    "- partial: 핵심 사실 일부만 맞힘\n"
    "- incorrect: 틀렸거나, 찾을 수 없다고 답했거나, 핵심 사실이 없음\n"
    f"정답이 '{UNANSWERABLE}'이면 문서에 답이 없는 질문입니다. 답변이 문서에서 찾을 수 없다고 밝히면 correct, "
    "구체적인 답(연도 · 수치 · 이름 등)을 제시하면 문서 밖 지식이나 추측이므로 incorrect입니다.\n\n"
    "질문: {question}\n정답: {gold}\n답변: {answer}"  # f-string이 아닌 조각이라 중괄호 한 번
)

# 근거 충실도(faithfulness): 답변의 주장이 검색된 문맥에 근거하는지 (정답 데이터 없이도 측정 가능)
FAITHFULNESS_PROMPT = (
    "[문서]와 [답변]이 주어집니다. 답변을 사실 주장 단위로 나누고, 각 주장이 [문서]에 의해 뒷받침되는지 판정하세요.\n"
    "- 문서에 직접 쓰여 있거나 문서 내용만으로 바로 알 수 있으면 supported, 아니면 unsupported\n"
    "- '문서에서 찾을 수 없다'는 문장, 인사말, 근거 번호([1] 등)는 주장이 아니므로 제외\n"
    "- 주장이 하나도 없으면 빈 목록\n\n"
    "[문서]\n{context}\n\n[답변]\n{answer}"
)
