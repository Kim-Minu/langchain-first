"""RAG 파이프라인을 단계별 함수로 나눈 모듈 (화면 코드와 분리해 단계마다 결과를 확인할 수 있게 함).

A. 미리 준비하기 (인덱싱 · 자료를 등록하거나 변경할 때)
    1) load_pdf          문서 읽기      PDF → 페이지별 Document
    2) split_documents   청크로 나누기  작은 원문 조각
    3) embed_chunks      임베딩하기     청크 → 숫자 배열 (RETRIEVAL_DOCUMENT)
    4) save_to_store     벡터 스토어 저장 벡터 + 원문 + 출처를 PostgreSQL(pgvector)에 함께 저장

B. 질문에 답하기 (런타임 · 질문이 들어올 때마다)
    1) embed_question    질문 입력      질문도 임베딩 (RETRIEVAL_QUERY)
    2) search_chunks     관련 청크 검색  벡터 비교 → 원문 반환
    3) build_prompt      프롬프트 구성   질문 + 원문 + 지시문
    4) generate_answer   LLM → 답변     근거를 읽고 생성

build_rag_chain은 B의 1~4를 LCEL 체인 하나로 연결한 것 (찾는 일과 쓰는 일을 연결한 전체 흐름 = 체인).
"""

import base64
import bisect
import hashlib
import io
import json
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import google.auth
import psycopg
import pypdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg.types.json import Jsonb

from rag_app import config
from rag_app.config import (
    DATABASE_URL,
    EMBEDDING_CACHE,
    EMBEDDING_CONCURRENCY,
    EMBEDDING_LOCATION,
    GOOGLE_CLOUD_PROJECT,
    IMAGE_CONCURRENCY,
    IMAGE_MIN_SIZE,
    IMAGE_PROMPT,
    LLM_LOCATION,
    MAX_OUTPUT_TOKENS,
    PGVECTOR_URL,
)

PDF_MAGIC = b"%PDF-"
IMAGE_TEXT_MARK = "[이미지 설명]"

# 지시문은 사용자가 화면에서 고치므로 템플릿에 직접 넣지 않고 변수로 받음 ({ } 문자가 있어도 깨지지 않음)
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", "{instructions}\n\n[문서]\n{context}"),
        ("human", "{question}"),
    ]
)


def _vertex_credentials():
    credentials, default_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return credentials, GOOGLE_CLOUD_PROJECT or default_project


# ---------------------------------------------------------------------------
# A. 미리 준비하기 (인덱싱)
# ---------------------------------------------------------------------------


class GeminiImageReader:
    """PDF 속 이미지 → Gemini가 받아 적은 글자 + 그림 설명 텍스트.

    - LangChain의 LLMImageBlobParser는 응답 content가 문자열이라고 가정(assert)하는데,
      Gemini 3 계열은 content를 블록 리스트로 돌려줄 수 있어 msg.text로 읽도록 따로 둠
    - 결과 앞에 [이미지 설명] 표시를 붙여 청크 · 프롬프트에서 본문과 구분되게 함
    - 이미지 하나가 실패해도 PDF 전체 인덱싱이 멈추지 않도록, 재시도 후에도 실패하면 건너뛰고 failures에 기록
      (인덱싱마다 새로 만들어 써야 failures가 섞이지 않음)
    """

    def __init__(
        self,
        model: BaseChatModel,
        prompt: str = IMAGE_PROMPT,
        attempts: int = 2,
        mark: str | None = IMAGE_TEXT_MARK,
        empty_ok: bool = False,
    ):
        self.model = model
        self.prompt = prompt
        self.attempts = attempts
        self.mark = mark  # OCR처럼 결과가 페이지 본문 자체일 때는 None으로 표시 생략
        # OCR은 글자 없는 전면 사진 페이지가 있어 빈 응답이 정상일 수 있음 → 재시도 후에도 비면 실패가 아니라 빈 페이지로 인정
        self.empty_ok = empty_ok
        self.failures: list[str] = []

    def describe(self, img, label: str) -> str:
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="PNG")  # 흑백 1비트 · CMYK 등도 PNG로 통일
        image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        message = HumanMessage(
            content=[
                {"type": "text", "text": self.prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
        for attempt in range(1, self.attempts + 1):
            try:
                text = self.model.invoke([message]).text.strip()
            except Exception as e:  # 일시적인 504 · 429 등
                if attempt == self.attempts:
                    self.failures.append(f"{label} · {type(e).__name__}: {str(e)[:200]}")
                continue
            if text:
                return f"{self.mark}\n{text}" if self.mark else text
            # 작은 이미지 · 단색 이미지는 미리 걸렀으므로 빈 응답은 일시적인 현상일 수 있어 재시도,
            # 그래도 비면 조용히 사라지지 않게 기록
            if attempt == self.attempts and not self.empty_ok:
                self.failures.append(f"{label} · Gemini 빈 응답 (내용 없음으로 판단했거나 일시적 오류)")
        return ""


def extract_page_images(data: bytes, reader: GeminiImageReader) -> list[list[tuple[str, object]]]:
    """페이지별 (라벨, PIL 이미지) 목록. 아이콘 · 로고 같은 작은 이미지는 호출 비용만 들어 건너뜀.

    PyPDFLoader의 extract_images는 JPEG · 일부 무손실 형식만 꺼내고 흑백 스캔에 흔한
    CCITTFax · JBIG2 등은 조용히 건너뛰므로, 형식 해석을 Pillow에 맡기는 pypdf의 page.images를 직접 사용.
    """
    pdf = pypdf.PdfReader(io.BytesIO(data))
    pages: list[list[tuple[str, object]]] = []
    for page_no, page in enumerate(pdf.pages, start=1):
        found = []
        try:
            files = list(page.images)
        except Exception as e:  # 이미지 목록 자체를 못 읽는 손상된 페이지
            reader.failures.append(f"p.{page_no} · 이미지 목록 읽기 실패 · {type(e).__name__}: {e}")
            files = []
        for i, file in enumerate(files, start=1):
            label = f"p.{page_no} 이미지 {i}"
            try:
                img = file.image
            except Exception as e:  # 지원하지 않는 압축 형식 등
                reader.failures.append(f"{label} · 디코딩 실패 · {type(e).__name__}: {e}")
                continue
            if min(img.size) < IMAGE_MIN_SIZE:
                continue
            # 디코딩은 됐지만 한 가지 색뿐인 이미지: 빈 이미지이거나 특수 형식이 잘못 풀린 경우라
            # Gemini에 보내도 빈 응답만 오므로, 조용히 사라지지 않게 기록만 하고 건너뜀
            low, high = img.convert("L").getextrema()
            if low == high:
                reader.failures.append(f"{label} · 단색 이미지라 건너뜀 (빈 이미지이거나 형식 해석 문제)")
                continue
            found.append((label, img))
        pages.append(found)
    return pages


def _read_page_images(data: bytes, reader: GeminiImageReader) -> list[str]:
    """페이지별 이미지 설명 텍스트. 모든 이미지를 IMAGE_CONCURRENCY개씩 동시에 Gemini에 요청."""
    page_images = extract_page_images(data, reader)
    jobs = [(page_idx, label, img) for page_idx, images in enumerate(page_images) for label, img in images]
    with ThreadPoolExecutor(max_workers=IMAGE_CONCURRENCY) as pool:
        texts = list(pool.map(lambda job: reader.describe(job[2], job[1]), jobs))
    per_page = [[] for _ in page_images]
    for (page_idx, _, _), text in zip(jobs, texts, strict=True):
        if text:
            per_page[page_idx].append(text)
    return ["\n\n".join(texts) for texts in per_page]


def load_pdf(data: bytes, file_name: str, image_reader: GeminiImageReader | None = None) -> list[Document]:
    """1) 문서 읽기: PDF 바이트 → 페이지마다 Document 하나 (PDF만 허용).

    image_reader를 주면 페이지 속 이미지도 텍스트로 바꿔 그 페이지 본문 뒤에 붙임 (스캔 PDF 대응).
    """
    if not data.startswith(PDF_MAGIC):
        raise ValueError(f"PDF 파일이 아닙니다: {file_name}")

    # PyPDFLoader는 파일 경로를 받으므로 업로드 바이트를 임시 파일로 저장해서 읽음
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(data)
        tmp.flush()
        pages = PyPDFLoader(tmp.name).load()

    image_texts = _read_page_images(data, image_reader) if image_reader else []
    file_hash = file_sha256(data)
    for i, doc in enumerate(pages):
        if i < len(image_texts) and image_texts[i]:
            doc.page_content = f"{doc.page_content.rstrip()}\n\n{image_texts[i]}".lstrip()
        # source가 임시 파일 경로로 들어가므로 원래 파일명으로 교체, 페이지는 사람이 읽는 1부터
        doc.metadata = {
            "source": file_name,
            "file_sha256": file_hash,  # 같은 내용의 파일인지 판단 (청크까지 그대로 전달됨)
            "page": doc.metadata.get("page", 0) + 1,
            "total_pages": doc.metadata.get("total_pages", len(pages)),
        }
    return pages


SENTENCE_ENDINGS = (".", "?", "!", "”", '"', "’", "'", "」", ")", ">", "〉")


def _join_pages(pages: list[Document]) -> tuple[str, list[int]]:
    """페이지들을 한 문자열로 이어 붙이고 각 페이지가 시작하는 글자 위치를 함께 반환.

    앞 페이지가 문장 끝으로 끝나면 빈 줄로, 아니면(문장이 다음 페이지로 이어짐) 그대로 붙임.
    예: 5쪽 끝 '백제는 ‘욱' + 6쪽 시작 '리하郁利河’' → '욱리하郁利河’'
    """
    text, starts = "", []
    for page in pages:
        body = page.page_content.strip()
        if text and body:
            text += "\n\n" if text.endswith(SENTENCE_ENDINGS) else ""
        starts.append(len(text))
        text += body
    return text, starts


def split_documents(
    pages: list[Document], chunk_size: int, chunk_overlap: int, across_pages: bool = False,
) -> list[Document]:
    """2) 청크로 나누기: 문단 → 줄 → 문장 → 단어 순으로 끊어 chunk_size 글자 이하 조각으로.

    across_pages=True면 페이지를 이어 붙인 뒤 나눠, 페이지 경계에서 끊긴 문장도 한 청크에 들어가게 함.
    이때 청크가 걸친 페이지 범위를 page ~ page_end로 기록.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,  # 페이지 안에서 청크가 시작하는 글자 위치 (출처 표시용)
    )
    if not across_pages:
        chunks = splitter.split_documents(pages)
    else:
        text, starts = _join_pages(pages)
        chunks = splitter.create_documents([text]) if text else []
        for chunk in chunks:
            begin = chunk.metadata["start_index"]
            first = bisect.bisect_right(starts, begin) - 1
            last = bisect.bisect_right(starts, begin + len(chunk.page_content) - 1) - 1
            chunk.metadata = {
                **pages[first].metadata,
                "page_end": pages[last].metadata["page"],
                "start_index": begin - starts[first],
            }
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


EMBEDDING_CACHE_DDL = """
    CREATE TABLE IF NOT EXISTS embedding_cache (
        model       TEXT NOT NULL,
        dimension   INT NOT NULL,
        task_type   TEXT NOT NULL,
        text_sha256 TEXT NOT NULL,
        embedding   REAL[] NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (model, dimension, task_type, text_sha256)
    )
"""
DOCUMENT_TASK, QUERY_TASK = "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"


class CachedEmbeddings(Embeddings):
    """Gemini 임베딩 앞에 PostgreSQL 캐시(embedding_cache 테이블)를 둔 래퍼.

    - 키: 모델 + 차원 + task_type + 텍스트 sha256. gemini-embedding-2는 문서용(RETRIEVAL_DOCUMENT)과
      질문용(RETRIEVAL_QUERY) 벡터가 다르고 차원별로도 다르므로 모두 키에 넣음
    - 여러 텍스트는 한 번의 쿼리로 조회하고, 없는 것만 Gemini에 요청한 뒤 저장
    - gemini-embedding-2는 배치 요청("only supports one content at a time")을 거부하므로
      캐시에 없는 텍스트는 하나씩, 순서를 유지한 채 EMBEDDING_CONCURRENCY개씩 동시에 요청
    - 같은 DB를 쓰므로 두 화면 · 평가 스크립트가 캐시를 공유하고 앱을 재시작해도 남음
    """

    def __init__(self, inner: Embeddings, model: str, dimension: int, enabled: bool = True):
        self.inner = inner
        self.model = model
        self.dimension = dimension
        self.enabled = enabled
        if enabled:
            with psycopg.connect(DATABASE_URL) as conn:
                conn.execute(EMBEDDING_CACHE_DDL)

    def _lookup(self, task: str, hashes: list[str]) -> dict[str, list[float]]:
        with psycopg.connect(DATABASE_URL) as conn:
            rows = conn.execute(
                "SELECT text_sha256, embedding FROM embedding_cache"
                " WHERE model = %s AND dimension = %s AND task_type = %s AND text_sha256 = ANY(%s)",
                (self.model, self.dimension, task, hashes),
            ).fetchall()
        return dict(rows)

    def _save(self, task: str, items: list[tuple[str, list[float]]]) -> None:
        with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO embedding_cache (model, dimension, task_type, text_sha256, embedding)"
                " VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [(self.model, self.dimension, task, h, vector) for h, vector in items],
            )

    def embed_documents_with_stats(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """문서용 벡터와 캐시에서 가져온 개수."""
        hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        cached = self._lookup(DOCUMENT_TASK, list(set(hashes))) if self.enabled and texts else {}
        missing = list({h: text for h, text in zip(hashes, texts, strict=True) if h not in cached}.items())
        with ThreadPoolExecutor(max_workers=EMBEDDING_CONCURRENCY) as pool:
            fresh = list(pool.map(lambda item: self.inner.embed_documents([item[1]])[0], missing))
        computed = {h: vector for (h, _), vector in zip(missing, fresh, strict=True)}
        if self.enabled and computed:
            self._save(DOCUMENT_TASK, list(computed.items()))
        hits = sum(h in cached for h in hashes)
        return [cached.get(h) or computed[h] for h in hashes], hits

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents_with_stats(texts)[0]

    def embed_query(self, text: str) -> list[float]:
        if not self.enabled:
            return self.inner.embed_query(text)
        h = hashlib.sha256(text.encode()).hexdigest()
        if cached := self._lookup(QUERY_TASK, [h]).get(h):
            return cached
        vector = self.inner.embed_query(text)
        self._save(QUERY_TASK, [(h, vector)])
        return vector


def get_embedding_cache_stats() -> dict:
    """PostgreSQL embedding_cache 테이블의 통계(전체 개수, 문서/질문별 개수, 최신 캐시 일시 등)를 조회합니다."""
    if not EMBEDDING_CACHE:
        return {"enabled": False, "total": 0, "doc_count": 0, "query_count": 0, "last_cached_at": None}
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute(EMBEDDING_CACHE_DDL)
            row = conn.execute(
                """
                SELECT 
                    count(*) as total,
                    count(*) FILTER (WHERE task_type = 'RETRIEVAL_DOCUMENT') as doc_count,
                    count(*) FILTER (WHERE task_type = 'RETRIEVAL_QUERY') as query_count,
                    max(created_at) as last_cached_at
                FROM embedding_cache
                """
            ).fetchone()
            if row:
                return {
                    "enabled": True,
                    "total": row[0] or 0,
                    "doc_count": row[1] or 0,
                    "query_count": row[2] or 0,
                    "last_cached_at": row[3],
                }
    except Exception:
        pass
    return {"enabled": EMBEDDING_CACHE, "total": 0, "doc_count": 0, "query_count": 0, "last_cached_at": None}


def clear_embedding_cache() -> int:
    """PostgreSQL embedding_cache 테이블의 모든 캐시 데이터를 삭제합니다."""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute(EMBEDDING_CACHE_DDL)
            cur = conn.execute("DELETE FROM embedding_cache")
            return cur.rowcount
    except Exception:
        return 0


def create_embeddings(model: str, dimension: int) -> CachedEmbeddings:
    credentials, project = _vertex_credentials()
    gemini = GoogleGenerativeAIEmbeddings(
        model=model,
        vertexai=True,
        credentials=credentials,
        project=project,
        location=EMBEDDING_LOCATION,
        output_dimensionality=dimension,
    )
    return CachedEmbeddings(gemini, model, dimension, enabled=EMBEDDING_CACHE)


def embed_chunks(embeddings: Embeddings, chunks: list[Document]) -> tuple[list[list[float]], int]:
    """3) 임베딩하기: 청크 텍스트 → 벡터 (문서용 task_type). 반환: (벡터, 캐시에서 가져온 개수).

    CachedEmbeddings면 같은 텍스트의 벡터를 캐시에서 재사용하고 나머지만 Gemini에 요청.
    """
    texts = [chunk.page_content for chunk in chunks]
    if isinstance(embeddings, CachedEmbeddings):
        return embeddings.embed_documents_with_stats(texts)
    return embeddings.embed_documents(texts), 0


def get_vector_store(embeddings: Embeddings, collection_name: str) -> PGVector:
    """PostgreSQL(pgvector) 벡터 스토어. 처음 연결할 때 vector 확장과 테이블을 만든다."""
    return PGVector(
        embeddings=embeddings,
        connection=PGVECTOR_URL,
        collection_name=collection_name,
        use_jsonb=True,
    )


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def slugify(text: str, max_length: int = 24) -> str:
    """파일명 등에서 한글, 영문, 숫자를 제외한 특수문자를 언더스코어로 치환하고 길이를 제한."""
    stem = Path(text).stem
    slug = re.sub(r"[^\w\s-]", "", stem).strip()
    slug = re.sub(r"[-\s]+", "_", slug)
    if not slug:
        slug = "doc"
    return slug[:max_length].rstrip("_")


def build_collection_name(
    file_name: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_dim: int,
    image_mode: str = "",
) -> str:
    """문서명과 파이프라인 하이퍼파라미터를 결합하여 고유한 컬렉션 이름을 생성."""
    slug = slugify(file_name)
    ocr_suffix = "_ocr" if image_mode == config.IMAGE_GEMINI else ""
    return f"{slug}_c{chunk_size}_o{chunk_overlap}_d{embedding_dim}{ocr_suffix}"


def list_stored_collections() -> list[dict]:
    """DB에 저장된 pgvector 컬렉션 목록과 메타데이터, 청크 수를 반환."""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            rows = conn.execute(
                """
                SELECT c.name,
                       c.cmetadata,
                       (SELECT count(*) FROM langchain_pg_embedding e WHERE e.collection_id = c.uuid) AS stored_chunks
                FROM langchain_pg_collection c
                ORDER BY c.name
                """
            ).fetchall()
            return [
                {
                    "name": row[0],
                    "metadata": row[1] or {},
                    "stored_chunks": row[2] or 0,
                    "file_name": (row[1] or {}).get("file_name", ""),
                    "indexed_at": (row[1] or {}).get("indexed_at", ""),
                }
                for row in rows
            ]
    except Exception:
        return []


def delete_stored_collection(collection_name: str) -> bool:
    """지정한 컬렉션과 소속 임베딩을 DB에서 완전히 삭제."""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            cur = conn.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (collection_name,))
            row = cur.fetchone()
            if not row:
                return False
            coll_uuid = row[0]
            conn.execute("DELETE FROM langchain_pg_embedding WHERE collection_id = %s", (coll_uuid,))
            conn.execute("DELETE FROM langchain_pg_collection WHERE uuid = %s", (coll_uuid,))
            return True
    except Exception:
        return False


def make_index_key(file_hash: str, content_settings: dict) -> str:
    """같은 문서 판단 기준: 파일 내용 해시 + 저장 결과를 바꾸는 설정(청크 · 임베딩 · 이미지 읽기).

    파일명은 넣지 않음 — 이름만 바뀐 같은 파일은 같은 문서, 이름이 같아도 내용이 바뀌면 다른 문서.
    """
    payload = json.dumps({"file_sha256": file_hash, **content_settings}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def get_collection_info(collection_name: str) -> dict | None:
    """컬렉션 metadata(마지막 인덱싱 정보)와 실제 저장된 청크 수. 컬렉션이 없으면 None."""
    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            """
            SELECT c.cmetadata,
                   (SELECT count(*) FROM langchain_pg_embedding e WHERE e.collection_id = c.uuid)
            FROM langchain_pg_collection c
            WHERE c.name = %s
            """,  # cmetadata는 json 타입이라 GROUP BY에 쓸 수 없어 청크 수는 서브쿼리로
            (collection_name,),
        ).fetchone()
    if row is None:
        return None
    return {**(row[0] or {}), "stored_chunks": row[1]}


def find_same_index(collection_name: str, index_key: str) -> dict | None:
    """같은 내용 · 같은 설정으로 이미 온전히 인덱싱돼 있으면 그 정보를, 아니면 None.

    이미지 변환 실패가 있었거나 저장된 청크 수가 기록과 다르면(저장 도중 실패) 다시 인덱싱해야 하므로 None.
    """
    info = get_collection_info(collection_name)
    if (
        info is None
        or info.get("index_key") != index_key
        or info.get("image_failures", 0) > 0
        or info.get("chunks") != info["stored_chunks"]
    ):
        return None
    return info


def save_to_store(
    store: PGVector, chunks: list[Document], vectors: list[list[float]], index_info: dict,
) -> list[str]:
    """4) 벡터 스토어 저장: 컬렉션을 비우고 벡터 + 원문 + 출처(metadata)를 한 행으로 저장.

    - add_documents는 내부에서 다시 임베딩하므로, 3단계에서 만든 벡터를 그대로 쓰는 add_embeddings 사용
    - index_info(인덱싱 키 · 파일 해시 · 청크 수 등)는 청크 저장이 끝난 뒤 컬렉션 metadata에 기록
      → 저장 도중 실패하면 키가 남지 않아 다음 인덱싱에서 잘못 건너뛰지 않음
    """
    store.delete_collection()
    store.create_collection()
    ids = store.add_embeddings(
        texts=[chunk.page_content for chunk in chunks],
        embeddings=vectors,
        metadatas=[chunk.metadata for chunk in chunks],
    )
    info = {**index_info, "chunks": len(chunks), "indexed_at": datetime.now().isoformat(timespec="seconds")}
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            "UPDATE langchain_pg_collection SET cmetadata = %s WHERE name = %s",
            (Jsonb(info), store.collection_name),
        )
    return ids


@dataclass
class StoredRow:
    id: str
    document: str
    metadata: dict
    dimension: int
    vector_preview: str


def inspect_collection(collection_name: str, limit: int = 5) -> tuple[int, list[StoredRow]]:
    """저장 결과 확인: 컬렉션의 행 개수와 앞쪽 몇 행 (DB에 실제로 들어간 모습 그대로)."""
    sql = """
        SELECT e.id, e.document, e.cmetadata, vector_dims(e.embedding), left(e.embedding::text, 120)
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
        ORDER BY (e.cmetadata ->> 'chunk_index')::int NULLS LAST
    """
    with psycopg.connect(DATABASE_URL) as conn:
        total = conn.execute(
            """
            SELECT count(*) FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s
            """,
            (collection_name,),
        ).fetchone()[0]
        rows = conn.execute(sql + " LIMIT %s", (collection_name, limit)).fetchall()
    return total, [StoredRow(r[0], r[1], r[2], r[3], r[4] + " …") for r in rows]


# ---------------------------------------------------------------------------
# B. 질문에 답하기 (런타임)
# ---------------------------------------------------------------------------


def embed_question(embeddings: Embeddings, question: str) -> list[float]:
    """1) 질문 입력: 질문도 같은 모델로 임베딩 (embed_query는 질문용 task_type으로 요청)."""
    return embeddings.embed_query(question)


def search_chunks(store: PGVector, question_vector: list[float], k: int) -> list[tuple[Document, float]]:
    """2) 관련 청크 검색: 코사인 거리가 가까운 순으로 k개 (거리 0 = 같은 방향, 작을수록 관련 높음)."""
    return store.similarity_search_with_score_by_vector(question_vector, k=k)


def page_label(doc: Document) -> str:
    """'5' 또는 페이지를 넘어 나눈 청크면 '5~6'."""
    page, page_end = doc.metadata.get("page", "?"), doc.metadata.get("page_end")
    return f"{page}~{page_end}" if page_end and page_end != page else str(page)


def format_context(docs: list[Document]) -> str:
    """검색된 청크를 번호 · 출처와 함께 프롬프트에 넣을 문자열로."""
    return "\n\n".join(
        f"[{i}] ({doc.metadata.get('source', '?')} p.{page_label(doc)})\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def build_prompt(instructions: str, docs: list[Document], question: str) -> list[BaseMessage]:
    """3) 프롬프트 구성: 지시문 + 원문(검색된 청크) + 원래 질문."""
    return PROMPT_TEMPLATE.format_messages(
        instructions=instructions,
        context=format_context(docs),
        question=question,
    )


def create_llm(model: str, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> ChatGoogleGenerativeAI:
    credentials, project = _vertex_credentials()
    return ChatGoogleGenerativeAI(
        model=model,
        vertexai=True,
        credentials=credentials,
        project=project,
        location=LLM_LOCATION,
        reasoning_effort="low",  # 근거 문서를 읽고 정리하는 작업이라 깊은 추론은 불필요
        max_output_tokens=max_output_tokens,
        timeout=120,
        max_retries=3,
    )


def create_ollama_llm(model: str, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> BaseChatModel:
    """로컬 Ollama 모델 (chat_app과 같은 설정). Gemma 선택 시에만 필요하므로 지연 import."""
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=0, num_predict=max_output_tokens)


def create_deepseek_llm(max_output_tokens: int = MAX_OUTPUT_TOKENS) -> BaseChatModel:
    """DeepSeek V4.1 Flash (Vertex AI MaaS OpenAPI)."""
    from model_selector_chatbot.model_service import GcpMaasChatModel

    _, project = _vertex_credentials()
    proj = project or config.GOOGLE_CLOUD_PROJECT or "iceu-songpa05"
    return GcpMaasChatModel(
        model_name=config.DEEPSEEK_FLASH_MODEL,
        project_id=proj,
        region="global",
        max_tokens=max_output_tokens,
        service_name="DeepSeek V4.1 Flash API Service",
    )


def create_gemma_maas_llm(max_output_tokens: int = MAX_OUTPUT_TOKENS) -> BaseChatModel:
    """Gemma 4 26B IT (Vertex AI MaaS OpenAPI, Thinking 지원)."""
    from model_selector_chatbot.model_service import GcpMaasChatModel

    _, project = _vertex_credentials()
    proj = project or config.GOOGLE_CLOUD_PROJECT or "iceu-songpa05"
    return GcpMaasChatModel(
        model_name=config.GEMMA_26B_MODEL,
        project_id=proj,
        region="global",
        max_tokens=max_output_tokens,
        enable_thinking=True,
        service_name="Gemma 4 26B IT API Service",
    )


def create_gemini_flash_lite_llm(max_output_tokens: int = MAX_OUTPUT_TOKENS) -> BaseChatModel:
    """Gemini 3.5 Flash Lite (Vertex AI)."""
    from gemini_flash_lite_chatbot.config import GeminiConfig
    from gemini_flash_lite_chatbot.gemini_service import GeminiFlashLiteService

    _, project = _vertex_credentials()
    proj = project or config.GOOGLE_CLOUD_PROJECT or "iceu-songpa05"
    cfg = GeminiConfig(
        model_name=config.GEMINI_FLASH_LITE_MODEL,
        project_id=proj,
        location=config.LLM_LOCATION,
        max_output_tokens=max_output_tokens,
        reasoning_effort="low",
    )
    return GeminiFlashLiteService(config=cfg)


def create_claude_haiku_llm(max_output_tokens: int = MAX_OUTPUT_TOKENS) -> BaseChatModel:
    """Anthropic Claude Haiku 5.5 (Vertex AI Anthropic SDK)."""
    from claude_haiku_chatbot.claude_service import ClaudeHaikuVertexChatModel
    from claude_haiku_chatbot.config import ClaudeHaikuConfig

    _, project = _vertex_credentials()
    proj = project or config.GOOGLE_CLOUD_PROJECT or "iceu-songpa05"
    cfg = ClaudeHaikuConfig(
        model_name=config.CLAUDE_HAIKU_MODEL,
        project_id=proj,
        region="global",
        max_tokens=max_output_tokens,
    )
    return ClaudeHaikuVertexChatModel(config=cfg)


def create_answer_llm(
    choice: str,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    gemini_model: str = config.LLM_MODEL,
) -> BaseChatModel:
    """사이드바에서 선택된 모델 종류(choice)에 맞는 LLM 객체를 반환합니다."""
    if choice == config.LLM_DEEPSEEK_FLASH:
        return create_deepseek_llm(max_output_tokens)
    if choice == config.LLM_GEMMA_26B:
        return create_gemma_maas_llm(max_output_tokens)
    if choice == config.LLM_GEMINI_FLASH_LITE:
        return create_gemini_flash_lite_llm(max_output_tokens)
    if choice == config.LLM_CLAUDE_HAIKU:
        return create_claude_haiku_llm(max_output_tokens)
    if choice == config.LLM_OLLAMA_GEMMA:
        return create_ollama_llm(config.OLLAMA_MODEL, max_output_tokens)
    return create_llm(gemini_model, max_output_tokens)


def generate_answer(llm: BaseChatModel, messages: list[BaseMessage]):
    """4) LLM → 답변: 구성된 프롬프트를 그대로 보내 AIMessage(본문 + 토큰 사용량)를 받음."""
    return llm.invoke(messages)


def build_rag_chain(store: PGVector, llm: BaseChatModel, instructions: str, k: int) -> Runnable:
    """B의 1~4를 하나로 연결한 LCEL 체인. 입력: 질문 문자열 → 출력: 답변 문자열.

    - context: 질문 → retriever(질문 임베딩 + 벡터 검색) → 청크 문자열
    - question: RunnablePassthrough로 원래 질문을 그대로 프롬프트에 전달 (그림의 "원래 질문도 그대로 전달")
    """
    retriever = store.as_retriever(search_kwargs={"k": k})
    return (
        {"context": retriever | RunnableLambda(format_context), "question": RunnablePassthrough()}
        | PROMPT_TEMPLATE.partial(instructions=instructions)
        | llm
        | StrOutputParser()
    )


def pdf_name_is_valid(file_name: str) -> bool:
    return Path(file_name).suffix.lower() == ".pdf"
