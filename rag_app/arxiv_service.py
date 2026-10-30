"""arXiv 논문 검색 → PDF 다운로드 → pgvector 저장 → 질의 (화면 코드와 분리한 서비스 모듈).

- 모든 논문을 한 컬렉션(config.ARXIV_COLLECTION)에 누적하고 청크 metadata의 arxiv_id로 구분
- 컬렉션 metadata의 papers에 논문별 인덱싱 기록(index_key · 청크 수)을 남겨 같은 논문은 다시 저장하지 않음
- 외부 의존(arXiv 클라이언트 · HTTP · 임베딩 · LLM)은 인자로 받아 테스트에서 가짜로 바꿀 수 있게 함
- 저장: 텍스트 레이어가 없는 스캔 페이지만 Gemini로 OCR (R8)
- 질의: 의미 검색(PGVector) + 키워드 검색(BM25)을 가중 RRF로 합쳐 상위 ARXIV_TOP_K개 (R9 · R10)
"""

import hashlib
import re
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import arxiv
import psycopg
import requests
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from psycopg.types.json import Jsonb

from rag_app import config
from rag_app import pipeline as rag

NO_HITS_ANSWER = "저장된 논문에서 관련 내용을 찾지 못했습니다."


@dataclass(frozen=True)
class Paper:
    arxiv_id: str  # 버전 포함 short id (예: 2401.01234v2, 옛 형식은 hep-th/9901001v1)
    title: str
    authors: tuple[str, ...]
    published: str  # YYYY-MM-DD
    summary: str
    pdf_url: str | None


@dataclass(frozen=True)
class IndexResult:
    arxiv_id: str
    skipped: bool  # 같은 내용 · 같은 설정으로 이미 저장돼 있어 건너뜀
    chunks: int
    pdf_kind: str = "text"  # text · image(스캔) · mixed (R8)
    ocr_pages: tuple[int, ...] = ()  # OCR한 페이지 번호
    ocr_failures: tuple[str, ...] = ()  # LLM OCR 실패 — 다음 저장에서 다시 시도
    ocr_warnings: tuple[str, ...] = ()  # 이미지 추출 문제(디코딩 실패 · 단색 등) — 다시 해도 같으므로 표시만


def _one_line(text: str) -> str:
    # arXiv 제목 · 요약은 줄바꿈과 겹친 공백이 섞여 옴
    return " ".join(text.split())


def search_papers(
    query: str, max_results: int = config.ARXIV_MAX_RESULTS, client: arxiv.Client | None = None,
) -> list[Paper]:
    """R1 검색: 키워드 → 관련도순 논문 목록. arxiv.Client가 3초 간격 · 3회 재시도를 맡음."""
    query = query.strip()
    if not query:
        return []
    client = client or arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results)
    return [
        Paper(
            arxiv_id=result.get_short_id(),
            title=_one_line(result.title),
            authors=tuple(author.name for author in result.authors),
            published=result.published.date().isoformat(),
            summary=_one_line(result.summary),
            pdf_url=result.pdf_url,
        )
        for result in client.results(search)
    ]


def pdf_path_for(arxiv_id: str, data_dir: Path = config.ARXIV_DATA_DIR) -> Path:
    # 옛 형식 ID의 '/'는 파일명에 쓸 수 없어 '_'로
    return data_dir / f"{arxiv_id.replace('/', '_')}.pdf"


def paper_from_record(record: dict) -> Paper:
    """저장 기록 → Paper (받아 둔 PDF가 없어도 다시 받을 수 있게 PDF 주소를 채움)."""
    arxiv_id = record["arxiv_id"]
    return Paper(
        arxiv_id=arxiv_id,
        title=record.get("title", ""),
        authors=tuple(record.get("authors", ())),
        published=record.get("published", ""),
        summary="",
        pdf_url=record.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}",
    )


def _http_get(url: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def download_pdf(
    paper: Paper, data_dir: Path = config.ARXIV_DATA_DIR, http_get: Callable[[str], bytes] = _http_get,
) -> Path:
    """R2 다운로드: 이미 받은 PDF는 재사용. 임시 파일에 다 쓴 뒤 이름을 바꿔 받다 만 파일이 남지 않게 함."""
    path = pdf_path_for(paper.arxiv_id, data_dir)
    if path.exists():
        with path.open("rb") as f:
            if f.read(len(rag.PDF_MAGIC)) == rag.PDF_MAGIC:
                return path
    if not paper.pdf_url:
        raise ValueError(f"PDF 링크가 없는 논문입니다: {paper.arxiv_id}")

    data = http_get(paper.pdf_url)
    if not data.startswith(rag.PDF_MAGIC):
        raise ValueError(f"PDF가 아닌 응답입니다: {paper.pdf_url}")
    data_dir.mkdir(parents=True, exist_ok=True)
    # 같은 논문을 동시에 받아도 서로의 임시 파일을 덮어쓰지 않도록 임시 파일명은 매번 다르게
    with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".part", delete=False) as tmp:
        tmp.write(data)
    Path(tmp.name).replace(path)
    return path


# ---------------------------------------------------------------------------
# R8 페이지별 OCR
# ---------------------------------------------------------------------------
OCR_CACHE_DIR = config.ARXIV_DATA_DIR / "ocr_pages"


@dataclass
class _PdfPages:
    pages: list[Document]  # 페이지별 텍스트 레이어 (OCR 전)
    page_images: list[list[tuple[str, object]]]  # 스캔 후보 페이지만 채움, 나머지는 빈 목록
    scanned: list[int]  # OCR할 페이지 인덱스 (0부터)
    warnings: list[str]  # 스캔 후보 페이지의 이미지 추출 문제 (재시도해도 같음)
    failures: list[str]  # LLM OCR 실패 (일시적일 수 있어 다음 저장에서 재시도)

    @property
    def kind(self) -> str:
        if not self.scanned:
            return "text"
        return "image" if len(self.scanned) == len(self.pages) else "mixed"


def _analyze_pdf(data: bytes, file_name: str) -> _PdfPages:
    """LLM 호출 없이 페이지마다 텍스트 레이어 글자 수를 보고, 적은 페이지만 이미지를 꺼내 스캔 페이지를 고름.

    텍스트 PDF는 그림 이미지를 디코딩하지도 않음 (논문 그림이 많아도 비용 없음).
    """
    pages = rag.load_pdf(data, file_name)
    low_text = {i for i, page in enumerate(pages) if len(page.page_content.strip()) < config.ARXIV_OCR_MIN_CHARS}
    if not low_text:
        return _PdfPages(pages, [[] for _ in pages], [], [], [])
    collector = rag.GeminiImageReader(model=None)  # 이미지 추출 실패 기록용 (LLM은 부르지 않음)
    page_images = rag.extract_page_images(data, collector)
    page_images = [images if i in low_text else [] for i, images in enumerate(page_images)]
    # 글자가 충분한 페이지의 그림 디코딩 실패는 OCR과 무관하므로 버림
    warnings = [f for f in collector.failures if int(re.match(r"p\.(\d+)", f).group(1)) - 1 in low_text]
    scanned = [i for i in sorted(low_text) if page_images[i]]
    return _PdfPages(pages, page_images, scanned, warnings, [])


def _ocr_cache_path(images: list[tuple[str, object]], model: str) -> Path:
    """페이지 이미지 픽셀 + 모델 + 프롬프트 해시 (evaluation.ocr_pdf와 같은 방식, 논문 전용 폴더)."""
    digest = hashlib.sha256(f"{model}|{config.ARXIV_OCR_PROMPT}".encode())
    for _, img in images:
        digest.update(hashlib.sha256(img.tobytes()).digest())
    return OCR_CACHE_DIR / f"{digest.hexdigest()[:24]}.txt"


def _model_name(llm: BaseChatModel) -> str:
    return getattr(llm, "model", None) or type(llm).__name__


def _run_ocr(pdf: _PdfPages, ocr_llm: BaseChatModel) -> None:
    """스캔 페이지만 OCR해 pdf.pages 본문을 채우고 실패는 pdf.failures에. 성공한 페이지는 캐시해 재시도 때 재사용."""
    reader = rag.GeminiImageReader(ocr_llm, config.ARXIV_OCR_PROMPT, mark=None, empty_ok=True)
    model = _model_name(ocr_llm)

    def read_page(index: int) -> tuple[int, str]:
        images = pdf.page_images[index]
        cache = _ocr_cache_path(images, model)
        if cache.exists():
            return index, cache.read_text()
        before = len(reader.failures)
        text = "\n\n".join(t for label, img in images if (t := reader.describe(img, label)))
        if len(reader.failures) == before:  # 이 페이지에서 실패가 없을 때만 저장
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text)
        return index, text

    with ThreadPoolExecutor(max_workers=config.IMAGE_CONCURRENCY) as pool:
        for index, text in pool.map(read_page, pdf.scanned):
            pdf.pages[index].page_content = f"{pdf.pages[index].page_content.strip()}\n\n{text}".strip()
    pdf.failures.extend(reader.failures)


def _index_settings(ocr_model: str | None = None) -> dict:
    """저장 결과를 바꾸는 설정 (index_key 계산용). OCR한 페이지가 있을 때만 OCR 모델 · 프롬프트 포함
    (텍스트 PDF는 OCR 설정이 바뀌어도 다시 저장할 필요가 없음)."""
    settings = {
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "across_pages": config.CHUNK_ACROSS_PAGES,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dim": config.EMBEDDING_DIMENSION,
    }
    if ocr_model:
        settings |= {"ocr_model": ocr_model, "ocr_prompt": config.ARXIV_OCR_PROMPT}
    return settings


@contextmanager
def _paper_lock(collection_name: str, arxiv_id: str) -> Iterator[None]:
    """같은 논문의 저장을 한 번에 하나씩 (다른 논문은 병렬).

    PGVector.add_embeddings가 자체 세션을 쓰므로 삭제 · 저장 · 기록을 한 트랜잭션으로 묶을 수 없어
    트랜잭션과 무관한 세션 단위 advisory lock을 저장 내내 잡아 둠.
    """
    key = f"{collection_name}:{arxiv_id}"
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(hashtext(%s))", (key,))
        try:
            yield
        finally:
            conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (key,))


def list_indexed_papers(collection_name: str = config.ARXIV_COLLECTION) -> list[dict]:
    """저장된 논문 기록 (최근 저장순). 컬렉션이 없으면 빈 목록."""
    info = rag.get_collection_info(collection_name) or {}
    papers = [{"arxiv_id": arxiv_id, **record} for arxiv_id, record in info.get(rag.PAPER_REGISTRY_KEY, {}).items()]
    return sorted(papers, key=lambda p: p.get("indexed_at", ""), reverse=True)


def _count_paper_chunks(collection_name: str, arxiv_id: str, generation: str) -> int:
    """그 세대의 청크 수 (실패한 저장이 남긴 다른 세대 청크는 세지 않음)."""
    with psycopg.connect(config.DATABASE_URL) as conn:
        return conn.execute(
            """
            SELECT count(*) FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s AND e.cmetadata @> %s
            """,  # @>는 PGVector가 만든 cmetadata GIN 인덱스를 씀
            (collection_name, Jsonb({"arxiv_id": arxiv_id, "generation": generation})),
        ).fetchone()[0]


def _find_same_paper(collection_name: str, arxiv_id: str, index_key: str) -> dict | None:
    """같은 내용 · 설정으로 온전히 저장돼 있으면 그 기록.

    청크 수가 기록과 다르거나(저장 도중 실패) LLM OCR 실패 페이지가 있었으면 None → 다시 저장 (OCR 재시도)."""
    record = _paper_record(collection_name, arxiv_id)
    if (
        record is None
        or record.get("index_key") != index_key
        or record.get("ocr_failures")
        or "generation" not in record  # 세대 도입 전 기록 (backfill_generation 전) → 다시 저장
        or record.get("chunks") != _count_paper_chunks(collection_name, arxiv_id, record["generation"])
    ):
        return None
    return record


def _paper_record(collection_name: str, arxiv_id: str) -> dict | None:
    return next((p for p in list_indexed_papers(collection_name) if p["arxiv_id"] == arxiv_id), None)


def _delete_paper_chunks(collection_name: str, arxiv_id: str, keep_generation: str | None) -> None:
    """그 논문의 청크 중 keep_generation 세대가 아닌 것을 삭제.

    keep_generation=None(세대 없는 옛 기록이 살아 있을 때)이면 세대가 붙은 청크만 지우고 세대 없는 옛 청크는 남김
    — IS DISTINCT FROM NULL은 세대가 있는 행만 참. 논문을 통째로 지울 때는 _delete_all_paper_chunks.
    """
    with psycopg.connect(config.DATABASE_URL) as conn:
        conn.execute(
            """
            DELETE FROM langchain_pg_embedding e USING langchain_pg_collection c
            WHERE e.collection_id = c.uuid AND c.name = %s AND e.cmetadata @> %s
              AND e.cmetadata->>'generation' IS DISTINCT FROM %s
            """,
            (collection_name, Jsonb({"arxiv_id": arxiv_id}), keep_generation),
        )


def _delete_all_paper_chunks(collection_name: str, arxiv_id: str) -> None:
    """세대와 상관없이 그 논문의 청크를 모두 삭제 (세대 id 도입 전 청크 포함)."""
    with psycopg.connect(config.DATABASE_URL) as conn:
        conn.execute(
            """
            DELETE FROM langchain_pg_embedding e USING langchain_pg_collection c
            WHERE e.collection_id = c.uuid AND c.name = %s AND e.cmetadata @> %s
            """,
            (collection_name, Jsonb({"arxiv_id": arxiv_id})),
        )


def _record_paper(collection_name: str, arxiv_id: str, record: dict) -> None:
    """컬렉션 metadata의 papers에 논문 하나를 한 문장으로 병합 (다른 논문 기록과 동시에 갱신돼도 유실 없음).

    PGVector가 만든 컬렉션의 cmetadata는 SQL NULL이 아니라 JSON null이라 nullif로 바꾼 뒤 coalesce
    (그대로 두면 null || 객체가 배열이 됨).
    """
    with psycopg.connect(config.DATABASE_URL) as conn:
        conn.execute(
            """
            UPDATE langchain_pg_collection
            SET cmetadata = (
                coalesce(nullif(cmetadata::jsonb, 'null'::jsonb), '{}'::jsonb)
                || jsonb_build_object(
                    'papers',
                    coalesce(cmetadata::jsonb -> 'papers', '{}'::jsonb) || jsonb_build_object(%s::text, %s::jsonb)
                )
            )::json
            WHERE name = %s
            """,
            (arxiv_id, Jsonb(record), collection_name),
        )


def index_paper(
    paper: Paper,
    pdf_path: Path,
    embeddings: Embeddings,
    collection_name: str = config.ARXIV_COLLECTION,
    ocr_llm: BaseChatModel | None = None,
) -> IndexResult:
    """R3 저장: 페이지 분석 → 논문 잠금 → 이미 저장? → (스캔 페이지 OCR) → 청크 · 임베딩
    → 새 세대 저장 → 기록 교체 → 옛 세대 삭제.

    저장할 때마다 새 generation id를 청크 · 기록에 붙이고 질의는 기록된 세대만 검색하므로, 어느 단계에서
    실패해도 이전 버전은 그대로 질의되고 다음 시도에서 새 세대를 처음부터 다시 저장함 (R6).
    index_key는 "같은 입력인가"(건너뛰기)만 판단 — OCR 재시도처럼 같은 index_key로 다시 저장해도 세대는 새로 만듦.
    ocr_llm: 스캔 페이지가 있을 때 OCR에 쓸 모델 (R8). 텍스트 PDF면 부르지 않음.
    """
    data = pdf_path.read_bytes()
    pdf = _analyze_pdf(data, pdf_path.name)  # LLM 호출 없음 — OCR 여부를 알아야 index_key를 정할 수 있음
    if pdf.scanned and ocr_llm is None:
        raise ValueError(f"스캔 페이지 {len(pdf.scanned)}쪽을 OCR할 모델이 필요합니다: {paper.arxiv_id}")
    ocr_model = _model_name(ocr_llm) if pdf.scanned else None
    index_key = rag.make_index_key(rag.file_sha256(data), _index_settings(ocr_model))
    store = rag.get_vector_store(embeddings, collection_name)  # 컬렉션이 없으면 만듦

    with _paper_lock(collection_name, paper.arxiv_id):
        same = _find_same_paper(collection_name, paper.arxiv_id, index_key)
        if same:
            return IndexResult(
                paper.arxiv_id, skipped=True, chunks=same["chunks"],
                pdf_kind=same.get("pdf_kind", "text"), ocr_pages=tuple(same.get("ocr_pages", ())),
                ocr_warnings=tuple(same.get("ocr_warnings", ())),
            )
        live = _paper_record(collection_name, paper.arxiv_id)
        generation = uuid.uuid4().hex

        if pdf.scanned:
            _run_ocr(pdf, ocr_llm)
        chunks = rag.split_documents(pdf.pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.CHUNK_ACROSS_PAGES)
        if not chunks:
            raise ValueError(f"PDF에서 텍스트를 추출하지 못했습니다: {paper.arxiv_id}")
        for chunk in chunks:
            # source는 근거 표시에 쓰이므로 파일명 대신 논문 제목
            chunk.metadata.update(source=paper.title, arxiv_id=paper.arxiv_id, index_key=index_key, generation=generation)
        vectors, _ = rag.embed_chunks(embeddings, chunks)

        # 지금 검색되는 세대만 남기고 이전 실패가 남긴 청크 정리 (기록이 없으면 모두)
        _delete_paper_chunks(collection_name, paper.arxiv_id, live.get("generation") if live else None)
        store.add_embeddings(
            texts=[chunk.page_content for chunk in chunks],
            embeddings=vectors,
            metadatas=[chunk.metadata for chunk in chunks],
        )
        ocr_pages = tuple(i + 1 for i in pdf.scanned)
        _record_paper(collection_name, paper.arxiv_id, {
            "title": paper.title,
            "authors": list(paper.authors),
            "published": paper.published,
            "pdf_url": paper.pdf_url,
            "index_key": index_key,
            "generation": generation,
            "chunks": len(chunks),
            "pdf_kind": pdf.kind,
            "ocr_pages": list(ocr_pages),
            "ocr_failures": pdf.failures[:20],  # 있으면 다음 저장에서 다시 OCR
            "ocr_warnings": pdf.warnings[:20],  # 다시 해도 같으므로 건너뛰기를 막지 않음
            "indexed_at": datetime.now().isoformat(timespec="seconds"),
        })
        # 기록이 새 세대를 가리킨 뒤에 옛 세대를 지움 — 여기서 실패해도 옛 청크는 검색되지 않음
        _delete_paper_chunks(collection_name, paper.arxiv_id, generation)
    return IndexResult(
        paper.arxiv_id, skipped=False, chunks=len(chunks), pdf_kind=pdf.kind, ocr_pages=ocr_pages,
        ocr_failures=tuple(pdf.failures), ocr_warnings=tuple(pdf.warnings),
    )


# ---------------------------------------------------------------------------
# R9 · R10 하이브리드 질의
# ---------------------------------------------------------------------------
_HANGUL = re.compile(r"[가-힣]")
_BM25_CACHE: dict[tuple, BM25Retriever] = {}
_BM25_CACHE_MAX = 8


def _tokenize(text: str) -> list[str]:
    # 기본 str.split은 구두점이 붙어 "attention,"과 "attention"을 다른 단어로 봄
    return re.findall(r"\w+", text.lower())


def _paper_generations(papers: list[dict]) -> list[dict]:
    """논문마다 자기 세대만 고르는 조건 (arxiv_id IN … AND generation IN …는 다른 논문의 세대까지 섞일 수 있음)."""
    return [{"arxiv_id": p["arxiv_id"], "generation": p["generation"]} for p in papers]


def _load_chunks(collection_name: str, papers: list[dict]) -> list[Document]:
    with psycopg.connect(config.DATABASE_URL) as conn:
        rows = conn.execute(
            """
            SELECT e.document, e.cmetadata FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s AND e.cmetadata @> ANY(%s::jsonb[])
            """,
            (collection_name, [Jsonb(g) for g in _paper_generations(papers)]),
        ).fetchall()
    return [Document(page_content=text, metadata=metadata) for text, metadata in rows]


def _bm25_retriever(collection_name: str, papers: list[dict]) -> BM25Retriever | None:
    """대상 논문 · 세대가 같으면 캐시한 BM25 인덱스를 재사용. 다시 저장해 세대가 바뀌면 키가 달라져 새로 만듦."""
    key = (collection_name, frozenset((p["arxiv_id"], p["generation"]) for p in papers))
    if key not in _BM25_CACHE:
        docs = _load_chunks(collection_name, papers)
        if not docs:
            return None
        if len(_BM25_CACHE) >= _BM25_CACHE_MAX:
            _BM25_CACHE.pop(next(iter(_BM25_CACHE)))  # 가장 오래된 것부터
        _BM25_CACHE[key] = BM25Retriever.from_documents(docs, preprocess_func=_tokenize, k=config.ARXIV_CANDIDATES)
    return _BM25_CACHE[key]


def keyword_query(question: str, llm: BaseChatModel) -> str:
    """BM25용 질문. 한글이 있으면 LLM으로 영어 검색 키워드를 만들고, 없으면 그대로 (추가 호출 없음)."""
    if not _HANGUL.search(question):
        return question
    try:
        keywords = llm.invoke(config.ARXIV_KEYWORD_PROMPT.format(question=question)).text.strip()
    except Exception:  # 429 등 — 키워드 검색이 약해질 뿐 의미 검색은 되므로 질의 전체를 실패시키지 않음
        return question
    return keywords or question


def ask(
    question: str,
    llm: BaseChatModel,
    embeddings: Embeddings,
    collection_name: str = config.ARXIV_COLLECTION,
    arxiv_ids: list[str] | None = None,
    k: int = config.ARXIV_TOP_K,
) -> tuple[str, list[tuple[Document, dict]]]:
    """R4 질의: 저장된 논문 전체 또는 arxiv_ids만 대상으로 하이브리드 검색해 답변.

    반환: (답변, [(근거 청크, {"rrf_rank", "vector_rank", "bm25_rank"})]) — 순위가 None이면 그 검색에선 못 찾음.
    기록된 세대(generation)의 청크만 검색해 저장 도중 실패로 남은 청크는 근거로 쓰지 않음.
    """
    papers = [
        p for p in list_indexed_papers(collection_name)
        if (not arxiv_ids or p["arxiv_id"] in arxiv_ids) and "generation" in p  # 세대 없는 옛 기록은 이전 전까지 제외
    ]
    if not papers:
        return NO_HITS_ANSWER, []

    store = rag.get_vector_store(embeddings, collection_name)
    vector = store.as_retriever(search_kwargs={
        "k": config.ARXIV_CANDIDATES,
        "filter": {"$or": [{"$and": [{f: {"$eq": v}} for f, v in g.items()]} for g in _paper_generations(papers)]},
    })
    vector_docs = vector.invoke(question)  # 의미 검색은 원문 질문 그대로
    bm25 = _bm25_retriever(collection_name, papers)
    bm25_docs = bm25.invoke(keyword_query(question, llm)) if bm25 else []

    # EnsembleRetriever.invoke는 두 리트리버에 같은 질문을 넘기므로, 따로 검색한 결과를 가중 RRF로만 합침
    if bm25:
        fused = EnsembleRetriever(
            retrievers=[bm25, vector], weights=list(config.ARXIV_HYBRID_WEIGHTS),
        ).weighted_reciprocal_rank([bm25_docs, vector_docs])[:k]
    else:
        fused = vector_docs[:k]
    if not fused:
        return NO_HITS_ANSWER, []

    def rank_in(docs: list[Document], doc: Document) -> int | None:
        return next((i for i, d in enumerate(docs, start=1) if d.page_content == doc.page_content), None)

    hits = [
        (doc, {"rrf_rank": i, "vector_rank": rank_in(vector_docs, doc), "bm25_rank": rank_in(bm25_docs, doc)})
        for i, doc in enumerate(fused, start=1)
    ]
    reply = rag.generate_answer(llm, rag.build_prompt(config.SYSTEM_PROMPT, fused, question))
    return reply.text, hits


# ---------------------------------------------------------------------------
# R11 삭제 · R12 기존 데이터 이전
# ---------------------------------------------------------------------------
def delete_paper(arxiv_id: str, collection_name: str = config.ARXIV_COLLECTION) -> bool:
    """R11: 논문 기록 → 모든 세대 청크 순으로 삭제 (받아 둔 PDF는 캐시로 남김). 기록이 없었으면 False.

    기록을 먼저 지워 목록 · 질의에서 바로 사라지게 하고, 청크 삭제가 실패해도 기록이 없어 검색되지 않음.
    """
    with _paper_lock(collection_name, arxiv_id):
        with psycopg.connect(config.DATABASE_URL) as conn:
            removed = conn.execute(
                """
                UPDATE langchain_pg_collection
                SET cmetadata = (cmetadata::jsonb #- ARRAY['papers', %s])::json
                WHERE name = %s AND cmetadata::jsonb -> 'papers' ? %s
                """,
                (arxiv_id, collection_name, arxiv_id),
            ).rowcount
        _delete_all_paper_chunks(collection_name, arxiv_id)
    return removed > 0


def _backfill_generation(conn: psycopg.Connection, collection_uuid, papers: dict) -> None:
    """세대 id가 없던 기록 · 청크에 generation = index_key를 채움 (index_key가 그 저장의 세대였으므로)."""
    for arxiv_id, record in papers.items():
        conn.execute(
            """
            UPDATE langchain_pg_embedding
            SET cmetadata = cmetadata || jsonb_build_object('index_key', %s::text, 'generation', %s::text)
            WHERE collection_id = %s AND cmetadata @> %s AND NOT cmetadata ? 'generation'
            """,
            (record["index_key"], record["index_key"], collection_uuid, Jsonb({"arxiv_id": arxiv_id})),
        )
    filled = {i: {"generation": r["index_key"], **r} for i, r in papers.items()}
    conn.execute(
        """
        UPDATE langchain_pg_collection
        SET cmetadata = (cmetadata::jsonb || jsonb_build_object('papers', %s::jsonb))::json WHERE uuid = %s
        """,
        (Jsonb(filled), collection_uuid),
    )


def migrate_legacy_collection(old_name: str = "arxiv_papers", new_name: str = config.ARXIV_COLLECTION) -> int:
    """R12: 세대 표시 전 이름으로 저장된 논문 컬렉션을 재임베딩 없이 새 이름 · 규칙으로 옮김. 반환: 옮긴 논문 수.

    한 트랜잭션에서 컬렉션 이름을 바꾸고 청크 · 기록에 세대를 채움 — 도중에 실패하면 전부 롤백.
    """
    with psycopg.connect(config.DATABASE_URL) as conn, conn.transaction():
        if conn.execute("SELECT 1 FROM langchain_pg_collection WHERE name = %s", (new_name,)).fetchone():
            raise ValueError(f"`{new_name}` 컬렉션이 이미 있어 옮기지 않았습니다.")
        row = conn.execute(
            "SELECT uuid, cmetadata FROM langchain_pg_collection WHERE name = %s FOR UPDATE", (old_name,),
        ).fetchone()
        if row is None:
            return 0
        uuid_, metadata = row
        papers = (metadata or {}).get(rag.PAPER_REGISTRY_KEY, {})
        conn.execute("UPDATE langchain_pg_collection SET name = %s WHERE uuid = %s", (new_name, uuid_))
        _backfill_generation(conn, uuid_, papers)
    return len(papers)


def backfill_generation(collection_name: str = config.ARXIV_COLLECTION) -> int:
    """generation 도입 전에 저장 · 이전된 논문에 세대를 채움 (한 트랜잭션). 반환: 채운 논문 수."""
    with psycopg.connect(config.DATABASE_URL) as conn, conn.transaction():
        row = conn.execute(
            "SELECT uuid, cmetadata FROM langchain_pg_collection WHERE name = %s FOR UPDATE", (collection_name,),
        ).fetchone()
        if row is None:
            return 0
        papers = (row[1] or {}).get(rag.PAPER_REGISTRY_KEY, {})
        missing = {i: r for i, r in papers.items() if "generation" not in r}
        if missing:
            _backfill_generation(conn, row[0], {**papers, **missing})
    return len(missing)
