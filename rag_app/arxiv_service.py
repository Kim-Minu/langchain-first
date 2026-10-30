"""arXiv 논문 검색 → PDF 다운로드 → pgvector 저장 → 질의 (화면 코드와 분리한 서비스 모듈).

- 모든 논문을 한 컬렉션(config.ARXIV_COLLECTION)에 누적하고 청크 metadata의 arxiv_id로 구분
- 컬렉션 metadata의 papers에 논문별 인덱싱 기록(index_key · 청크 수)을 남겨 같은 논문은 다시 저장하지 않음
- 외부 의존(arXiv 클라이언트 · HTTP · 임베딩 · LLM)은 인자로 받아 테스트에서 가짜로 바꿀 수 있게 함
"""

import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import arxiv
import psycopg
import requests
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


def _index_settings() -> dict:
    """저장 결과를 바꾸는 설정 (index_key 계산용)."""
    return {
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "across_pages": config.CHUNK_ACROSS_PAGES,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dim": config.EMBEDDING_DIMENSION,
    }


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


def _count_paper_chunks(collection_name: str, arxiv_id: str, index_key: str) -> int:
    """index_key 세대의 청크 수 (실패한 저장이 남긴 다른 세대 청크는 세지 않음)."""
    with psycopg.connect(config.DATABASE_URL) as conn:
        return conn.execute(
            """
            SELECT count(*) FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s AND e.cmetadata @> %s
            """,  # @>는 PGVector가 만든 cmetadata GIN 인덱스를 씀
            (collection_name, Jsonb({"arxiv_id": arxiv_id, "index_key": index_key})),
        ).fetchone()[0]


def _find_same_paper(collection_name: str, arxiv_id: str, index_key: str) -> dict | None:
    """같은 내용 · 설정으로 온전히 저장돼 있으면 그 기록. 청크 수가 기록과 다르면(저장 도중 실패) None."""
    record = next((p for p in list_indexed_papers(collection_name) if p["arxiv_id"] == arxiv_id), None)
    if (
        record is None
        or record.get("index_key") != index_key
        or record.get("chunks") != _count_paper_chunks(collection_name, arxiv_id, index_key)
    ):
        return None
    return record


def _delete_paper_chunks(collection_name: str, arxiv_id: str, index_key: str, keep: bool) -> None:
    """keep=False면 index_key 세대만, keep=True면 index_key 세대를 뺀 나머지(옛 세대)를 삭제."""
    with psycopg.connect(config.DATABASE_URL) as conn:
        conn.execute(
            f"""
            DELETE FROM langchain_pg_embedding e USING langchain_pg_collection c
            WHERE e.collection_id = c.uuid AND c.name = %s AND e.cmetadata @> %s
              AND e.cmetadata->>'index_key' IS {"DISTINCT" if keep else "NOT DISTINCT"} FROM %s
            """,
            (collection_name, Jsonb({"arxiv_id": arxiv_id}), index_key),
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
    paper: Paper, pdf_path: Path, embeddings: Embeddings, collection_name: str = config.ARXIV_COLLECTION,
) -> IndexResult:
    """R3 저장: 논문 잠금 → 이미 저장? → 청크 · 임베딩 → 새 세대 저장 → 기록 교체 → 옛 세대 삭제.

    청크마다 index_key(세대)를 붙이고 질의는 기록된 세대만 검색하므로, 어느 단계에서 실패해도
    이전 버전은 그대로 질의되고 다음 시도에서 새 세대를 처음부터 다시 저장함 (R6).
    """
    data = pdf_path.read_bytes()
    index_key = rag.make_index_key(rag.file_sha256(data), _index_settings())
    store = rag.get_vector_store(embeddings, collection_name)  # 컬렉션이 없으면 만듦

    with _paper_lock(collection_name, paper.arxiv_id):
        same = _find_same_paper(collection_name, paper.arxiv_id, index_key)
        if same:
            return IndexResult(paper.arxiv_id, skipped=True, chunks=same["chunks"])

        pages = rag.load_pdf(data, pdf_path.name)
        chunks = rag.split_documents(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.CHUNK_ACROSS_PAGES)
        if not chunks:
            raise ValueError(f"PDF에서 텍스트를 추출하지 못했습니다: {paper.arxiv_id}")
        for chunk in chunks:
            # source는 근거 표시에 쓰이므로 파일명 대신 논문 제목
            chunk.metadata.update(source=paper.title, arxiv_id=paper.arxiv_id, index_key=index_key)
        vectors, _ = rag.embed_chunks(embeddings, chunks)

        _delete_paper_chunks(collection_name, paper.arxiv_id, index_key, keep=False)  # 이전 실패가 남긴 같은 세대
        store.add_embeddings(
            texts=[chunk.page_content for chunk in chunks],
            embeddings=vectors,
            metadatas=[chunk.metadata for chunk in chunks],
        )
        _record_paper(collection_name, paper.arxiv_id, {
            "title": paper.title,
            "authors": list(paper.authors),
            "published": paper.published,
            "pdf_url": paper.pdf_url,
            "index_key": index_key,
            "chunks": len(chunks),
            "indexed_at": datetime.now().isoformat(timespec="seconds"),
        })
        # 기록이 새 세대를 가리킨 뒤에 옛 세대를 지움 — 여기서 실패해도 옛 청크는 검색되지 않음
        _delete_paper_chunks(collection_name, paper.arxiv_id, index_key, keep=True)
    return IndexResult(paper.arxiv_id, skipped=False, chunks=len(chunks))


def ask(
    question: str,
    llm: BaseChatModel,
    embeddings: Embeddings,
    collection_name: str = config.ARXIV_COLLECTION,
    arxiv_ids: list[str] | None = None,
    k: int = config.TOP_K,
) -> tuple[str, list[tuple[Document, float]]]:
    """R4 질의: 저장된 논문 전체 또는 arxiv_ids만 검색해 답변. 반환: (답변, [(근거 청크, 거리)]).

    기록된 세대(index_key)의 청크만 검색해 저장 도중 실패로 남은 청크는 근거로 쓰지 않음.
    """
    papers = [p for p in list_indexed_papers(collection_name) if not arxiv_ids or p["arxiv_id"] in arxiv_ids]
    if not papers:
        return NO_HITS_ANSWER, []
    store = rag.get_vector_store(embeddings, collection_name)
    question_vector = rag.embed_question(embeddings, question)
    search_filter = {"$and": [
        {"arxiv_id": {"$in": [p["arxiv_id"] for p in papers]}},
        {"index_key": {"$in": [p["index_key"] for p in papers]}},
    ]}
    hits = store.similarity_search_with_score_by_vector(question_vector, k=k, filter=search_filter)
    if not hits:
        return NO_HITS_ANSWER, []
    reply = rag.generate_answer(llm, rag.build_prompt(config.SYSTEM_PROMPT, [doc for doc, _ in hits], question))
    return reply.text, hits

