"""arxiv_service 테스트.

arXiv API · HTTP · 임베딩 · LLM은 가짜로 바꾸고, pgvector는 로컬 DB(docker compose)의 테스트 전용 컬렉션을 쓴다.
"""

import hashlib
import math
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import arxiv
import psycopg
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_postgres import PGVector

from rag_app import arxiv_service as ax
from rag_app import config
from rag_app import pipeline as rag


# ---------------------------------------------------------------------------
# 가짜 의존성
# ---------------------------------------------------------------------------
def make_pdf(pages: list[str]) -> bytes:
    """텍스트 페이지로 된 최소 PDF (pypdf가 읽을 수 있는 형식, ASCII만)."""
    n = len(pages)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{4 + 2 * i} 0 R' for i in range(n))}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        stream = f"BT /F1 10 Tf 20 700 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


class FakeEmbeddings(Embeddings):
    """단어 해시 bag-of-words 벡터 (같은 단어가 많을수록 가까움). delay로 느린 임베딩을 흉내."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * config.EMBEDDING_DIMENSION
        for word in text.lower().split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % len(vector)] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        time.sleep(self.delay)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeArxivClient:
    def __init__(self, results: list[arxiv.Result]):
        self._results = results
        self.searches: list[arxiv.Search] = []

    def results(self, search: arxiv.Search):
        self.searches.append(search)
        return iter(self._results)


def make_paper(arxiv_id: str, title: str = "A Test Paper", pdf_url: str | None = "https://arxiv.org/pdf/x") -> ax.Paper:
    return ax.Paper(arxiv_id, title, ("Kim",), "2024-01-02", "summary", pdf_url)


def write_paper_pdf(tmp_path, paper: ax.Paper, topic: str, pages: int = 2) -> Path:
    # 페이지마다 1000자를 넘겨 청크가 여러 개 생기게 함
    text = " ".join(f"{topic} sentence {i}." for i in range(120))
    path = ax.pdf_path_for(paper.arxiv_id, tmp_path)
    path.write_bytes(make_pdf([text] * pages))
    return path


def chunk_count(collection: str, arxiv_id: str) -> int:
    """세대와 상관없이 그 논문의 청크 전체 수."""
    with psycopg.connect(config.DATABASE_URL) as conn:
        return conn.execute(
            "SELECT count(*) FROM langchain_pg_embedding e JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
            "WHERE c.name = %s AND e.cmetadata->>'arxiv_id' = %s",
            (collection, arxiv_id),
        ).fetchone()[0]


@pytest.fixture
def collection():
    name = f"test_arxiv_{uuid.uuid4().hex[:8]}"
    yield name
    rag.delete_stored_collection(name)


# ---------------------------------------------------------------------------
# R1 검색
# ---------------------------------------------------------------------------
def test_search_papers_maps_results():
    result = arxiv.Result(
        "http://arxiv.org/abs/2401.00001v2",
        published=datetime(2024, 1, 2, 9, 30),
        title="Retrieval\n  Augmented   Generation",
        authors=[arxiv.Result.Author("Kim"), arxiv.Result.Author("Lee")],
        summary="Line one\nline two",
        links=[arxiv.Result.Link("http://arxiv.org/pdf/2401.00001v2", title="pdf")],
    )
    client = FakeArxivClient([result])

    papers = ax.search_papers("  rag  ", max_results=5, client=client)

    assert papers == [ax.Paper(
        "2401.00001v2", "Retrieval Augmented Generation", ("Kim", "Lee"), "2024-01-02",
        "Line one line two", "http://arxiv.org/pdf/2401.00001v2",
    )]
    assert client.searches[0].query == "rag"
    assert client.searches[0].max_results == 5


def test_search_papers_empty():
    client = FakeArxivClient([])
    assert ax.search_papers("nothing matches", client=client) == []
    assert ax.search_papers("   ", client=client) == []
    assert len(client.searches) == 1  # 빈 검색어는 API를 부르지 않음


# ---------------------------------------------------------------------------
# R2 다운로드
# ---------------------------------------------------------------------------
def test_download_pdf_saves_file(tmp_path):
    data = make_pdf(["hello"])
    path = ax.download_pdf(make_paper("2401.00001v1"), tmp_path, http_get=lambda url: data)

    assert path == tmp_path / "2401.00001v1.pdf"
    assert path.read_bytes() == data
    assert list(tmp_path.glob("*.part")) == []


def test_download_pdf_reuses_existing_file(tmp_path):
    paper = make_paper("2401.00001v1")
    ax.download_pdf(paper, tmp_path, http_get=lambda url: make_pdf(["hello"]))

    def must_not_call(url):
        raise AssertionError("이미 받은 파일인데 다시 다운로드함")

    assert ax.download_pdf(paper, tmp_path, http_get=must_not_call).exists()


def test_pdf_path_for_old_style_id(tmp_path):
    assert ax.pdf_path_for("hep-th/9901001v1", tmp_path) == tmp_path / "hep-th_9901001v1.pdf"


def test_paper_from_record_fills_pdf_url():
    # 저장 기록에 PDF 주소가 없으면(이전 기록 등) arXiv 규칙으로 만들어 다시 받을 수 있게 함
    paper = ax.paper_from_record({"arxiv_id": "2401.00001v1", "title": "T", "authors": ["Kim"], "published": "2024-01-02"})
    assert paper == ax.Paper("2401.00001v1", "T", ("Kim",), "2024-01-02", "", "https://arxiv.org/pdf/2401.00001v1")
    assert ax.paper_from_record({"arxiv_id": "x", "pdf_url": "https://a/b.pdf"}).pdf_url == "https://a/b.pdf"


def test_download_pdf_rejects_non_pdf(tmp_path):
    with pytest.raises(ValueError, match="PDF가 아닌"):
        ax.download_pdf(make_paper("2401.00001v1"), tmp_path, http_get=lambda url: b"<html>captcha</html>")
    assert list(tmp_path.iterdir()) == []


def test_download_pdf_without_url(tmp_path):
    with pytest.raises(ValueError, match="PDF 링크가 없는"):
        ax.download_pdf(make_paper("2401.00001v1", pdf_url=None), tmp_path, http_get=lambda url: b"")


# ---------------------------------------------------------------------------
# R3 저장 · R6 실패 복구 · 동시성
# ---------------------------------------------------------------------------
def test_index_paper_stores_chunks_and_registry(tmp_path, collection):
    paper = make_paper("2401.00001v1", title="Attention Paper")
    pdf = write_paper_pdf(tmp_path, paper, "attention")

    result = ax.index_paper(paper, pdf, FakeEmbeddings(), collection)

    assert not result.skipped and result.chunks > 1
    assert chunk_count(collection, paper.arxiv_id) == result.chunks
    [record] = ax.list_indexed_papers(collection)
    assert record["arxiv_id"] == paper.arxiv_id
    assert record["title"] == "Attention Paper"
    assert record["chunks"] == result.chunks
    assert record["pdf_url"] == paper.pdf_url


def test_index_paper_skips_same_index(tmp_path, collection):
    paper = make_paper("2401.00001v1")
    pdf = write_paper_pdf(tmp_path, paper, "attention")
    first = ax.index_paper(paper, pdf, FakeEmbeddings(), collection)

    second = ax.index_paper(paper, pdf, FakeEmbeddings(), collection)

    assert second.skipped and second.chunks == first.chunks
    assert chunk_count(collection, paper.arxiv_id) == first.chunks


def test_index_paper_recovers_after_failure(tmp_path, collection, monkeypatch):
    other = make_paper("2401.00002v1", title="Other")
    other_result = ax.index_paper(other, write_paper_pdf(tmp_path, other, "graph"), FakeEmbeddings(), collection)
    paper = make_paper("2401.00001v1")
    pdf = write_paper_pdf(tmp_path, paper, "attention", pages=3)

    original_add = PGVector.add_embeddings

    def add_half_then_fail(self, texts, embeddings, metadatas=None, **kwargs):
        half = len(texts) // 2
        original_add(self, texts[:half], embeddings[:half], metadatas[:half], **kwargs)
        raise RuntimeError("DB 연결 끊김")

    monkeypatch.setattr(PGVector, "add_embeddings", add_half_then_fail)
    with pytest.raises(RuntimeError):
        ax.index_paper(paper, pdf, FakeEmbeddings(), collection)
    # 기록이 남지 않아 "저장됨"으로 보이지 않음
    assert paper.arxiv_id not in {p["arxiv_id"] for p in ax.list_indexed_papers(collection)}

    monkeypatch.setattr(PGVector, "add_embeddings", original_add)
    retry = ax.index_paper(paper, pdf, FakeEmbeddings(), collection)

    assert not retry.skipped
    assert chunk_count(collection, paper.arxiv_id) == retry.chunks  # 반쯤 저장된 청크가 남지 않음
    assert chunk_count(collection, other.arxiv_id) == other_result.chunks  # 다른 논문은 그대로


def test_index_paper_concurrent_same_paper(tmp_path, collection):
    paper = make_paper("2401.00001v1")
    pdf = write_paper_pdf(tmp_path, paper, "attention")
    ax.index_paper(make_paper("2401.00009v1"), write_paper_pdf(tmp_path, make_paper("2401.00009v1"), "warmup"),
                   FakeEmbeddings(), collection)  # 컬렉션 · 테이블 생성을 미리 끝내 두 스레드가 같은 조건에서 시작

    barrier = threading.Barrier(2)
    results, errors = [], []

    def worker():
        barrier.wait()
        try:
            results.append(ax.index_paper(paper, pdf, FakeEmbeddings(delay=0.5), collection))
        except Exception as e:  # pragma: no cover - 실패 시 원인 확인용
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert sorted(r.skipped for r in results) == [False, True]
    assert chunk_count(collection, paper.arxiv_id) == results[0].chunks


# ---------------------------------------------------------------------------
# R4 질의
# ---------------------------------------------------------------------------
def test_ask_filters_by_arxiv_id(tmp_path, collection):
    attention = make_paper("2401.00001v1", title="Attention Paper")
    graph = make_paper("2401.00002v1", title="Graph Paper")
    ax.index_paper(attention, write_paper_pdf(tmp_path, attention, "attention"), FakeEmbeddings(), collection)
    ax.index_paper(graph, write_paper_pdf(tmp_path, graph, "graph"), FakeEmbeddings(), collection)
    llm = FakeListChatModel(responses=["가짜 답변"])

    # 질문은 attention 논문에 가깝지만 대상은 graph 논문만
    answer, hits = ax.ask("attention sentence", llm, FakeEmbeddings(), collection, arxiv_ids=[graph.arxiv_id], k=3)

    assert answer == "가짜 답변"
    assert hits and {doc.metadata["arxiv_id"] for doc, _ in hits} == {graph.arxiv_id}
    assert {doc.metadata["source"] for doc, _ in hits} == {"Graph Paper"}

    # 대상을 고르지 않으면 전체에서 가장 가까운 attention 논문이 나옴
    _, all_hits = ax.ask("attention sentence", llm, FakeEmbeddings(), collection, k=1)
    assert all_hits[0][0].metadata["arxiv_id"] == attention.arxiv_id


# ---------------------------------------------------------------------------
# 코드 리뷰 반영: 재저장 실패 시 기존 버전 유지 · 컬렉션 분리 · 기록 동시 갱신
# ---------------------------------------------------------------------------
def test_index_paper_reindex_failure_keeps_old_version(tmp_path, collection, monkeypatch):
    paper = make_paper("2401.00001v1", title="Attention Paper")
    pdf = write_paper_pdf(tmp_path, paper, "attention")
    old = ax.index_paper(paper, pdf, FakeEmbeddings(), collection)
    old_key = ax.list_indexed_papers(collection)[0]["index_key"]

    # 청크 설정이 바뀌어 다시 저장하는데 새 청크 저장 도중 실패
    monkeypatch.setattr(config, "CHUNK_SIZE", 500)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 100)
    original_add = PGVector.add_embeddings

    def add_half_then_fail(self, texts, embeddings, metadatas=None, **kwargs):
        half = len(texts) // 2
        original_add(self, texts[:half], embeddings[:half], metadatas[:half], **kwargs)
        raise RuntimeError("DB 연결 끊김")

    monkeypatch.setattr(PGVector, "add_embeddings", add_half_then_fail)
    with pytest.raises(RuntimeError):
        ax.index_paper(paper, pdf, FakeEmbeddings(), collection)

    # 기록은 옛 버전 그대로, 질의도 옛 버전 청크만 근거로 씀
    [record] = ax.list_indexed_papers(collection)
    assert (record["index_key"], record["chunks"]) == (old_key, old.chunks)
    _, hits = ax.ask("attention sentence", FakeListChatModel(responses=["답"]), FakeEmbeddings(), collection, k=20)
    assert hits and {doc.metadata["index_key"] for doc, _ in hits} == {old_key}

    # 재시도하면 새 버전으로 교체되고 옛 청크 · 실패한 청크는 남지 않음
    monkeypatch.setattr(PGVector, "add_embeddings", original_add)
    new = ax.index_paper(paper, pdf, FakeEmbeddings(), collection)
    assert not new.skipped and new.chunks != old.chunks
    assert ax.list_indexed_papers(collection)[0]["index_key"] != old_key
    assert chunk_count(collection, paper.arxiv_id) == new.chunks


def test_record_paper_concurrent_different_papers(tmp_path, collection):
    papers = [make_paper(f"2401.0000{i}v1", title=f"Paper {i}") for i in range(1, 5)]
    pdfs = [write_paper_pdf(tmp_path, p, f"topic{i}", pages=1) for i, p in enumerate(papers)]
    ax.index_paper(make_paper("2401.00009v1"), write_paper_pdf(tmp_path, make_paper("2401.00009v1"), "warmup"),
                   FakeEmbeddings(), collection)
    barrier = threading.Barrier(len(papers))

    def worker(paper, pdf):
        barrier.wait()
        ax.index_paper(paper, pdf, FakeEmbeddings(), collection)

    threads = [threading.Thread(target=worker, args=pair) for pair in zip(papers, pdfs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 서로 다른 논문의 기록이 동시에 병합돼도 하나도 유실되지 않음
    assert {p["arxiv_id"] for p in ax.list_indexed_papers(collection)} == {p.arxiv_id for p in papers} | {"2401.00009v1"}


def test_ask_without_papers(collection):
    llm = FakeListChatModel(responses=["불리면 안 됨"])
    assert ax.ask("anything", llm, FakeEmbeddings(), collection) == (ax.NO_HITS_ANSWER, [])


def test_save_to_store_refuses_paper_collection(tmp_path, collection):
    paper = make_paper("2401.00001v1")
    stored = ax.index_paper(paper, write_paper_pdf(tmp_path, paper, "attention"), FakeEmbeddings(), collection)
    store = rag.get_vector_store(FakeEmbeddings(), collection)

    # 기존 PDF RAG · OCR 평가 화면이 같은 이름으로 저장해도 논문 컬렉션을 비우지 않음
    with pytest.raises(ValueError, match="arXiv 논문 컬렉션"):
        rag.save_to_store(store, [], [], {})
    assert chunk_count(collection, paper.arxiv_id) == stored.chunks
    assert rag.is_paper_collection(rag.get_collection_info(collection))
    # 이름이 비슷해도 논문 기록이 없는 컬렉션은 해당하지 않음 (예: arxiv_papers.pdf를 올려 만든 컬렉션)
    assert not rag.is_paper_collection({"file_name": "arxiv_papers.pdf"})
    assert not rag.is_paper_collection(None)


def test_arxiv_collection_is_separated_by_embedding():
    assert config.ARXIV_COLLECTION == (
        f"{config.ARXIV_COLLECTION_PREFIX}_{config.EMBEDDING_MODEL.replace('-', '_')}_d{config.EMBEDDING_DIMENSION}"
    )


# ---------------------------------------------------------------------------
# R5 · R7 화면: 검색 → 미리보기
# ---------------------------------------------------------------------------
def test_page_search_and_preview(tmp_path, monkeypatch):
    from unittest import mock

    from streamlit.testing.v1 import AppTest

    result = arxiv.Result(
        "http://arxiv.org/abs/2401.00001v1", published=datetime(2024, 1, 2), title="Fake Paper",
        authors=[arxiv.Result.Author("Kim")], summary="s",
        links=[arxiv.Result.Link("http://arxiv.org/pdf/2401.00001v1", title="pdf")],
    )
    stored = [{"arxiv_id": "2401.00002v1", "title": "Stored Paper", "chunks": 3, "pdf_kind": "text", "generation": "g"}]
    downloaded, deleted = [], []

    def fake_download(paper):
        downloaded.append((paper.arxiv_id, paper.pdf_url))
        path = tmp_path / "preview.pdf"
        path.write_bytes(make_pdf(["hello"]))
        return path

    def fake_delete(arxiv_id):
        deleted.append(arxiv_id)
        stored.clear()

    monkeypatch.setattr(ax, "list_indexed_papers", lambda *a, **k: list(stored))
    monkeypatch.setattr(ax, "download_pdf", fake_download)
    monkeypatch.setattr(ax, "delete_paper", fake_delete)
    # ADC 인증 없는 환경에서도 돌도록 화면이 만드는 Gemini 임베딩을 가짜로
    monkeypatch.setattr(rag, "create_embeddings", lambda *a, **k: FakeEmbeddings())
    # AppTest는 페이지 파일을 진입점으로 실행해 사이드바의 다른 페이지 링크를 찾지 못하므로 링크만 무시
    with mock.patch("streamlit.page_link"), mock.patch("arxiv.Client.results", lambda self, s: iter([result])):
        at = AppTest.from_file(str(Path(__file__).parents[1] / "rag_app/pages/arxiv_qa.py"), default_timeout=60).run()
        assert not at.exception
        at.text_input[0].input("rag")
        at.button[0].click().run()
        search_table, stored_table = (e.value for e in at.get("dataframe"))  # 검색 결과 · 저장된 논문 표
        table = search_table
        assert list(table["PDF"]) == ["http://arxiv.org/pdf/2401.00001v1"]  # 행마다 원문 PDF 링크 (R1)
        assert downloaded == []  # 검색만으로는 내려받지 않음

        # 저장된 논문 표: 행을 고르기 전엔 미리보기 · 삭제 버튼이 없음
        grid_key = "arxiv_stored_grid_" + hashlib.md5(b"2401.00002v1").hexdigest()[:8]
        assert not [b for b in at.button if b.key in ("arxiv_view", "arxiv_del")]
        assert list(stored_table["arXiv ID"]) == ["2401.00002v1"]
        assert list(stored_table["PDF"]) == ["https://arxiv.org/pdf/2401.00002v1"]

        # 행 선택 → 미리보기: 받아 둔 파일이 없으면 기록의 PDF 주소로 받아 화면 안에 표시 (R7)
        at.session_state[grid_key] = {"selection": {"rows": [0], "columns": [], "cells": []}}
        at.run()
        at.button(key="arxiv_view").click().run()
        assert not at.exception
        assert downloaded == [("2401.00002v1", "https://arxiv.org/pdf/2401.00002v1")]
        assert at.session_state["arxiv_preview"] == {"arxiv_id": "2401.00002v1", "path": str(tmp_path / "preview.pdf")}
        assert at.get("bidi_component")  # st.pdf 뷰어

        # 닫기 → 뷰어 · 상태 제거
        at.button(key="arxiv_preview_close").click().run()
        assert "arxiv_preview" not in at.session_state and not at.get("bidi_component")

        # 미리보던 논문을 삭제하면 미리보기도 함께 닫힘 (R11)
        at.button(key="arxiv_view").click().run()
        at.button(key="arxiv_del").click().run()
    assert not at.exception
    assert deleted == ["2401.00002v1"]
    assert "arxiv_preview" not in at.session_state and not at.get("bidi_component")

def test_save_to_store_still_works_for_normal_collection(collection):
    # 보호 장치를 넣은 뒤에도 일반 컬렉션은 기존처럼 비우고 다시 저장됨
    store = rag.get_vector_store(FakeEmbeddings(), collection)
    docs = [Document(page_content=f"chunk {i}", metadata={"page": 1}) for i in range(3)]
    vectors = FakeEmbeddings().embed_documents([d.page_content for d in docs])
    rag.save_to_store(store, docs, vectors, {"index_key": "k1"})
    rag.save_to_store(store, docs[:2], vectors[:2], {"index_key": "k2"})

    info = rag.get_collection_info(collection)
    assert (info["stored_chunks"], info["index_key"], info["chunks"]) == (2, "k2", 2)
    assert not rag.is_paper_collection(info)


# ---------------------------------------------------------------------------
# 2차: R8 OCR · R9 top-k · R10 하이브리드 · R11 삭제 · R12 이전
# ---------------------------------------------------------------------------
import io  # noqa: E402

import pypdf  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


class RecordingLLM(BaseChatModel):
    """받은 프롬프트 텍스트를 기록하고 정해진 답을 돌려줌. fail=True면 항상 오류 (OCR 실패 흉내)."""

    responses: list[str] = ["가짜 답변"]
    fail: bool = False
    prompts: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        content = messages[-1].content
        text = content if isinstance(content, str) else " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        self.prompts.append(text)
        if self.fail:
            raise RuntimeError("503 UNAVAILABLE")
        answer = self.responses[(len(self.prompts) - 1) % len(self.responses)]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])


class ConstantEmbeddings(FakeEmbeddings):
    """모든 텍스트가 같은 벡터 → 의미 검색이 아무 정보도 주지 못하는 상황 (BM25만 구분 가능)."""

    def _vector(self, text: str) -> list[float]:
        return [1 / math.sqrt(config.EMBEDDING_DIMENSION)] * config.EMBEDDING_DIMENSION


def make_image_pdf(pages: int) -> bytes:
    """텍스트 레이어 없이 이미지만 있는 스캔 PDF."""
    images = []
    for i in range(pages):
        img = Image.new("RGB", (400, 300), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle((5, 5, 395, 295), outline="black")
        draw.text((20, 20), f"scanned page {i + 1}", fill="black")
        images.append(img)
    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def merge_pdfs(*parts: bytes) -> bytes:
    writer = pypdf.PdfWriter()
    for part in parts:
        for page in pypdf.PdfReader(io.BytesIO(part)).pages:
            writer.add_page(page)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


OCR_TEXT = " ".join(f"ocr scanned sentence {i}." for i in range(60))


@pytest.fixture
def ocr_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ax, "OCR_CACHE_DIR", tmp_path / "ocr_pages")


def test_analyze_text_pdf_skips_ocr(tmp_path, collection, ocr_cache):
    paper = make_paper("2401.00001v1")
    ocr = RecordingLLM(responses=[OCR_TEXT])
    result = ax.index_paper(paper, write_paper_pdf(tmp_path, paper, "attention"), FakeEmbeddings(), collection, ocr)

    assert (result.pdf_kind, result.ocr_pages) == ("text", ())
    assert ocr.prompts == []  # 텍스트 PDF는 LLM을 부르지 않음
    assert ax.list_indexed_papers(collection)[0]["pdf_kind"] == "text"


def test_index_scanned_pdf_runs_ocr(tmp_path, collection, ocr_cache):
    paper = make_paper("2401.00001v1")
    pdf = ax.pdf_path_for(paper.arxiv_id, tmp_path)
    pdf.write_bytes(make_image_pdf(2))
    ocr = RecordingLLM(responses=[OCR_TEXT])

    result = ax.index_paper(paper, pdf, FakeEmbeddings(), collection, ocr)

    assert (result.pdf_kind, result.ocr_pages, result.ocr_failures) == ("image", (1, 2), ())
    assert len(ocr.prompts) == 2 and all("학술 논문" in p for p in ocr.prompts)  # 논문용 OCR 프롬프트
    _, hits = ax.ask("ocr scanned sentence", RecordingLLM(), FakeEmbeddings(), collection)
    assert hits and all("ocr scanned" in doc.page_content for doc, _ in hits)  # OCR 결과가 청크로 저장됨


def test_index_mixed_pdf_ocr_only_image_pages(tmp_path, collection, ocr_cache):
    paper = make_paper("2401.00001v1")
    text_page = make_pdf([" ".join(f"attention sentence {i}." for i in range(120))])
    pdf = ax.pdf_path_for(paper.arxiv_id, tmp_path)
    pdf.write_bytes(merge_pdfs(text_page, make_image_pdf(1)))
    ocr = RecordingLLM(responses=[OCR_TEXT])

    result = ax.index_paper(paper, pdf, FakeEmbeddings(), collection, ocr)

    assert (result.pdf_kind, result.ocr_pages) == ("mixed", (2,))
    assert len(ocr.prompts) == 1


def test_index_scanned_pdf_requires_ocr_llm(tmp_path, collection, ocr_cache):
    paper = make_paper("2401.00001v1")
    pdf = ax.pdf_path_for(paper.arxiv_id, tmp_path)
    pdf.write_bytes(make_image_pdf(1))
    with pytest.raises(ValueError, match="OCR할 모델"):
        ax.index_paper(paper, pdf, FakeEmbeddings(), collection)
    assert ax.list_indexed_papers(collection) == []


def test_ocr_failure_is_retried_next_time(tmp_path, collection, ocr_cache):
    paper = make_paper("2401.00001v1")
    pdf = ax.pdf_path_for(paper.arxiv_id, tmp_path)
    pdf.write_bytes(merge_pdfs(make_pdf([" ".join(f"attention sentence {i}." for i in range(120))]), make_image_pdf(1)))

    failed = ax.index_paper(paper, pdf, FakeEmbeddings(), collection, RecordingLLM(fail=True))
    assert failed.ocr_failures and ax.list_indexed_papers(collection)[0]["ocr_failures"]

    # OCR 실패 기록이 있으면 같은 설정이어도 건너뛰지 않고 다시 OCR
    ocr = RecordingLLM(responses=[OCR_TEXT])
    retried = ax.index_paper(paper, pdf, FakeEmbeddings(), collection, ocr)
    assert not retried.skipped and retried.ocr_failures == () and len(ocr.prompts) == 1
    assert ax.list_indexed_papers(collection)[0]["ocr_failures"] == []

    # 이제 온전히 저장됐으므로 건너뜀 (OCR 호출 없음)
    again = RecordingLLM(responses=[OCR_TEXT])
    assert ax.index_paper(paper, pdf, FakeEmbeddings(), collection, again).skipped and again.prompts == []


def _paper_with_rare_word(tmp_path, collection, embeddings):
    paper = make_paper("2401.00001v1", title="Rare Word Paper")
    # 페이지마다 다른 문장 (내용이 같은 청크는 RRF에서 하나로 합쳐지므로)
    pages = [" ".join(f"model training sentence p{n} {i}." for i in range(100)) for n in range(9)]
    pages[4] = " ".join(f"model training sentence p4 {i}." for i in range(50)) + " the zygomorphic estimator works."
    pdf = ax.pdf_path_for(paper.arxiv_id, tmp_path)
    pdf.write_bytes(make_pdf(pages))
    ax.index_paper(paper, pdf, embeddings, collection)
    return paper


def test_hybrid_finds_keyword_missed_by_vector(tmp_path, collection):
    _paper_with_rare_word(tmp_path, collection, ConstantEmbeddings())

    _, hits = ax.ask("zygomorphic estimator", RecordingLLM(), ConstantEmbeddings(), collection)

    rare = [(doc, ranks) for doc, ranks in hits if "zygomorphic" in doc.page_content]
    assert rare and rare[0][1]["bm25_rank"] == 1  # 의미 검색으로는 구분 못 하는 정확한 키워드를 BM25가 1위로


def test_ask_returns_top_5(tmp_path, collection):
    _paper_with_rare_word(tmp_path, collection, FakeEmbeddings())
    _, hits = ax.ask("model training", RecordingLLM(), FakeEmbeddings(), collection)
    assert config.ARXIV_TOP_K == 5 and len(hits) == 5
    assert [ranks["rrf_rank"] for _, ranks in hits] == [1, 2, 3, 4, 5]


def test_bm25_query_translation(tmp_path, collection):
    # 영어 질문: 변환 호출 없이 그대로
    llm = RecordingLLM()
    assert ax.keyword_query("zygomorphic estimator", llm) == "zygomorphic estimator" and llm.prompts == []

    # 한글 질문: LLM이 만든 영어 키워드로 BM25 검색, 답변은 같은 LLM의 다음 호출
    _paper_with_rare_word(tmp_path, collection, ConstantEmbeddings())
    llm = RecordingLLM(responses=["zygomorphic estimator", "가짜 답변"])
    answer, hits = ax.ask("자이고모픽 추정량이 뭐야?", llm, ConstantEmbeddings(), collection)

    assert "자이고모픽 추정량이 뭐야?" in llm.prompts[0] and "영어 키워드" in llm.prompts[0]
    assert answer == "가짜 답변"
    assert any("zygomorphic" in doc.page_content and ranks["bm25_rank"] == 1 for doc, ranks in hits)


def test_bm25_cache_follows_generation(tmp_path, collection, monkeypatch):
    paper = make_paper("2401.00001v1")
    pdf = write_paper_pdf(tmp_path, paper, "attention")
    ax.index_paper(paper, pdf, FakeEmbeddings(), collection)
    ax.ask("attention", RecordingLLM(), FakeEmbeddings(), collection)

    monkeypatch.setattr(config, "CHUNK_SIZE", 500)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 100)
    ax.index_paper(paper, pdf, FakeEmbeddings(), collection)
    new_key = ax.list_indexed_papers(collection)[0]["index_key"]
    _, hits = ax.ask("attention", RecordingLLM(), FakeEmbeddings(), collection)

    # 옛 세대는 지워졌고 BM25도 새 세대 청크로 다시 만들어 BM25에서 찾은 근거가 모두 새 세대
    bm25_hits = [doc for doc, ranks in hits if ranks["bm25_rank"]]
    assert bm25_hits and {doc.metadata["index_key"] for doc in bm25_hits} == {new_key}


def test_delete_paper(tmp_path, collection):
    a, b = make_paper("2401.00001v1", title="A"), make_paper("2401.00002v1", title="B")
    ax.index_paper(a, write_paper_pdf(tmp_path, a, "attention"), FakeEmbeddings(), collection)
    b_result = ax.index_paper(b, write_paper_pdf(tmp_path, b, "graph"), FakeEmbeddings(), collection)

    assert ax.delete_paper(a.arxiv_id, collection)

    assert [p["arxiv_id"] for p in ax.list_indexed_papers(collection)] == [b.arxiv_id]
    assert chunk_count(collection, a.arxiv_id) == 0
    assert chunk_count(collection, b.arxiv_id) == b_result.chunks
    assert ax.ask("attention", RecordingLLM(), FakeEmbeddings(), collection, arxiv_ids=[a.arxiv_id]) == (ax.NO_HITS_ANSWER, [])
    assert ax.pdf_path_for(a.arxiv_id, tmp_path).exists()  # 받아 둔 PDF는 남김
    assert not ax.delete_paper(a.arxiv_id, collection)  # 이미 없으면 False


@pytest.fixture
def legacy_collection():
    name = f"test_arxiv_legacy_{uuid.uuid4().hex[:8]}"
    yield name
    rag.delete_stored_collection(name)


def _make_legacy(tmp_path, legacy_name: str) -> tuple[ax.Paper, int]:
    """세대 표시(index_key)가 청크에 없던 수정 전 형태의 컬렉션을 만듦."""
    paper = make_paper("2401.00001v1", title="Legacy Paper")
    result = ax.index_paper(paper, write_paper_pdf(tmp_path, paper, "attention"), FakeEmbeddings(), legacy_name)
    with psycopg.connect(config.DATABASE_URL) as conn:
        conn.execute(
            "UPDATE langchain_pg_embedding e SET cmetadata = e.cmetadata - 'index_key' - 'generation' "
            "FROM langchain_pg_collection c WHERE e.collection_id = c.uuid AND c.name = %s",
            (legacy_name,),
        )
        conn.execute(  # 기록에도 generation이 없던 시절
            "UPDATE langchain_pg_collection SET cmetadata = (cmetadata::jsonb #- %s)::json WHERE name = %s",
            (["papers", paper.arxiv_id, "generation"], legacy_name),
        )
    return paper, result.chunks


def test_migrate_legacy_collection(tmp_path, legacy_collection, collection):
    paper, chunks = _make_legacy(tmp_path, legacy_collection)
    assert ax.ask("attention", RecordingLLM(), FakeEmbeddings(), legacy_collection) == (ax.NO_HITS_ANSWER, [])  # 이전 전엔 검색 안 됨

    assert ax.migrate_legacy_collection(legacy_collection, collection) == 1

    assert rag.get_collection_info(legacy_collection) is None
    _, hits = ax.ask("attention", RecordingLLM(), FakeEmbeddings(), collection)
    assert hits and chunk_count(collection, paper.arxiv_id) == chunks
    # 재임베딩 없이 "이미 저장됨"으로 인정
    assert ax.index_paper(paper, ax.pdf_path_for(paper.arxiv_id, tmp_path), FakeEmbeddings(), collection).skipped


def test_migrate_refuses_existing_target(tmp_path, legacy_collection, collection):
    paper, chunks = _make_legacy(tmp_path, legacy_collection)
    rag.get_vector_store(FakeEmbeddings(), collection)  # 새 이름이 이미 있음

    with pytest.raises(ValueError, match="이미 있어"):
        ax.migrate_legacy_collection(legacy_collection, collection)
    assert chunk_count(legacy_collection, paper.arxiv_id) == chunks  # 원본 그대로


# ---------------------------------------------------------------------------
# 2차 리뷰 반영: 저장마다 세대 id (M1) · OCR 실패와 추출 경고 분리 (M2) · 키워드 변환 실패
# ---------------------------------------------------------------------------
def _mixed_paper(tmp_path):
    paper = make_paper("2401.00001v1")
    pdf = ax.pdf_path_for(paper.arxiv_id, tmp_path)
    pdf.write_bytes(merge_pdfs(make_pdf([" ".join(f"attention sentence {i}." for i in range(120))]), make_image_pdf(1)))
    return paper, pdf


OCR_RARE = " ".join(f"ocr scanned sentence {i}." for i in range(40)) + " the quasicrystal lattice appears."


def test_ocr_retry_failure_keeps_live_generation(tmp_path, collection, ocr_cache, monkeypatch):
    paper, pdf = _mixed_paper(tmp_path)
    first = ax.index_paper(paper, pdf, FakeEmbeddings(), collection, RecordingLLM(fail=True))
    live = ax.list_indexed_papers(collection)[0]

    # OCR 재시도는 같은 index_key지만, 저장 도중 실패해도 지금 검색되는 세대를 먼저 지우지 않아야 함
    original_add = PGVector.add_embeddings

    def add_half_then_fail(self, texts, embeddings, metadatas=None, **kwargs):
        half = len(texts) // 2
        original_add(self, texts[:half], embeddings[:half], metadatas[:half], **kwargs)
        raise RuntimeError("DB 연결 끊김")

    monkeypatch.setattr(PGVector, "add_embeddings", add_half_then_fail)
    with pytest.raises(RuntimeError):
        ax.index_paper(paper, pdf, FakeEmbeddings(), collection, RecordingLLM(responses=[OCR_RARE]))

    record = ax.list_indexed_papers(collection)[0]
    assert record["generation"] == live["generation"]
    assert ax._count_paper_chunks(collection, paper.arxiv_id, live["generation"]) == first.chunks
    _, hits = ax.ask("attention sentence", RecordingLLM(), FakeEmbeddings(), collection, k=20)
    assert hits and {doc.metadata["generation"] for doc, _ in hits} == {live["generation"]}


def test_ocr_retry_refreshes_bm25_cache(tmp_path, collection, ocr_cache):
    paper, pdf = _mixed_paper(tmp_path)
    ax.index_paper(paper, pdf, FakeEmbeddings(), collection, RecordingLLM(fail=True))
    ax.ask("quasicrystal lattice", RecordingLLM(), FakeEmbeddings(), collection)  # 실패 상태로 BM25 캐시가 만들어짐

    ax.index_paper(paper, pdf, FakeEmbeddings(), collection, RecordingLLM(responses=[OCR_RARE]))
    _, hits = ax.ask("quasicrystal lattice", RecordingLLM(), FakeEmbeddings(), collection)

    # 같은 index_key로 다시 저장했어도 세대가 바뀌어 BM25가 새 청크(OCR 결과)를 찾음
    assert any("quasicrystal" in doc.page_content and ranks["bm25_rank"] == 1 for doc, ranks in hits)
    live = ax.list_indexed_papers(collection)[0]["generation"]
    assert {doc.metadata["generation"] for doc, _ in hits} == {live}


def test_extraction_warning_does_not_block_skip(tmp_path, collection, ocr_cache):
    # 글자 없는 페이지에 단색(흰색) 이미지 → 다시 해도 같은 추출 문제라 경고만, 건너뛰기는 정상
    paper = make_paper("2401.00001v1")
    blank = io.BytesIO()
    Image.new("RGB", (300, 300), "white").save(blank, format="PDF")
    pdf = ax.pdf_path_for(paper.arxiv_id, tmp_path)
    pdf.write_bytes(merge_pdfs(make_pdf([" ".join(f"attention sentence {i}." for i in range(120))]), blank.getvalue()))
    ocr = RecordingLLM(responses=[OCR_TEXT])

    first = ax.index_paper(paper, pdf, FakeEmbeddings(), collection, ocr)
    assert first.ocr_warnings and "단색" in first.ocr_warnings[0] and first.ocr_failures == ()
    assert ocr.prompts == []  # OCR할 이미지가 없어 LLM 호출 없음
    assert ax.index_paper(paper, pdf, FakeEmbeddings(), collection, ocr).skipped


def test_keyword_query_falls_back_on_error():
    # 변환 호출이 429 등으로 실패해도 원문으로 BM25 진행 (의미 검색은 영향 없음)
    assert ax.keyword_query("자이고모픽 추정량", RecordingLLM(fail=True)) == "자이고모픽 추정량"


def test_backfill_generation(tmp_path, legacy_collection):
    # 이미 새 이름으로 옮겼지만 세대 id가 없던 컬렉션 (실제 데이터의 상황)
    paper, chunks = _make_legacy(tmp_path, legacy_collection)
    assert ax.ask("attention", RecordingLLM(), FakeEmbeddings(), legacy_collection) == (ax.NO_HITS_ANSWER, [])

    assert ax.backfill_generation(legacy_collection) == 1
    assert ax.backfill_generation(legacy_collection) == 0  # 두 번 실행해도 그대로

    _, hits = ax.ask("attention", RecordingLLM(), FakeEmbeddings(), legacy_collection)
    assert hits and chunk_count(legacy_collection, paper.arxiv_id) == chunks
    assert ax.index_paper(paper, ax.pdf_path_for(paper.arxiv_id, tmp_path), FakeEmbeddings(), legacy_collection).skipped


def test_delete_paper_removes_legacy_chunks(tmp_path, legacy_collection):
    # 세대 id가 없는 옛 청크도 논문 삭제 시 남지 않음 (R11)
    paper, _ = _make_legacy(tmp_path, legacy_collection)
    assert ax.delete_paper(paper.arxiv_id, legacy_collection)
    assert chunk_count(legacy_collection, paper.arxiv_id) == 0
