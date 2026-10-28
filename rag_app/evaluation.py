"""OCR · 질문/정답 생성 · RAG 평가 (pages/ocr_eval.py에서 사용).

검색 · 프롬프트 · 답변 단계는 pipeline.py의 함수를 그대로 쓰고, 여기에는 평가에만 필요한 것을 둔다.
    ocr_pdf          스캔 PDF → 페이지별 인쇄 텍스트 (Gemini, 사진 · 그림 제외) + 페이지 단위 캐시
    generate_qa      OCR 본문 → 질문 · 정답 · 근거 페이지 N쌍 (Gemini 구조화 출력)
    evaluate         질문마다 RAG 실행 → 정답 페이지 검색 여부 + Gemini 채점 (correct / partial / incorrect)
    summarize        정답률 · 검색 적중률 · 지연 시간 · 토큰 집계
"""

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_postgres import PGVector
from pydantic import BaseModel, Field

from rag_app import pipeline as rag
from rag_app.config import (
    EVAL_CONCURRENCY,
    EVAL_DATA_DIR,
    FAITHFULNESS_PROMPT,
    IMAGE_CONCURRENCY,
    JUDGE_PROMPT,
    QA_GENERATION_PROMPT,
)

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


OCR_PAGE_CACHE_DIR = EVAL_DATA_DIR / "ocr_pages"


def _page_cache_path(images: list[tuple[str, object]], model: str, prompt: str) -> Path:
    """페이지 이미지 픽셀 + 모델 + 프롬프트로 만든 캐시 파일 경로. 다른 PDF라도 같은 페이지면 같은 키."""
    digest = hashlib.sha256(f"{model}|{prompt}".encode())
    for _, img in images:
        digest.update(hashlib.sha256(img.tobytes()).digest())
    return OCR_PAGE_CACHE_DIR / f"{digest.hexdigest()[:24]}.txt"


def ocr_pdf(
    data: bytes,
    file_name: str,
    llm: BaseChatModel,
    model: str,
    prompt: str,
    use_cache: bool = True,
    on_page: Callable[[int, int, int, bool], None] | None = None,
) -> tuple[list[Document], list[str], int]:
    """스캔 PDF를 페이지마다 Gemini로 받아 적음. 반환: (페이지 Document, 실패 목록, 캐시에서 가져온 페이지 수).

    - pipeline의 GeminiImageReader · extract_page_images를 그대로 쓰되 OCR 프롬프트를 주고 [이미지 설명] 표시는 생략
    - 페이지 단위 캐시(eval/data/ocr_pages/): 성공한 페이지는 바로 저장하고, 실패한 페이지만 다음 실행 때 다시 OCR.
      키가 페이지 이미지 기준이라 1~20쪽과 1~100쪽처럼 같은 페이지를 담은 다른 PDF도 캐시를 공유
    - 페이지를 IMAGE_CONCURRENCY개씩 동시에 처리, on_page(끝난 수, 전체, 페이지 번호, 캐시 사용)는 호출한 스레드에서 불림
    """
    pages = rag.load_pdf(data, file_name)  # 텍스트 레이어 (스캔 PDF는 빈 문자열) + 페이지 metadata
    reader = rag.GeminiImageReader(llm, prompt, mark=None, empty_ok=True)
    page_images = rag.extract_page_images(data, reader)

    def read_page(index: int) -> tuple[int, str, bool]:
        images = page_images[index]
        if not images:
            return index, "", False
        cache = _page_cache_path(images, model, prompt)
        if use_cache and cache.exists():
            return index, cache.read_text(), True
        before = len(reader.failures)
        text = "\n\n".join(t for label, img in images if (t := reader.describe(img, label)))
        if len(reader.failures) == before:  # 이 페이지에서 실패가 없을 때만 저장
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text)
        return index, text, False

    cached = 0
    with ThreadPoolExecutor(max_workers=IMAGE_CONCURRENCY) as pool:
        futures = [pool.submit(read_page, i) for i in range(len(pages))]
        for done, future in enumerate(as_completed(futures), start=1):
            index, text, from_cache = future.result()
            cached += from_cache
            pages[index].page_content = f"{pages[index].page_content.strip()}\n\n{text}".strip()
            if on_page:
                on_page(done, len(pages), index + 1, from_cache)
    return pages, reader.failures, cached


# ---------------------------------------------------------------------------
# 질문 · 정답 생성
# ---------------------------------------------------------------------------


class QADraft(BaseModel):
    question: str = Field(description="문서 내용만으로 답할 수 있는 질문")
    answer: str = Field(description="문서 표현을 살린 한두 문장의 정답")
    page: int = Field(description="정답의 근거가 되는 페이지 번호")


class QADraftSet(BaseModel):
    items: list[QADraft]


@dataclass
class QAItem:
    id: int
    question: str
    answer: str
    page: int
    note: str = ""  # 검수 메모 (예: 여러 페이지에 걸친 내용)
    # 같은 질문을 문서 표현을 피해 사용자 말투로 바꾼 것. 생성된 질문은 문서 문장을 그대로 따라 써서
    # 검색이 실제보다 쉬워지므로, 같은 정답으로 이 질문도 평가해 현실적인 난이도를 봄
    paraphrase: str = ""
    # 문서에 답이 없는 질문 (정답: config.UNANSWERABLE, page 0). 지어내지 않고 모른다고 하는지(거절 정확도) 측정
    answerable: bool = True


def generate_qa(llm: BaseChatModel, pages: list[Document], count: int) -> list[QAItem]:
    body = "\n\n".join(f"[p.{p.metadata['page']}]\n{p.page_content}" for p in pages if p.page_content.strip())
    drafts = llm.with_structured_output(QADraftSet).invoke(QA_GENERATION_PROMPT.format(count=count, pages=body))
    return [QAItem(id=i, **d.model_dump()) for i, d in enumerate(drafts.items[:count], start=1)]


def load_qa(path: Path) -> list[QAItem]:
    return [QAItem(**item) for item in json.loads(path.read_text())]


def save_qa(path: Path, items: list[QAItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2) + "\n")


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

Verdict = Literal["correct", "partial", "incorrect"]
VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0, "error": 0.0}


class Judgement(BaseModel):
    verdict: Verdict
    reason: str = Field(description="판정 이유 한 문장")


class ClaimCheck(BaseModel):
    claim: str = Field(description="답변 속 사실 주장 하나")
    supported: bool = Field(description="문서에 의해 뒷받침되면 true")


class FaithfulnessCheck(BaseModel):
    claims: list[ClaimCheck]


@dataclass
class EvalResult:
    id: int
    question: str
    gold: str
    gold_page: int
    answer: str = ""
    verdict: str = "error"
    reason: str = ""
    retrieved: list[dict] = field(default_factory=list)  # 순위별 page · page_end · 거리 · chunk_index
    page_hit: bool = False  # 정답 페이지가 검색된 청크 안에 있었는지
    gold_rank: int = 0  # 정답 페이지를 포함한 첫 청크의 순위 (1부터, 못 찾으면 0)
    answerable: bool = True
    # 근거 충실도: 답변 주장 중 검색된 문맥에 근거한 비율 (주장이 없으면 None, 예: "찾을 수 없습니다")
    faithfulness: float | None = None
    unsupported: list[str] = field(default_factory=list)  # 문맥에 근거가 없는 주장
    latency: float = 0.0  # 답변 생성 시간(초), 검색 · 채점 · 할당량 대기 제외
    tokens: int = 0


def judge(llm: BaseChatModel, question: str, gold: str, answer: str) -> Judgement:
    return llm.with_structured_output(Judgement).invoke(JUDGE_PROMPT.format(question=question, gold=gold, answer=answer))


def check_faithfulness(llm: BaseChatModel, context: str, answer: str) -> tuple[float | None, list[str]]:
    """답변을 주장 단위로 나눠 문맥 근거 여부를 판정. 반환: (근거 있는 주장 비율, 근거 없는 주장 목록)."""
    checked = llm.with_structured_output(FaithfulnessCheck).invoke(
        FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    )
    if not checked.claims:
        return None, []
    supported = sum(c.supported for c in checked.claims)
    return round(supported / len(checked.claims), 3), [c.claim for c in checked.claims if not c.supported]


def _covers(doc: Document, page: int) -> bool:
    start = doc.metadata.get("page", -1)
    return start <= page <= doc.metadata.get("page_end", start)


RATE_LIMIT_WAITS = (10, 30, 60)  # 할당량 초과(429) 시 기다릴 초 — 평가 점수에 API 오류가 섞이지 않게


def _with_rate_limit_retry(call):
    for wait in (*RATE_LIMIT_WAITS, None):
        try:
            return call()
        except Exception as e:
            if wait is None or "RESOURCE_EXHAUSTED" not in str(e) and "429" not in str(e):
                raise
            time.sleep(wait)


def evaluate_one(
    item: QAItem,
    store: PGVector,
    answer_llm: BaseChatModel,
    judge_llm: BaseChatModel,
    instructions: str,
    k: int,
    use_paraphrase: bool = False,
) -> EvalResult:
    question = item.paraphrase if use_paraphrase and item.paraphrase else item.question
    result = EvalResult(item.id, question, item.answer, item.page, answerable=item.answerable)
    try:
        hits = _with_rate_limit_retry(
            lambda: rag.search_chunks(store, rag.embed_question(store.embeddings, question), k)
        )
        docs = [doc for doc, _ in hits]
        started = time.perf_counter()  # 할당량 대기 시간이 섞이지 않게 답변 호출만 잼
        reply = _with_rate_limit_retry(
            lambda: rag.generate_answer(answer_llm, rag.build_prompt(instructions, docs, question))
        )
        result.latency = round(time.perf_counter() - started, 2)
        result.answer = reply.text
        result.tokens = (reply.usage_metadata or {}).get("total_tokens", 0)
        result.retrieved = [
            {
                "page": doc.metadata.get("page"),
                "page_end": doc.metadata.get("page_end", doc.metadata.get("page")),
                "distance": round(score, 4),
                "chunk": doc.metadata.get("chunk_index"),
            }
            for doc, score in hits
        ]
        result.gold_rank = next((rank for rank, doc in enumerate(docs, start=1) if _covers(doc, item.page)), 0)
        result.page_hit = result.gold_rank > 0
        judgement = _with_rate_limit_retry(lambda: judge(judge_llm, question, item.answer, result.answer))
        result.verdict, result.reason = judgement.verdict, judgement.reason
        result.faithfulness, result.unsupported = _with_rate_limit_retry(
            lambda: check_faithfulness(judge_llm, rag.format_context(docs), result.answer)
        )
    except Exception as e:  # 한 문항의 실패가 전체 평가를 멈추지 않게
        result.verdict, result.reason = "error", f"{type(e).__name__}: {str(e)[:200]}"
    return result


def evaluate(
    items: list[QAItem],
    store: PGVector,
    answer_llm: BaseChatModel,
    judge_llm: BaseChatModel,
    instructions: str,
    k: int,
    use_paraphrase: bool = False,
    concurrency: int = EVAL_CONCURRENCY,
    on_done: Callable[[int, EvalResult], None] | None = None,
) -> list[EvalResult]:
    """질문들을 concurrency개씩 동시에 평가. on_done(끝난 개수, 결과)은 호출한 스레드에서 불림 (화면 진행 표시용)."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(evaluate_one, item, store, answer_llm, judge_llm, instructions, k, use_paraphrase): item.id
            for item in items
        }
        results = []
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if on_done:
                on_done(done, results[-1])
    return sorted(results, key=lambda r: r.id)


def summarize(results: list[EvalResult]) -> dict:
    """답이 있는 질문으로 점수 · 검색 지표를, 답이 없는 질문으로 거절 정확도를 따로 집계."""
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    n = len(answerable) or 1
    faithful = [r.faithfulness for r in results if r.faithfulness is not None]
    return {
        "questions": len(results),
        "score": round(sum(VERDICT_SCORE[r.verdict] for r in answerable) / n, 3),  # correct 1 · partial 0.5
        **{v: sum(r.verdict == v for r in answerable) for v in VERDICT_SCORE},
        "page_hit_rate": round(sum(r.page_hit for r in answerable) / n, 3),
        # 정답 페이지 청크 순위의 역수 평균 (1위면 1, 2위면 0.5, 못 찾으면 0)
        "mrr": round(sum(1 / r.gold_rank for r in answerable if r.gold_rank) / n, 3),
        # 답이 없는 질문에 "찾을 수 없다"고 답한 비율
        "refusal_accuracy": round(sum(r.verdict == "correct" for r in unanswerable) / len(unanswerable), 3)
        if unanswerable else None,
        "faithfulness": round(sum(faithful) / len(faithful), 3) if faithful else None,
        "unsupported_answers": sum(bool(r.unsupported) for r in results),
        "avg_latency": round(sum(r.latency for r in results) / (len(results) or 1), 2),
        "avg_tokens": round(sum(r.tokens for r in results) / (len(results) or 1)),
    }
