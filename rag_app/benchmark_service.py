"""RAG 앱 및 LLM 모델별 성능(지연 시간, 속도, 토큰 수, 비용) 측정 서비스."""

import time
from dataclasses import dataclass
from typing import Callable, Generator

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_postgres import PGVector

from rag_app import billing, config
from rag_app import pipeline as rag

EXCHANGE_RATE_KRW = 1350.0  # 1 USD = 1,350 KRW (참고 환율)


@dataclass
class BenchmarkResult:
    model_key: str
    model_name: str
    model_id: str
    prompt: str
    ttft_sec: float  # Time To First Token (초)
    total_time_sec: float  # 총 소요 시간 (초, RAG의 경우 검색+생성)
    gen_time_sec: float  # LLM 생성 소요 시간 (초)
    throughput: float  # 초당 출력 토큰 수 (tokens/sec)
    input_tokens: int
    output_tokens: int
    cost_usd: float  # 1회 호출 비용 ($)
    cost_krw: float  # 1회 호출 비용 (원)
    cost_1k_usd: float  # 1,000회 호출 비용 ($)
    cost_1k_krw: float  # 1,000회 호출 비용 (원)
    response_text: str
    error: str | None = None


@dataclass
class RagBenchmarkSummary:
    question: str
    retrieval_time_sec: float
    retrieved_chunks: list[tuple[Document, float]]
    model_results: list[BenchmarkResult]


def estimate_tokens(text: str) -> int:
    """토큰 메타데이터가 없을 경우의 글자수 기반 근사치 계산 (한국어/영문 혼합 기준 약 2.2자 = 1토큰)."""
    if not text:
        return 0
    return max(1, int(len(text) / 2.2))


def get_model_identifier(choice: str) -> str:
    mapping = {
        config.LLM_DEEPSEEK_FLASH: config.DEEPSEEK_FLASH_MODEL,
        config.LLM_GEMMA_26B: config.GEMMA_26B_MODEL,
        config.LLM_GEMINI_FLASH_LITE: config.GEMINI_FLASH_LITE_MODEL,
        config.LLM_CLAUDE_HAIKU: config.CLAUDE_HAIKU_MODEL,
        config.LLM_GEMINI: config.LLM_MODEL,
        config.LLM_OLLAMA_GEMMA: config.OLLAMA_MODEL,
    }
    return mapping.get(choice, config.LLM_MODEL)


def calculate_cost(model_id_or_key: str, input_tokens: int, output_tokens: int) -> dict[str, float]:
    """입출력 토큰 수와 모델 식별자를 바탕으로 1회 및 1,000회 비용을 계산합니다."""
    model_id = get_model_identifier(model_id_or_key)
    pricing = config.MODEL_PRICING.get(model_id, {"input": 0.15, "output": 0.60})

    cost_in = (input_tokens / 1_000_000.0) * pricing.get("input", 0.0)
    cost_out = (output_tokens / 1_000_000.0) * pricing.get("output", 0.0)
    cost_usd = cost_in + cost_out
    cost_krw = cost_usd * EXCHANGE_RATE_KRW

    return {
        "cost_usd": cost_usd,
        "cost_krw": cost_krw,
        "cost_1k_usd": cost_usd * 1000.0,
        "cost_1k_krw": cost_krw * 1000.0,
    }


def run_single_benchmark(
    choice: str,
    prompt: str = "",
    messages: list[BaseMessage] | None = None,
    search_time_sec: float = 0.0,
    max_output_tokens: int = 1024,
    on_chunk: Callable[[str], None] | None = None,
) -> BenchmarkResult:
    """단일 모델에 대해 스트리밍으로 호출하여 지연시간, 처리량, 토큰수, 비용을 측정합니다."""
    model_id = get_model_identifier(choice)
    model_display_name = config.LLM_OPTIONS.get(choice, choice)

    if messages is None:
        messages = [HumanMessage(content=prompt)]
    prompt_preview = prompt or (messages[-1].content if messages else "")

    start_time = time.perf_counter()
    first_token_time = None
    chunks: list[str] = []
    input_tokens = 0
    output_tokens = 0
    error_msg = None

    try:
        llm = rag.create_answer_llm(choice, max_output_tokens=max_output_tokens)

        for chunk in llm.stream(messages):
            now = time.perf_counter()
            content = chunk.content if hasattr(chunk, "content") else str(chunk)

            if isinstance(content, list):
                text_part = "".join(str(p.get("text", p) if isinstance(p, dict) else p) for p in content)
            else:
                text_part = str(content)

            if text_part and first_token_time is None:
                first_token_time = now

            chunks.append(text_part)
            if on_chunk and text_part:
                on_chunk(text_part)

            # usage_metadata 확인
            usage = getattr(chunk, "usage_metadata", None)
            if usage and isinstance(usage, dict):
                if usage.get("input_tokens"):
                    input_tokens = usage["input_tokens"]
                if usage.get("output_tokens"):
                    output_tokens = usage["output_tokens"]

        end_time = time.perf_counter()

    except Exception as e:
        end_time = time.perf_counter()
        error_msg = str(e)

    full_text = "".join(chunks)
    gen_time = max(0.001, end_time - start_time)
    total_time = gen_time + search_time_sec
    ttft = (first_token_time - start_time) if first_token_time else gen_time

    # usage_metadata가 누락된 경우 글자 수 기반 추정
    if input_tokens == 0:
        total_prompt_chars = sum(len(str(m.content)) for m in messages)
        input_tokens = estimate_tokens("a" * total_prompt_chars)
    if output_tokens == 0 and full_text:
        output_tokens = estimate_tokens(full_text)

    # 출력 생성 속도 (TTFT 제외)
    pure_gen_time = max(0.001, gen_time - ttft)
    throughput = (output_tokens / pure_gen_time) if output_tokens > 0 else (output_tokens / gen_time)

    # 비용 산출
    costs = calculate_cost(model_id, input_tokens, output_tokens)

    # 성공 시 과금 카운터에 기록 (HTTP 200 성공 응답만 과금)
    if error_msg is None:
        billing.record_billing_event(
            stage="벤치마크 테스트",
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    return BenchmarkResult(
        model_key=choice,
        model_name=model_display_name,
        model_id=model_id,
        prompt=str(prompt_preview),
        ttft_sec=round(ttft, 3),
        total_time_sec=round(total_time, 3),
        gen_time_sec=round(gen_time, 3),
        throughput=round(throughput, 1),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=costs["cost_usd"],
        cost_krw=round(costs["cost_krw"], 3),
        cost_1k_usd=round(costs["cost_1k_usd"], 4),
        cost_1k_krw=round(costs["cost_1k_krw"], 1),
        response_text=full_text,
        error=error_msg,
    )


def run_rag_benchmark(
    question: str,
    store: PGVector,
    embeddings: Embeddings,
    instructions: str,
    top_k: int = 4,
    models: list[str] | None = None,
    max_output_tokens: int = 1024,
) -> RagBenchmarkSummary:
    """RAG 파이프라인(질문 임베딩 -> 청크 검색 -> 프롬프트 빌드 -> 모델별 생성) 전체 성능을 실측합니다."""
    t_search_start = time.perf_counter()
    query_vector = rag.embed_question(embeddings, question)
    hits = rag.search_chunks(store, query_vector, top_k)
    t_search_end = time.perf_counter()
    search_time = max(0.001, t_search_end - t_search_start)

    docs = [doc for doc, _ in hits]
    messages = rag.build_prompt(instructions, docs, question)

    target_models = models or [
        config.LLM_DEEPSEEK_FLASH,
        config.LLM_GEMMA_26B,
        config.LLM_GEMINI_FLASH_LITE,
        config.LLM_CLAUDE_HAIKU,
        config.LLM_GEMINI,
    ]

    results = []
    for model_key in target_models:
        res = run_single_benchmark(
            choice=model_key,
            prompt=question,
            messages=messages,
            search_time_sec=search_time,
            max_output_tokens=max_output_tokens,
        )
        results.append(res)

    return RagBenchmarkSummary(
        question=question,
        retrieval_time_sec=round(search_time, 3),
        retrieved_chunks=hits,
        model_results=results,
    )


def run_all_benchmarks(
    prompt: str,
    models: list[str] | None = None,
    max_output_tokens: int = 1024,
) -> Generator[BenchmarkResult, None, None]:
    """선택한 모든 모델을 순차 테스트하며 결과를 yield합니다."""
    target_models = models or [
        config.LLM_DEEPSEEK_FLASH,
        config.LLM_GEMMA_26B,
        config.LLM_GEMINI_FLASH_LITE,
        config.LLM_CLAUDE_HAIKU,
        config.LLM_GEMINI,
    ]
    for model_key in target_models:
        yield run_single_benchmark(model_key, prompt, max_output_tokens=max_output_tokens)
