#!/usr/bin/env python3
"""LLM 모델별 속도(TTFT, 총 소요시간, 토큰당 속도) 및 비용 비교 벤치마크 CLI (RAG 파이프라인 지원)."""

import argparse
import sys
import time
import uuid
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가하여 단독 실행 가능하도록 지원
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_app import benchmark_service, billing, config
from rag_app import pipeline as rag


DEFAULT_PROMPT = (
    "문서에 기반하여 가장 핵심적인 3가지 내용을 요약해 주세요."
)


def main():
    parser = argparse.ArgumentParser(description="LLM 모델별 RAG 및 프롬프트 성능 벤치마크")
    parser.add_argument(
        "--prompt", "-p", default=DEFAULT_PROMPT, help="테스트할 질문"
    )
    parser.add_argument(
        "--max-tokens", "-m", type=int, default=512, help="최대 출력 토큰 수"
    )
    parser.add_argument(
        "--rag", action="store_true", help="현재 pgvector 컬렉션에서 문서를 검색하여 RAG 컨텍스트로 주입"
    )
    parser.add_argument(
        "--collection", "-c", default=config.COLLECTION_NAME, help="RAG 컬렉션 이름"
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=4, help="검색할 청크 수"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 LLM 모델 성능 및 비용 비교 벤치마크 시작")
    print(f"모드: {'📄 RAG 파이프라인 (문서 검색 + LLM 답변)' if args.rag else '💬 일반 프롬프트'}")
    print(f"질문: {args.prompt}")
    print(f"최대 토큰: {args.max_tokens}")
    if args.rag:
        print(f"컬렉션: {args.collection} (top-k: {args.top_k})")
    print("=" * 80)

    messages = None
    search_time = 0.0
    cli_run_id = f"cli_{int(time.time())}_{uuid.uuid4().hex[:4]}"

    if args.rag:
        print("\n🔍 1단계: PostgreSQL pgvector에서 관련 문서 청크 검색 중...")
        try:
            t0 = time.perf_counter()
            embeddings = rag.create_embeddings(config.EMBEDDING_MODEL, config.EMBEDDING_DIMENSION)
            store = rag.create_vector_store(args.collection, embeddings)
            q_vec = rag.embed_question(embeddings, args.prompt)
            t1 = time.perf_counter()
            search_time = max(0.001, t1 - t0)
            # 질문 임베딩 과금 기록
            billing.record_billing_event(
                stage="질문 임베딩",
                model_id=config.EMBEDDING_MODEL,
                input_tokens=max(1, int(len(args.prompt) / 2.2)),
                output_tokens=0,
                collection_name=args.collection,
                run_id=cli_run_id,
                latency_sec=round(search_time, 3),
                top_k=args.top_k,
            )
            hits = rag.search_chunks(store, q_vec, args.top_k)
            print(f"  • 검색 완료: {len(hits)}개 청크 검색됨 ({search_time:.3f}초 소요)")
            messages = rag.build_prompt(config.SYSTEM_PROMPT, [d for d, _ in hits], args.prompt)
            total_chars = sum(len(str(m.content)) for m in messages)
            print(f"  • RAG 프롬프트 크기: 약 {total_chars:,}자 (문맥 주입 완료)")
        except Exception as e:
            print(f"  ⚠️ RAG 문서 검색 실패: {e}")
            print("  -> 일반 프롬프트 모드로 전환합니다.")
            messages = None
            search_time = 0.0

    target_models = [
        config.LLM_DEEPSEEK_FLASH,
        config.LLM_GEMMA_26B,
        config.LLM_GEMINI_FLASH_LITE,
        config.LLM_CLAUDE_HAIKU,
        config.LLM_GEMINI,
    ]

    results = []
    for model_key in target_models:
        res = benchmark_service.run_single_benchmark(
            choice=model_key,
            prompt=args.prompt,
            messages=messages,
            search_time_sec=search_time,
            max_output_tokens=args.max_tokens,
            collection_name=args.collection if args.rag else "-",
            run_id=cli_run_id,
            top_k=args.top_k if args.rag else 0,
        )
        status = "❌ 실패" if res.error else "✅ 성공"
        print(f"\n▶ [{res.model_name}] ({res.model_id}) - {status}")
        if res.error:
            print(f"  오류 내용: {res.error}")
            continue
        print(f"  • 첫 토큰 도착 (TTFT): {res.ttft_sec:.2f} 초")
        print(f"  • 총 응답 소요 시간:   {res.total_time_sec:.2f} 초 (LLM생성 {res.gen_time_sec:.2f}초 + 검색 {search_time:.2f}초)")
        print(f"  • 출력 생성 속도:       {res.throughput:.1f} 토큰/초")
        print(f"  • 토큰 사용량:         입력 {res.input_tokens:,} / 출력 {res.output_tokens:,} (총 {res.input_tokens + res.output_tokens:,})")
        print(f"  • 1회 호출 예상 비용:   ${res.cost_usd:.5f} (약 ₩{res.cost_krw:.2f})")
        print(f"  • 1,000회 기준 비용:    ${res.cost_1k_usd:.3f} (약 ₩{res.cost_1k_krw:,.0f})")
        print("  • 응답 내용 미리보기:")
        preview = res.response_text.strip().replace("\n", " ")[:120]
        print(f"    \"{preview}...\"")
        results.append(res)

    print("\n" + "=" * 80)
    print("📊 종합 성능 비교 요약표")
    print("=" * 80)
    header = f"{'모델명':<26} | {'TTFT(초)':<8} | {'총시간(초)':<10} | {'속도(t/s)':<9} | {'1회비용(원)':<10} | {'1천회비용($)':<12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.model_name:<26} | {r.ttft_sec:<8.2f} | {r.total_time_sec:<10.2f} | {r.throughput:<9.1f} | ₩{r.cost_krw:<9.2f} | ${r.cost_1k_usd:<12.3f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
