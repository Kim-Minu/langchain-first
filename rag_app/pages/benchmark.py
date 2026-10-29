"""RAG 앱 및 LLM 모델별 성능 · 속도 · 비용 비교 벤치마크 화면."""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from rag_app import benchmark_service, billing, config
from rag_app import pipeline as rag
from rag_app.ui_common import get_embeddings, get_store, render_sidebar, show_error

st.set_page_config(page_title="RAG & LLM 성능 벤치마크", layout="wide")

# 사이드바 공통 설정 렌더링
settings = render_sidebar(page="benchmark")

st.title("⚡ RAG 앱 & LLM 모델별 성능 · 비용 벤치마크")
st.caption(
    "DeepSeek V4.1 Flash, Gemma 4 26B MaaS, Gemini 3.5 Flash Lite, Claude Haiku 5.5, Gemini 3.8 Flash 모델이 "
    "**RAG 환경(실제 문서 청크 검색 + 프롬프트 컨텍스트)** 및 **일반 질의**에서 나타내는 속도, 지연시간, 토큰량, 1회 비용을 실측 비교합니다."
)

st.divider()

bench_mode = st.radio(
    "벤치마크 대상 모드",
    ["📄 RAG 파이프라인 벤치마크 (실제 문서 검색 + LLM 답변)", "💬 일반 LLM 질의 벤치마크 (단독 프롬프트)"],
    horizontal=True,
)

# 컬렉션 정보 확인
try:
    info = rag.get_collection_info(settings.collection) or {}
except Exception:
    info = {}
stored_chunks = info.get("stored_chunks", 0)

if "RAG 파이프라인" in bench_mode:
    if stored_chunks == 0:
        st.warning(
            f"현재 컬렉션 `{settings.collection}`에 인덱싱된 문서가 없습니다. "
            "메인 화면('PDF RAG 실습')에서 PDF를 인덱싱한 후 실행하거나, '일반 LLM 질의 벤치마크'를 선택하세요."
        )
    else:
        st.success(
            f"현재 연결된 컬렉션: `{settings.collection}` (인덱싱된 청크: **{stored_chunks}개**, top-k: **{settings.top_k}**)"
        )

col1, col2 = st.columns([3, 1])
with col1:
    default_q = (
        "인덱싱된 문서의 핵심 내용을 3가지 포인트로 정리해 주세요."
        if "RAG 파이프라인" in bench_mode
        else "대한민국의 인공지능(AI) 반도체 국가 전략의 핵심 3가지를 각 1문장씩 일목요연하게 설명해 주세요."
    )
    test_prompt = st.text_area(
        "테스트 질문 (Question)",
        value=default_q,
        height=90,
        help="모든 모델에 동일하게 전송할 질문입니다.",
    )
with col2:
    max_tokens = st.number_input(
        "최대 출력 토큰", min_value=128, max_value=4096, value=512, step=128
    )
    can_run = stored_chunks > 0 if "RAG 파이프라인" in bench_mode else True
    run_btn = st.button("🚀 벤치마크 실행", type="primary", width="stretch", disabled=not can_run)

if run_btn and test_prompt.strip():
    target_models = [
        config.LLM_DEEPSEEK_FLASH,
        config.LLM_GEMMA_26B,
        config.LLM_GEMINI_FLASH_LITE,
        config.LLM_CLAUDE_HAIKU,
        config.LLM_GEMINI,
    ]

    messages_to_send = None
    search_time = 0.0
    use_rag = "RAG 파이프라인" in bench_mode
    bench_run_id = f"bench_{int(pd.Timestamp.now().timestamp())}_{uuid.uuid4().hex[:4]}"

    if use_rag:
        with st.status("1단계: RAG 질문 임베딩 및 관련 문서 청크 검색 중…", expanded=True) as status:
            try:
                embeddings = get_embeddings(settings.embedding_model, settings.embedding_dim)
                store = get_store(settings.embedding_model, settings.embedding_dim, settings.collection)
                
                t_s_start = pd.Timestamp.now().timestamp()
                q_vec = rag.embed_question(embeddings, test_prompt)
                hits = rag.search_chunks(store, q_vec, settings.top_k)
                t_s_end = pd.Timestamp.now().timestamp()
                search_time = max(0.001, t_s_end - t_s_start)

                # 질문 임베딩 과금 기록
                billing.record_billing_event(
                    stage="질문 임베딩",
                    model_id=settings.embedding_model,
                    input_tokens=max(1, int(len(test_prompt) / 2.2)),
                    output_tokens=0,
                    collection_name=settings.collection,
                    run_id=bench_run_id,
                    prompt=test_prompt,
                    latency_sec=round(search_time, 3),
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    top_k=settings.top_k,
                )

                docs = [doc for doc, _ in hits]
                messages_to_send = rag.build_prompt(settings.instructions, docs, test_prompt)
                status.update(
                    label=f"1단계 완료: 청크 {len(hits)}개 검색 완료 ({search_time:.3f}초 소요)",
                    state="complete",
                    expanded=False,
                )
                with st.expander(f"검색된 청크 {len(hits)}개 (모든 모델에 컨텍스트로 제공됨)"):
                    for i, (d, score) in enumerate(hits, 1):
                        st.markdown(f"**[{i}] p.{d.metadata.get('page')} (거리 {score:.4f})**\n```\n{d.page_content[:200]}...\n```")
            except Exception as e:
                status.update(label="문서 검색 실패", state="error")
                show_error("문서 검색", e)
                st.stop()

    st.info("2단계: 5개 모델에 대해 RAG 답변 생성 및 실시간 성능을 측정합니다...")
    progress_bar = st.progress(0.0)
    status_holder = st.empty()

    results = []
    total_models = len(target_models)

    for idx, model_key in enumerate(target_models):
        res = benchmark_service.run_single_benchmark(
            choice=model_key,
            prompt=test_prompt,
            messages=messages_to_send,
            search_time_sec=search_time,
            max_output_tokens=max_tokens,
            collection_name=settings.collection if use_rag else "-",
            run_id=bench_run_id,
            chunk_size=settings.chunk_size if use_rag else 0,
            chunk_overlap=settings.chunk_overlap if use_rag else 0,
            top_k=settings.top_k if use_rag else 0,
        )
        results.append(res)
        progress_bar.progress((idx + 1) / total_models)
        status_holder.write(f"진행 중: **{res.model_name}** 완료 ({idx + 1}/{total_models})")

    status_holder.success("🎉 모든 모델의 벤치마크 측정이 완료되었습니다!")

    # 요약 테이블 생성
    table_data = []
    for r in results:
        table_data.append(
            {
                "모델": r.model_name,
                "식별자": r.model_id,
                "첫 토큰 (TTFT)": f"{r.ttft_sec:.2f}초",
                "전체 소요 시간": f"{r.total_time_sec:.2f}초",
                "생성 속도": f"{r.throughput:.1f} t/s",
                "입력 토큰 (문맥포함)": f"{r.input_tokens:,}",
                "출력 토큰": f"{r.output_tokens:,}",
                "1회 질의 비용": f"₩{r.cost_krw:,.2f}",
                "1,000회 비용": f"₩{r.cost_1k_krw:,.0f} (${r.cost_1k_usd:,.3f})",
                "상태": "❌ 오류" if r.error else "✅ 정상",
                "_ttft": r.ttft_sec,
                "_total_time": r.total_time_sec,
                "_throughput": r.throughput,
                "_cost_krw": r.cost_krw,
            }
        )

    df = pd.DataFrame(table_data)

    st.subheader("📊 RAG 종합 성능 비교표")
    if search_time > 0:
        st.caption(f"※ 전체 소요 시간 = 문서 검색({search_time:.2f}초) + 각 모델 LLM 답변 생성 시간")

    st.dataframe(
        df[
            [
                "모델",
                "식별자",
                "첫 토큰 (TTFT)",
                "전체 소요 시간",
                "생성 속도",
                "입력 토큰 (문맥포함)",
                "출력 토큰",
                "1회 질의 비용",
                "1,000회 비용",
                "상태",
            ]
        ],
        width="stretch",
    )

    # 시각화 차트
    st.subheader("📈 시각화 비교")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.caption("초당 생성 토큰 수 (높을수록 빠름)")
        chart_speed = df.set_index("모델")[["_throughput"]].rename(
            columns={"_throughput": "토큰/초"}
        )
        st.bar_chart(chart_speed)

    with c2:
        st.caption("첫 토큰 지연시간 TTFT (낮을수록 첫 반응 빠름)")
        chart_ttft = df.set_index("모델")[["_ttft"]].rename(
            columns={"_ttft": "TTFT (초)"}
        )
        st.bar_chart(chart_ttft)

    with c3:
        st.caption("1회 호출당 예상 비용 (낮을수록 경제적)")
        chart_cost = df.set_index("모델")[["_cost_krw"]].rename(
            columns={"_cost_krw": "비용 (원)"}
        )
        st.bar_chart(chart_cost)

    # 모델별 실제 응답 텍스트 확인
    st.subheader("💬 모델별 실제 RAG 답변 내용")
    for r in results:
        with st.expander(
            f"{r.model_name} (총 {r.total_time_sec:.2f}초 소요, {r.output_tokens:,} 토큰 생성, ₩{r.cost_krw:,.2f})"
        ):
            if r.error:
                st.error(f"오류: {r.error}")
            else:
                st.markdown(r.response_text)
