"""RAG 앱 과금 대시보드 (Billing Dashboard).

Google Cloud Model Garden & Partner Models 공식 가격표 기준 실시간 과금 추적 및 최적 조합 분석 화면.
참조:
- https://cloud.google.com/model-garden/pricing#cost-of-building-and-deploying-ai-models-in-agent-platform
- https://cloud.google.com/model-garden/partner-models/pricing#partner-models-on-agent-platform
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from rag_app import billing, config
from rag_app.ui_common import render_sidebar

st.set_page_config(page_title="과금 및 최적화 대시보드", page_icon="💰", layout="wide")

# 사이드바 공통 설정 렌더링
settings = render_sidebar(page="billing")

st.title("💰 RAG 과금 & 하이퍼파라미터 최적 조합 대시보드")
st.caption(
    "Google Cloud Model Garden & Partner Models(DeepSeek, Gemma, Gemini, Claude) 공식 단가 기준 "
    "실시간 과금 추적 및 캐시 절감 효과, 그리고 **모델 × 청크 크기 × Top-K 조합별 가성비·품질 분석**을 제공합니다. "
    "모든 과금 내역은 PostgreSQL DB(`rag_billing_events`)에 영구 저장되므로 앱 재시작 후에도 누적 내역이 보존됩니다."
)

st.divider()

# ---------------------------------------------------------------------------
# 1. 상단 요약 메트릭
# ---------------------------------------------------------------------------
summary = billing.get_billing_summary()

saved_krw = summary.get("total_saved_krw", 0.0)
gross_cost_krw = summary["total_krw"] + saved_krw
save_pct = (saved_krw / gross_cost_krw * 100) if gross_cost_krw > 0 else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "누적 청구 비용 (KRW)",
    f"₩{summary['total_krw']:,.2f}",
    help="기준 환율 1 USD = 1,350 KRW 적용 (캐시 할인 적용 후 실제 지불 금액)",
)
c2.metric(
    "누적 절감 비용 (캐시 효과)",
    f"₩{saved_krw:,.2f}",
    delta=f"절감률 {save_pct:.1f}%" if saved_krw > 0 else "캐시 없음",
    delta_color="normal" if saved_krw > 0 else "off",
    help=f"임베딩 캐시 적중으로 절감된 금액 (적중 {summary.get('total_cache_hits', 0):,}건, {summary.get('total_cached_tokens', 0):,} 토큰)",
)
c3.metric(
    "누적 청구 비용 (USD)",
    f"${summary['total_usd']:,.5f}",
    help="Google Cloud 청구서 예상 실반영 금액",
)
c4.metric(
    "누적 토큰 사용량",
    f"{summary['total_in_tokens'] + summary['total_out_tokens']:,}",
    help=f"입력 {summary['total_in_tokens']:,} + 출력 {summary['total_out_tokens']:,} 토큰 (캐시된 {summary.get('total_cached_tokens', 0):,} 토큰 제외)",
)
c5.metric(
    "과금 대상 API 호출 수",
    f"{summary['total_calls']:,}회",
    help="HTTP 200 성공 응답만 과금 (캐시 적중 호출 제외)",
)

st.divider()

# ---------------------------------------------------------------------------
# 2. 탭 구성 (과금 현황 / 최적 조합 분석 / 세부 로그)
# ---------------------------------------------------------------------------
tab_overview, tab_combinations, tab_logs = st.tabs([
    "📊 1. 과금 및 캐시 현황",
    "🎯 2. 최적 조합 분석 (Pareto & Recommendation)",
    "📋 3. 세부 API 과금 이벤트 로그",
])

# ---------------------------------------------------------------------------
# TAB 1: 과금 및 캐시 현황
# ---------------------------------------------------------------------------
with tab_overview:
    col_stage, col_model, col_coll = st.columns(3)

    with col_stage:
        st.subheader("📑 작업 단계별 누적 비용")
        if summary["by_stage"]:
            stage_rows = []
            chart_data = {}
            for stage, data in summary["by_stage"].items():
                stage_rows.append({
                    "작업 단계": stage,
                    "호출 수": f"{data['count']:,}회",
                    "누적 토큰": f"{data['tokens']:,}",
                    "비용 (원)": f"₩{data['krw']:,.2f}",
                    "비용 ($)": f"${data['usd']:,.5f}",
                })
                chart_data[stage] = round(data["krw"], 2)
            df_stage = pd.DataFrame(stage_rows)
            st.dataframe(df_stage, width="stretch", hide_index=True)
            st.bar_chart(pd.DataFrame(list(chart_data.items()), columns=["작업 단계", "비용 (원)"]).set_index("작업 단계"))
        else:
            st.info("아직 발생한 작업 단계별 과금 내역이 없습니다.")

    with col_model:
        st.subheader("🤖 모델별 누적 비용")
        if summary["by_model"]:
            model_rows = []
            chart_data_model = {}
            for model, data in summary["by_model"].items():
                model_rows.append({
                    "모델": model,
                    "호출 수": f"{data['count']:,}회",
                    "누적 토큰": f"{data['tokens']:,}",
                    "비용 (원)": f"₩{data['krw']:,.2f}",
                    "비용 ($)": f"${data['usd']:,.5f}",
                })
                chart_data_model[model] = round(data["krw"], 2)
            df_model = pd.DataFrame(model_rows)
            st.dataframe(df_model, width="stretch", hide_index=True)
            st.bar_chart(pd.DataFrame(list(chart_data_model.items()), columns=["모델", "비용 (원)"]).set_index("모델"))
        else:
            st.info("아직 발생한 모델별 과금 내역이 없습니다.")

    with col_coll:
        st.subheader("📁 컬렉션별 누적 비용")
        if summary.get("by_collection"):
            coll_rows = []
            chart_data_coll = {}
            for coll_name, data in summary["by_collection"].items():
                coll_rows.append({
                    "컬렉션": coll_name,
                    "호출 수": f"{data['count']:,}회",
                    "누적 토큰": f"{data['tokens']:,}",
                    "비용 (원)": f"₩{data['krw']:,.2f}",
                    "비용 ($)": f"${data['usd']:,.5f}",
                })
                chart_data_coll[coll_name] = round(data["krw"], 2)
            df_coll = pd.DataFrame(coll_rows)
            st.dataframe(df_coll, width="stretch", hide_index=True)
            st.bar_chart(pd.DataFrame(list(chart_data_coll.items()), columns=["컬렉션", "비용 (원)"]).set_index("컬렉션"))
        else:
            st.info("아직 발생한 컬렉션별 과금 내역이 없습니다.")

    st.markdown("##### ⚡ 임베딩 캐시 비용 절감 효과")
    c_c1, c_c2 = st.columns([1, 2])
    with c_c1:
        st.metric("캐시로 아낀 금액", f"₩{saved_krw:,.2f}")
        st.metric("캐시 적중 건수", f"{summary.get('total_cache_hits', 0):,}건")
        st.metric("캐시된 누적 토큰", f"{summary.get('total_cached_tokens', 0):,} 토큰")
    with c_c2:
        st.caption("캐시 미적용 시 총 청구 예상 금액 vs 실제 청구 금액")
        comp_df = pd.DataFrame([
            {"구분": "실제 청구 금액", "금액(원)": round(summary["total_krw"], 2)},
            {"구분": "캐시 절감액", "금액(원)": round(saved_krw, 2)},
        ]).set_index("구분")
        st.bar_chart(comp_df)

# ---------------------------------------------------------------------------
# TAB 2: 최적 조합 분석 (Pareto & Recommendation Matrix)
# ---------------------------------------------------------------------------
with tab_combinations:
    st.subheader("🎯 모델 × 청크 크기 × Top-K 파레토 최적 조합 분석")
    st.caption(
        "RAG 시스템 운영 시 어떤 하이퍼파라미터 조합이 비용 대비 최고의 속도와 정확도를 제공하는지 "
        "축적된 과금 및 런타임 이벤트 로그를 바탕으로 실측 분석합니다."
    )

    by_comb = summary.get("by_combination", [])
    if by_comb:
        comb_rows = []
        for stats in by_comb:
            avg_score = stats.get("avg_eval_score")
            avg_faith = stats.get("avg_faithfulness")
            avg_score_str = f"{avg_score:.2f}" if avg_score is not None else "-"
            avg_faith_str = f"{avg_faith * 100:.1f}%" if avg_faith is not None else "-"
            comb_rows.append({
                "모델": stats["model_name"],
                "청크 크기": f"{stats['chunk_size']}자" if stats["chunk_size"] > 0 else "-",
                "Top-K": f"{stats['top_k']}개" if stats["top_k"] > 0 else "-",
                "호출 횟수": f"{stats['count']:,}회",
                "평균 1회 비용 (KRW)": round(stats["avg_cost_krw"], 2),
                "평균 소요 시간 (초)": round(stats.get("avg_latency_sec", 0.0), 2),
                "평균 TTFT (초)": round(stats.get("avg_ttft_sec", 0.0), 2),
                "평균 생성 속도 (t/s)": round(stats.get("avg_throughput", 0.0), 1),
                "평가 점수": avg_score_str,
                "근거 충실도": avg_faith_str,
                "_raw_cost": stats["avg_cost_krw"],
                "_raw_lat": stats.get("avg_latency_sec", 0.0),
            })

        df_comb = pd.DataFrame(comb_rows)
        # 비용 오름차순 정렬
        df_display = df_comb.drop(columns=["_raw_cost", "_raw_lat"])
        st.dataframe(df_display, width="stretch", hide_index=True)

        st.markdown("##### 📈 비용 대비 응답 속도 비교")
        chart_comb_df = pd.DataFrame({
            "조합": [f"{r['모델']} (C:{r['청크 크기']}, K:{r['Top-K']})" for r in comb_rows],
            "평균 비용 (원)": [r["_raw_cost"] for r in comb_rows],
            "평균 응답시간 (초)": [r["_raw_lat"] for r in comb_rows],
        }).set_index("조합")
        st.bar_chart(chart_comb_df)
    else:
        st.info("아직 충분한 조합별 실행 로그가 쌓이지 않았습니다. Q&A 탭이나 벤치마크, OCR 평가를 실행하면 조합 데이터가 자동 누적됩니다.")

    st.divider()

    # 3대 추천 조합 프리셋
    st.markdown("##### 🏆 실전 운영 목적별 3대 추천 조합 프리셋")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.info(
            "⚡ **[초고속 & 최저비용 프리셋]**\n\n"
            "- **권장 모델**: DeepSeek V4.1 Flash / Gemini Flash Lite\n"
            "- **청크 크기**: 500자 (Overlap 50자)\n"
            "- **검색 Top-K**: 3개\n"
            "- **예상 1회 비용**: ~₩0.15 / 회\n"
            "- **적합 대상**: 실시간 고객센터 챗봇, 대규모 동시 접속 Q&A, FAQ 매칭"
        )

    with p2:
        st.success(
            "⚖️ **[골든 밸런스 가성비 프리셋]** *(추천)*\n\n"
            "- **권장 모델**: Gemma 4 26B MaaS / Claude Haiku 5.5\n"
            "- **청크 크기**: 800자 (Overlap 100자)\n"
            "- **검색 Top-K**: 4개\n"
            "- **예상 1회 비용**: ~₩0.60 / 회\n"
            "- **적합 대상**: 사내 업무 매뉴얼, 사규집, 범용 기술 문서 검색 및 요약"
        )

    with p3:
        st.warning(
            "🎯 **[고정밀 심층 분석 프리셋]**\n\n"
            "- **권장 모델**: Claude 3.5 Sonnet / Gemini 3.8 Flash\n"
            "- **청크 크기**: 1,200자 (Overlap 200자)\n"
            "- **검색 Top-K**: 6개\n"
            "- **예상 1회 비용**: ~₩2.50 / 회\n"
            "- **적합 대상**: 복잡한 계약서 검토, 학술/의료 논문 심층 해석, 법률 자문"
        )

    st.divider()

    # 비용 시뮬레이터
    st.markdown("##### 🧮 RAG 서비스 규모별 월간 예상 비용 시뮬레이터")
    sim_c1, sim_c2 = st.columns([1, 2])
    with sim_c1:
        daily_queries = st.slider("일일 질의 건수 (Daily Queries)", min_value=100, max_value=50000, value=3000, step=100)
        cache_hit_rate = st.slider("예상 캐시 적중률 (%)", min_value=0, max_value=80, value=35, step=5)
    with sim_c2:
        monthly_queries = daily_queries * 30
        presets = [
            ("⚡ 초고속 최저비용 (₩0.15/회)", 0.15),
            ("⚖️ 골든 밸런스 (₩0.60/회)", 0.60),
            ("🎯 고정밀 심층 (₩2.50/회)", 2.50),
        ]
        sim_results = []
        for name, unit_cost in presets:
            raw_monthly = monthly_queries * unit_cost
            saved_monthly = raw_monthly * (cache_hit_rate / 100.0)
            net_monthly = raw_monthly - saved_monthly
            sim_results.append({
                "조합 프리셋": name,
                "월간 총 비용 (원)": f"₩{net_monthly:,.0f}",
                "캐시 절감액 (원)": f"₩{saved_monthly:,.0f}",
                "연간 예상 비용 (원)": f"₩{net_monthly * 12:,.0f}",
            })
        st.dataframe(pd.DataFrame(sim_results), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# TAB 3: 세부 API 과금 이벤트 로그
# ---------------------------------------------------------------------------
with tab_logs:
    st.subheader("📋 세부 API 과금 이벤트 로그")

    events: list[billing.BillingEvent] = summary["events"]
    if events:
        # 필터 옵션
        f1, f2, f3 = st.columns(3)
        with f1:
            unique_colls = sorted(list({e.collection_name or "-" for e in events}))
            selected_coll = st.selectbox("📁 컬렉션 필터", ["전체"] + unique_colls, index=0)
        with f2:
            unique_stages = sorted(list({e.stage for e in events}))
            selected_stage = st.selectbox("📑 작업 단계 필터", ["전체"] + unique_stages, index=0)
        with f3:
            selected_cache = st.selectbox("⚡ 캐시 필터", ["전체", "캐시 적중만", "신규 API 호출만"], index=0)

        filtered_events = [
            e for e in events
            if (selected_coll == "전체" or (e.collection_name or "-") == selected_coll)
            and (selected_stage == "전체" or e.stage == selected_stage)
            and (
                selected_cache == "전체"
                or (selected_cache == "캐시 적중만" and e.cache_hit)
                or (selected_cache == "신규 API 호출만" and not e.cache_hit)
            )
        ]

        log_rows = []
        for e in reversed(filtered_events):
            chunk_cfg = f"{e.chunk_size}/{e.chunk_overlap}" if e.chunk_size > 0 else "-"
            top_k_str = f"{e.top_k}개" if e.top_k > 0 else "-"
            lat_str = f"{e.latency_sec:.2f}s" if e.latency_sec > 0 else "-"
            tput_str = f"{e.throughput:.1f}t/s" if e.throughput > 0 else "-"
            score_str = f"{e.eval_score:.1f}" if e.eval_score is not None else "-"

            cost_str = f"₩{e.cost_krw:,.3f}" if not e.cache_hit else "₩0.000 (캐시)"
            saved_str = f"₩{e.saved_cost_krw:,.3f}" if e.saved_cost_krw > 0 else "-"

            prompt_preview = (e.prompt[:40] + "…") if len(e.prompt) > 40 else (e.prompt or "-")
            log_rows.append({
                "시각": e.timestamp,
                "Run ID": e.run_id or "-",
                "단계": e.stage,
                "컬렉션": e.collection_name or "-",
                "질문 내용": prompt_preview,
                "모델명": e.model_name,
                "청크": chunk_cfg,
                "Top-K": top_k_str,
                "토큰 (입/출)": f"{e.input_tokens:,} / {e.output_tokens:,}",
                "지연시간": lat_str,
                "속도": tput_str,
                "청구 비용": cost_str,
                "절감 비용": saved_str,
                "평가 점수": score_str,
                "_full_prompt": e.prompt,
            })
        df_logs = pd.DataFrame(log_rows)
        # 화면 표출용 (내부 컬럼 제외)
        st.dataframe(df_logs.drop(columns=["_full_prompt"]), width="stretch", hide_index=True)

        c_btn1, c_btn2, _ = st.columns([1, 1, 2])
        with c_btn1:
            # CSV에는 전체 질문 내용이 들어가도록 치환
            df_csv = df_logs.copy()
            df_csv["질문 내용"] = df_csv["_full_prompt"]
            df_csv = df_csv.drop(columns=["_full_prompt"])
            csv_data = df_csv.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 세부 로그 CSV 다운로드",
                data=csv_data,
                file_name="rag_app_detailed_billing_log.csv",
                mime="text/csv",
                width="stretch",
            )
        with c_btn2:
            if st.button("🔄 과금 카운터 초기화", width="stretch"):
                billing.reset_billing_events()
                st.rerun()
    else:
        st.caption("인덱싱을 실행하거나 질문을 입력하면 실시간으로 개별 API 과금 이벤트가 기록됩니다.")

st.divider()

# ---------------------------------------------------------------------------
# 4. Google Cloud 공식 단가 기준 안내
# ---------------------------------------------------------------------------
with st.expander("ℹ️ Google Cloud Model Garden & Partner Models 공식 단가 기준 안내"):
    st.markdown(
        """
        본 과금 카운터는 **Google Cloud 공식 가격표(2025/2026)**를 준수하여 계산됩니다:
        - [Google Model Garden Pricing 공식 문서](https://cloud.google.com/model-garden/pricing#cost-of-building-and-deploying-ai-models-in-agent-platform)
        - [Partner Models on Agent Platform Pricing 공식 문서](https://cloud.google.com/model-garden/partner-models/pricing#partner-models-on-agent-platform)

        ##### 1) 핵심 과금 규칙
        - **성공 요청만 과금**: `HTTP 200` 정상 응답을 반환한 요청에 대해서만 입출력 토큰 비용이 청구됩니다. (4xx, 5xx 에러는 과금 $0)
        - **Context Tiers**: 200k 토큰 이하 표준 요율 적용.
        - **캐싱 할인**: 기저장된 청크 임베딩 캐시 재사용 시 해당 임베딩 API 호출 비용은 $0 처리되며 절감 비용으로 누적 집계됩니다.

        ##### 2) 현재 등록된 모델별 단가표 (1M 토큰 기준)
        """
    )

    pricing_table_rows = []
    for m_id, p in billing.MODEL_GARDEN_PRICING.items():
        pricing_table_rows.append({
            "모델명": p["name"],
            "호출 식별자": m_id,
            "분류": p["type"],
            "입력 1M 토큰": f"${p['input_per_1m']:,.3f} (₩{p['input_per_1m'] * billing.EXCHANGE_RATE_KRW:,.1f})",
            "출력 1M 토큰": f"${p['output_per_1m']:,.3f} (₩{p['output_per_1m'] * billing.EXCHANGE_RATE_KRW:,.1f})",
        })
    st.dataframe(pd.DataFrame(pricing_table_rows), width="stretch", hide_index=True)
