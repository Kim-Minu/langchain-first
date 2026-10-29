"""RAG 앱 과금 대시보드 (Billing Dashboard).

Google Cloud Model Garden & Partner Models 공식 가격표 기준 실시간 과금 추적 화면.
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

st.set_page_config(page_title="과금 대시보드", page_icon="💰", layout="wide")

# 사이드바 공통 설정 렌더링
settings = render_sidebar(page="billing")

st.title("💰 RAG 앱 과금 대시보드 (Billing Dashboard)")
st.caption(
    "Google Cloud Model Garden & Partner Models(Anthropic, DeepSeek, Gemma, Gemini) 공식 단가 기준 "
    "인덱싱(OCR/임베딩), 런타임 질의응답, 벤치마크에서 발생한 API 비용을 실시간으로 추적·분석합니다."
)

st.divider()

# ---------------------------------------------------------------------------
# 1. 상단 요약 메트릭
# ---------------------------------------------------------------------------
summary = billing.get_billing_summary()

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "누적 총 비용 (KRW)",
    f"₩{summary['total_krw']:,.2f}",
    help="기준 환율 1 USD = 1,350 KRW 적용",
)
c2.metric(
    "누적 총 비용 (USD)",
    f"${summary['total_usd']:.5f}",
    help="Google Cloud 청구서 예상 실반영 금액",
)
c3.metric(
    "누적 토큰 사용량",
    f"{summary['total_in_tokens'] + summary['total_out_tokens']:,}",
    help=f"입력 {summary['total_in_tokens']:,} + 출력 {summary['total_out_tokens']:,} 토큰",
)
c4.metric(
    "과금 대상 API 호출 수",
    f"{summary['total_calls']}회",
    help="HTTP 200 성공 응답만 과금 (4xx/5xx 실패 요청은 공식 규정에 따라 $0)",
)

st.divider()

# ---------------------------------------------------------------------------
# 2. 단계별 & 모델별 비용 분석 차트
# ---------------------------------------------------------------------------
col_stage, col_model = st.columns(2)

with col_stage:
    st.subheader("📑 작업 단계별 누적 비용")
    if summary["by_stage"]:
        stage_rows = []
        for stage, data in summary["by_stage"].items():
            stage_rows.append({
                "작업 단계": stage,
                "호출 수": data["count"],
                "누적 토큰": data["tokens"],
                "비용 (원)": round(data["krw"], 2),
                "비용 ($)": round(data["usd"], 5),
            })
        df_stage = pd.DataFrame(stage_rows)
        st.dataframe(df_stage, width="stretch", hide_index=True)
        st.bar_chart(df_stage.set_index("작업 단계")[["비용 (원)"]])
    else:
        st.info("아직 발생한 작업 단계별 과금 내역이 없습니다.")

with col_model:
    st.subheader("🤖 모델별 누적 비용")
    if summary["by_model"]:
        model_rows = []
        for model, data in summary["by_model"].items():
            model_rows.append({
                "모델": model,
                "호출 수": data["count"],
                "누적 토큰": data["tokens"],
                "비용 (원)": round(data["krw"], 2),
                "비용 ($)": round(data["usd"], 5),
            })
        df_model = pd.DataFrame(model_rows)
        st.dataframe(df_model, width="stretch", hide_index=True)
        st.bar_chart(df_model.set_index("모델")[["비용 (원)"]])
    else:
        st.info("아직 발생한 모델별 과금 내역이 없습니다.")

st.divider()

# ---------------------------------------------------------------------------
# 3. 실시간 API 과금 이벤트 상세 로그
# ---------------------------------------------------------------------------
st.subheader("📋 세부 API 과금 이벤트 로그")

events: list[billing.BillingEvent] = summary["events"]
if events:
    log_rows = []
    for e in reversed(events):
        log_rows.append({
            "시각": e.timestamp,
            "단계": e.stage,
            "모델명": e.model_name,
            "입력 토큰": f"{e.input_tokens:,}",
            "출력 토큰": f"{e.output_tokens:,}",
            "비용 (원)": f"₩{e.cost_krw:.3f}",
            "비용 ($)": f"${e.cost_usd:.6f}",
        })
    df_logs = pd.DataFrame(log_rows)
    st.dataframe(df_logs, width="stretch", hide_index=True)

    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
    with c_btn1:
        csv_data = df_logs.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 CSV 다운로드",
            data=csv_data,
            file_name="rag_app_billing_log.csv",
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
        - **캐싱 할인**: 기저장된 청크 임베딩 캐시 재사용 시 해당 임베딩 API 호출 비용은 $0 처리됩니다.

        ##### 2) 현재 등록된 모델별 단가표 (1M 토큰 기준)
        """
    )

    pricing_table_rows = []
    for m_id, p in billing.MODEL_GARDEN_PRICING.items():
        pricing_table_rows.append({
            "모델명": p["name"],
            "호출 식별자": m_id,
            "분류": p["type"],
            "입력 1M 토큰": f"${p['input_per_1m']:.3f} (₩{p['input_per_1m'] * billing.EXCHANGE_RATE_KRW:,.1f})",
            "출력 1M 토큰": f"${p['output_per_1m']:.3f} (₩{p['output_per_1m'] * billing.EXCHANGE_RATE_KRW:,.1f})",
        })
    st.dataframe(pd.DataFrame(pricing_table_rows), width="stretch", hide_index=True)

