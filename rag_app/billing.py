"""Google Cloud Model Garden & Partner Models 공식 가격 기준 과금 추적(Billing Tracker) 모듈.

참조:
- https://cloud.google.com/model-garden/pricing#cost-of-building-and-deploying-ai-models-in-agent-platform
- https://cloud.google.com/model-garden/partner-models/pricing#partner-models-on-agent-platform
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import streamlit as st

EXCHANGE_RATE_KRW = 1350.0  # 1 USD = 1,350 KRW (참고 환율)

# ---------------------------------------------------------------------------
# Google Model Garden & Partner Models 공식 단가표 (1M = 1,000,000 토큰 기준 USD)
# ---------------------------------------------------------------------------
MODEL_GARDEN_PRICING = {
    # 1. Partner Models (Model as a Service / Agent Platform)
    # DeepSeek V4.1 Flash MaaS
    "deepseek-ai/deepseek-v4.1-flash-maas": {
        "name": "DeepSeek V4.1 Flash (MaaS)",
        "input_per_1m": 0.14,
        "output_per_1m": 0.28,
        "type": "partner_maas",
    },
    # Gemma 4 26B A4B IT MaaS
    "google/gemma-4-26b-a4b-it-maas": {
        "name": "Gemma 4 26B (MaaS)",
        "input_per_1m": 0.20,
        "output_per_1m": 0.40,
        "type": "partner_maas",
    },
    # Anthropic Claude Haiku 5.5 / 3.5 Haiku (Vertex MaaS)
    "claude-haiku-5-5": {
        "name": "Claude Haiku 5.5",
        "input_per_1m": 0.80,
        "output_per_1m": 4.00,
        "type": "partner_maas",
    },
    # 2. Google First-Party Models (Gemini / Vertex AI)
    # Gemini 3.5 Flash Lite / 2.5 Flash Lite (<= 200k tokens)
    "gemini-3.5-flash-lite": {
        "name": "Gemini 3.5 Flash Lite",
        "input_per_1m": 0.075,
        "output_per_1m": 0.30,
        "type": "first_party",
    },
    # Gemini 3.8 Flash / 2.5 Flash (<= 200k tokens)
    "gemini-3.8-flash": {
        "name": "Gemini 3.8 Flash",
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
        "type": "first_party",
    },
    # 3. 임베딩 모델 (Vertex AI Embeddings)
    "gemini-embedding-2": {
        "name": "Gemini Embedding 2",
        "input_per_1m": 0.025,  # 1M 토큰당 $0.025 (약 1,000자당 $0.00002)
        "output_per_1m": 0.0,
        "type": "embedding",
    },
    "text-embedding-005": {
        "name": "Text Embedding 005",
        "input_per_1m": 0.025,
        "output_per_1m": 0.0,
        "type": "embedding",
    },
    "text-multilingual-embedding-002": {
        "name": "Text Multilingual 002",
        "input_per_1m": 0.025,
        "output_per_1m": 0.0,
        "type": "embedding",
    },
    # 4. 로컬 모델 (무료)
    "gemma4:e2b": {
        "name": "Gemma 4 (로컬 Ollama)",
        "input_per_1m": 0.0,
        "output_per_1m": 0.0,
        "type": "local",
    },
}


@dataclass
class BillingEvent:
    timestamp: str
    stage: str  # "문서 이미지 읽기", "문서 임베딩", "질문 임베딩", "LLM 답변"
    model_id: str
    model_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_krw: float


def get_model_pricing(model_id: str) -> dict[str, Any]:
    """해당 모델의 단가 정보를 반환합니다."""
    return MODEL_GARDEN_PRICING.get(
        model_id,
        {"name": model_id, "input_per_1m": 0.15, "output_per_1m": 0.60, "type": "unknown"},
    )


def record_billing_event(
    stage: str,
    model_id: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> BillingEvent:
    """API 호출(HTTP 200 성공 시)에 대한 과금 이벤트를 세션에 기록합니다."""
    pricing = get_model_pricing(model_id)
    cost_in = (input_tokens / 1_000_000.0) * pricing["input_per_1m"]
    cost_out = (output_tokens / 1_000_000.0) * pricing["output_per_1m"]
    cost_usd = cost_in + cost_out
    cost_krw = cost_usd * EXCHANGE_RATE_KRW

    event = BillingEvent(
        timestamp=datetime.now().strftime("%H:%M:%S"),
        stage=stage,
        model_id=model_id,
        model_name=pricing.get("name", model_id),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        cost_krw=cost_krw,
    )

    if "billing_events" not in st.session_state:
        st.session_state["billing_events"] = []
    st.session_state["billing_events"].append(event)

    return event


def get_billing_summary() -> dict[str, Any]:
    """현재 세션의 누적 과금 합계 및 통계를 반환합니다."""
    events: list[BillingEvent] = st.session_state.get("billing_events", [])
    total_usd = sum(e.cost_usd for e in events)
    total_krw = sum(e.cost_krw for e in events)
    total_in = sum(e.input_tokens for e in events)
    total_out = sum(e.output_tokens for e in events)

    by_stage: dict[str, dict[str, float]] = {}
    for e in events:
        s = by_stage.setdefault(e.stage, {"usd": 0.0, "krw": 0.0, "count": 0, "tokens": 0})
        s["usd"] += e.cost_usd
        s["krw"] += e.cost_krw
        s["count"] += 1
        s["tokens"] += (e.input_tokens + e.output_tokens)

    by_model: dict[str, dict[str, float]] = {}
    for e in events:
        m = by_model.setdefault(e.model_name, {"usd": 0.0, "krw": 0.0, "count": 0, "tokens": 0})
        m["usd"] += e.cost_usd
        m["krw"] += e.cost_krw
        m["count"] += 1
        m["tokens"] += (e.input_tokens + e.output_tokens)

    return {
        "total_usd": total_usd,
        "total_krw": total_krw,
        "total_in_tokens": total_in,
        "total_out_tokens": total_out,
        "total_calls": len(events),
        "events": events,
        "by_stage": by_stage,
        "by_model": by_model,
    }


def reset_billing_events() -> None:
    """과금 카운터를 초기화합니다."""
    st.session_state["billing_events"] = []


def render_billing_counter_widget(container: Any = None) -> None:
    """Streamlit UI에 과금 카운터 위젯을 렌더링합니다."""
    summary = get_billing_summary()
    c = container if container is not None else st.container()

    with c:
        st.markdown(
            f"""
            <div style="background-color: #1e222a; border: 1px solid #30363d; border-radius: 8px; padding: 12px 14px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 13px; font-weight: 600; color: #58a6ff;">💳 실시간 과금 카운터</span>
                    <span style="font-size: 11px; color: #8b949e;">호출 {summary['total_calls']}회</span>
                </div>
                <div style="margin-top: 6px; display: flex; align-items: baseline; gap: 8px;">
                    <span style="font-size: 20px; font-weight: 700; color: #f0f6fc;">₩{summary['total_krw']:,.2f}</span>
                    <span style="font-size: 12px; color: #8b949e;">(${summary['total_usd']:.5f})</span>
                </div>
                <div style="font-size: 11px; color: #8b949e; margin-top: 4px;">
                    누적 토큰: {summary['total_in_tokens'] + summary['total_out_tokens']:,} (입력 {summary['total_in_tokens']:,} / 출력 {summary['total_out_tokens']:,})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📊 과금 상세 내역 및 초기화", expanded=False):
            if summary["total_calls"] == 0:
                st.caption("아직 발생한 API 호출 과금이 없습니다.")
            else:
                st.caption("단계별 누적 비용")
                for stage, data in summary["by_stage"].items():
                    st.write(f"- **{stage}**: ₩{data['krw']:.2f} (${data['usd']:.5f}) · {data['count']}회")

                st.caption("모델별 누적 비용")
                for model, data in summary["by_model"].items():
                    st.write(f"- **{model}**: ₩{data['krw']:.2f} (${data['usd']:.5f}) · {data['tokens']:,} 토큰")

                if st.button("🔄 과금 카운터 초기화", key="btn_reset_billing", width="stretch"):
                    reset_billing_events()
                    st.rerun()
