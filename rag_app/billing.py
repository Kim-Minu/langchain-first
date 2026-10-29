"""Google Cloud Model Garden & Partner Models 공식 가격 기준 과금 추적(Billing Tracker) 모듈.

참조:
- https://cloud.google.com/model-garden/pricing#cost-of-building-and-deploying-ai-models-in-agent-platform
- https://cloud.google.com/model-garden/partner-models/pricing#partner-models-on-agent-platform
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import streamlit as st

from rag_app.config import DATABASE_URL

logger = logging.getLogger(__name__)

EXCHANGE_RATE_KRW = 1350.0  # 1 USD = 1,350 KRW (참고 환율)

# ---------------------------------------------------------------------------
# Google Model Garden & Partner Models 공식 단가표 (1M = 1,000,000 토큰 기준 USD)
# 참조:
# - https://cloud.google.com/model-garden/pricing#cost-of-building-and-deploying-ai-models-in-agent-platform
# - https://cloud.google.com/model-garden/partner-models/pricing#partner-models-on-agent-platform
# (표준 context <= 200k 토큰, HTTP 200 성공 응답 기준)
# ---------------------------------------------------------------------------
MODEL_GARDEN_PRICING = {
    # 1. Partner Models (Model as a Service / Agent Platform)
    # Anthropic Claude 3.5 Haiku / Haiku 계열 (Vertex AI MaaS 공식 정가)
    # <= 200k 토큰 기준: Input $1.00 / Output $5.00 per 1M tokens
    "claude-haiku-5-5": {
        "name": "Claude Haiku 5.5",
        "input_per_1m": 1.00,
        "output_per_1m": 5.00,
        "type": "partner_maas",
    },
    # DeepSeek V4.1 Flash MaaS (캐시 미적용 기본 호출 기준)
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
    # 2. Google First-Party Models (Gemini / Vertex AI Agent Platform)
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
        "input_per_1m": 0.025,  # 1M 토큰당 $0.025 (약 1,000자당 $0.000025)
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
    stage: str  # "문서 이미지 읽기 (OCR)", "문서 청크 임베딩", "질문 임베딩", "RAG LLM 답변", "벤치마크 테스트"
    model_id: str
    model_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_krw: float
    collection_name: str = "-"
    run_id: str = ""
    prompt: str = ""
    cache_hit: bool = False
    cache_hit_count: int = 0
    cached_tokens: int = 0
    saved_cost_krw: float = 0.0
    latency_sec: float = 0.0
    ttft_sec: float = 0.0
    throughput: float = 0.0
    chunk_size: int = 0
    chunk_overlap: int = 0
    top_k: int = 0
    eval_score: float | None = None
    faithfulness: float | None = None


def get_model_pricing(model_id: str) -> dict[str, Any]:
    """해당 모델의 단가 정보를 반환합니다."""
    return MODEL_GARDEN_PRICING.get(
        model_id,
        {"name": model_id, "input_per_1m": 0.15, "output_per_1m": 0.60, "type": "unknown"},
    )


def init_billing_table() -> None:
    """PostgreSQL 데이터베이스에 rag_billing_events 테이블이 없으면 생성합니다."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS rag_billing_events (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        timestamp_str VARCHAR(32) NOT NULL,
        stage VARCHAR(64) NOT NULL,
        model_id VARCHAR(128) NOT NULL,
        model_name VARCHAR(128) NOT NULL,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        cost_krw DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        collection_name VARCHAR(128) NOT NULL DEFAULT '-',
        run_id VARCHAR(64) NOT NULL DEFAULT '',
        cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
        cache_hit_count INTEGER NOT NULL DEFAULT 0,
        cached_tokens INTEGER NOT NULL DEFAULT 0,
        saved_cost_krw DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        latency_sec DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        ttft_sec DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        throughput DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        chunk_size INTEGER NOT NULL DEFAULT 0,
        chunk_overlap INTEGER NOT NULL DEFAULT 0,
        top_k INTEGER NOT NULL DEFAULT 0,
        eval_score DOUBLE PRECISION DEFAULT NULL,
        faithfulness DOUBLE PRECISION DEFAULT NULL
    );
    -- 기존 테이블이 이미 존재하는 경우 컬럼 자동 추가 (안전한 마이그레이션)
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS collection_name VARCHAR(128) NOT NULL DEFAULT '-';
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS run_id VARCHAR(64) NOT NULL DEFAULT '';
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS prompt TEXT DEFAULT '';
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS cache_hit BOOLEAN NOT NULL DEFAULT FALSE;
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS cache_hit_count INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS cached_tokens INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS saved_cost_krw DOUBLE PRECISION NOT NULL DEFAULT 0.0;
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS latency_sec DOUBLE PRECISION NOT NULL DEFAULT 0.0;
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS ttft_sec DOUBLE PRECISION NOT NULL DEFAULT 0.0;
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS throughput DOUBLE PRECISION NOT NULL DEFAULT 0.0;
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS chunk_size INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS chunk_overlap INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS top_k INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS eval_score DOUBLE PRECISION DEFAULT NULL;
    ALTER TABLE rag_billing_events ADD COLUMN IF NOT EXISTS faithfulness DOUBLE PRECISION DEFAULT NULL;

    CREATE INDEX IF NOT EXISTS idx_rag_billing_events_id ON rag_billing_events(id);
    CREATE INDEX IF NOT EXISTS idx_rag_billing_events_coll ON rag_billing_events(collection_name);
    CREATE INDEX IF NOT EXISTS idx_rag_billing_events_run ON rag_billing_events(run_id);
    """
    try:
        import psycopg
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
            conn.commit()
    except Exception as e:
        logger.warning("과금 테이블(rag_billing_events) 초기화 실패: %s", e)


def load_billing_events_from_db() -> list[BillingEvent]:
    """DB로부터 영구 저장된 과금 이벤트 목록을 읽어옵니다."""
    events: list[BillingEvent] = []
    try:
        import psycopg
        init_billing_table()
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT timestamp_str, stage, model_id, model_name,
                           input_tokens, output_tokens, cost_usd, cost_krw,
                           COALESCE(collection_name, '-'),
                           COALESCE(run_id, ''),
                           COALESCE(prompt, ''),
                           COALESCE(cache_hit, FALSE),
                           COALESCE(cache_hit_count, 0),
                           COALESCE(cached_tokens, 0),
                           COALESCE(saved_cost_krw, 0.0),
                           COALESCE(latency_sec, 0.0),
                           COALESCE(ttft_sec, 0.0),
                           COALESCE(throughput, 0.0),
                           COALESCE(chunk_size, 0),
                           COALESCE(chunk_overlap, 0),
                           COALESCE(top_k, 0),
                           eval_score,
                           faithfulness
                    FROM rag_billing_events
                    ORDER BY id ASC;
                """)
                rows = cur.fetchall()
                for r in rows:
                    events.append(BillingEvent(
                        timestamp=r[0],
                        stage=r[1],
                        model_id=r[2],
                        model_name=r[3],
                        input_tokens=int(r[4]),
                        output_tokens=int(r[5]),
                        cost_usd=float(r[6]),
                        cost_krw=float(r[7]),
                        collection_name=str(r[8]),
                        run_id=str(r[9]),
                        prompt=str(r[10]),
                        cache_hit=bool(r[11]),
                        cache_hit_count=int(r[12]),
                        cached_tokens=int(r[13]),
                        saved_cost_krw=float(r[14]),
                        latency_sec=float(r[15]),
                        ttft_sec=float(r[16]),
                        throughput=float(r[17]),
                        chunk_size=int(r[18]),
                        chunk_overlap=int(r[19]),
                        top_k=int(r[20]),
                        eval_score=float(r[21]) if r[21] is not None else None,
                        faithfulness=float(r[22]) if r[22] is not None else None,
                    ))
    except Exception as e:
        logger.warning("DB로부터 과금 내역 불러오기 실패: %s", e)
    return events


def save_billing_event_to_db(event: BillingEvent) -> None:
    """과금 이벤트를 DB에 영구 저장합니다."""
    try:
        import psycopg
        init_billing_table()
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO rag_billing_events (
                        timestamp_str, stage, model_id, model_name,
                        input_tokens, output_tokens, cost_usd, cost_krw,
                        collection_name, run_id, prompt, cache_hit, cache_hit_count,
                        cached_tokens, saved_cost_krw, latency_sec, ttft_sec,
                        throughput, chunk_size, chunk_overlap, top_k,
                        eval_score, faithfulness
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s
                    );
                """, (
                    event.timestamp,
                    event.stage,
                    event.model_id,
                    event.model_name,
                    event.input_tokens,
                    event.output_tokens,
                    event.cost_usd,
                    event.cost_krw,
                    event.collection_name,
                    event.run_id,
                    event.prompt,
                    event.cache_hit,
                    event.cache_hit_count,
                    event.cached_tokens,
                    event.saved_cost_krw,
                    event.latency_sec,
                    event.ttft_sec,
                    event.throughput,
                    event.chunk_size,
                    event.chunk_overlap,
                    event.top_k,
                    event.eval_score,
                    event.faithfulness,
                ))
            conn.commit()
    except Exception as e:
        logger.warning("과금 이벤트 DB 저장 실패: %s", e)


def delete_all_billing_events_db() -> None:
    """DB에 저장된 모든 과금 내역을 삭제합니다."""
    try:
        import psycopg
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE rag_billing_events RESTART IDENTITY;")
            conn.commit()
    except Exception as e:
        logger.warning("DB 과금 내역 초기화 실패: %s", e)


def record_billing_event(
    stage: str,
    model_id: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    collection_name: str | None = None,
    run_id: str = "",
    prompt: str = "",
    cache_hit: bool = False,
    cache_hit_count: int = 0,
    cached_tokens: int = 0,
    saved_cost_krw: float = 0.0,
    latency_sec: float = 0.0,
    ttft_sec: float = 0.0,
    throughput: float = 0.0,
    chunk_size: int = 0,
    chunk_overlap: int = 0,
    top_k: int = 0,
    eval_score: float | None = None,
    faithfulness: float | None = None,
) -> BillingEvent:
    """API 호출(HTTP 200 성공 시)에 대한 과금 이벤트를 DB와 세션에 영구 기록합니다."""
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
        collection_name=collection_name or "-",
        run_id=run_id,
        prompt=prompt,
        cache_hit=cache_hit,
        cache_hit_count=cache_hit_count,
        cached_tokens=cached_tokens,
        saved_cost_krw=saved_cost_krw,
        latency_sec=latency_sec,
        ttft_sec=ttft_sec,
        throughput=throughput,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        eval_score=eval_score,
        faithfulness=faithfulness,
    )

    # 1. PostgreSQL DB 영구 저장
    save_billing_event_to_db(event)

    # 2. Streamlit 세션 상태 동기화
    try:
        if "billing_events" not in st.session_state or st.session_state["billing_events"] is None:
            st.session_state["billing_events"] = load_billing_events_from_db()
        else:
            st.session_state["billing_events"].append(event)
    except Exception:
        pass

    return event


def get_billing_summary() -> dict[str, Any]:
    """누적 과금 합계 및 통계를 반환합니다 (세션 비어있을 시 DB에서 자동 복원)."""
    try:
        if "billing_events" not in st.session_state or st.session_state["billing_events"] is None:
            st.session_state["billing_events"] = load_billing_events_from_db()
        events: list[BillingEvent] = st.session_state.get("billing_events", [])
    except Exception:
        # CLI 환경 등 Streamlit 컨텍스트 밖에서는 DB에서 직접 로드
        events = load_billing_events_from_db()

    total_usd = sum(e.cost_usd for e in events)
    total_krw = sum(e.cost_krw for e in events)
    total_in = sum(e.input_tokens for e in events)
    total_out = sum(e.output_tokens for e in events)
    total_saved_krw = sum(e.saved_cost_krw for e in events)
    total_cached_tokens = sum(e.cached_tokens for e in events)
    total_cache_hits = sum(1 for e in events if e.cache_hit or e.cache_hit_count > 0)

    by_stage: dict[str, dict[str, float]] = {}
    for e in events:
        s = by_stage.setdefault(e.stage, {"usd": 0.0, "krw": 0.0, "count": 0, "tokens": 0, "saved_krw": 0.0})
        s["usd"] += e.cost_usd
        s["krw"] += e.cost_krw
        s["count"] += 1
        s["tokens"] += (e.input_tokens + e.output_tokens)
        s["saved_krw"] += e.saved_cost_krw

    by_model: dict[str, dict[str, float]] = {}
    for e in events:
        m = by_model.setdefault(e.model_name, {"usd": 0.0, "krw": 0.0, "count": 0, "tokens": 0, "avg_latency": 0.0, "latencies": []})
        m["usd"] += e.cost_usd
        m["krw"] += e.cost_krw
        m["count"] += 1
        m["tokens"] += (e.input_tokens + e.output_tokens)
        if e.latency_sec > 0:
            m["latencies"].append(e.latency_sec)

    for m in by_model.values():
        m["avg_latency"] = sum(m["latencies"]) / len(m["latencies"]) if m["latencies"] else 0.0

    by_collection: dict[str, dict[str, Any]] = {}
    for e in events:
        coll = e.collection_name or "-"
        c = by_collection.setdefault(coll, {"usd": 0.0, "krw": 0.0, "count": 0, "tokens": 0, "saved_krw": 0.0})
        c["usd"] += e.cost_usd
        c["krw"] += e.cost_krw
        c["count"] += 1
        c["tokens"] += (e.input_tokens + e.output_tokens)
        c["saved_krw"] += e.saved_cost_krw

    # 최적 조합 분석용 집계 (모델 × 청크크기 × top_k)
    by_combination: dict[tuple, dict[str, Any]] = {}
    for e in events:
        # 청크 크기나 top_k가 기록된 이벤트 위주로 집계
        if e.chunk_size > 0 or e.top_k > 0 or "RAG" in e.stage or "벤치마크" in e.stage:
            key = (e.model_name, e.chunk_size or 500, e.top_k or 4)
            c = by_combination.setdefault(key, {
                "model_name": e.model_name,
                "chunk_size": e.chunk_size or 500,
                "top_k": e.top_k or 4,
                "count": 0,
                "total_krw": 0.0,
                "total_tokens": 0,
                "latencies": [],
                "ttfts": [],
                "throughputs": [],
                "scores": [],
                "faithfulness": [],
            })
            c["count"] += 1
            c["total_krw"] += e.cost_krw
            c["total_tokens"] += (e.input_tokens + e.output_tokens)
            if e.latency_sec > 0:
                c["latencies"].append(e.latency_sec)
            if e.ttft_sec > 0:
                c["ttfts"].append(e.ttft_sec)
            if e.throughput > 0:
                c["throughputs"].append(e.throughput)
            if e.eval_score is not None:
                c["scores"].append(e.eval_score)
            if e.faithfulness is not None:
                c["faithfulness"].append(e.faithfulness)

    comb_summary: list[dict[str, Any]] = []
    for (m_name, c_size, tk), data in by_combination.items():
        n = data["count"]
        avg_cost = data["total_krw"] / n if n > 0 else 0.0
        avg_tokens = data["total_tokens"] / n if n > 0 else 0.0
        avg_lat = sum(data["latencies"]) / len(data["latencies"]) if data["latencies"] else 0.0
        avg_ttft = sum(data["ttfts"]) / len(data["ttfts"]) if data["ttfts"] else 0.0
        avg_tp = sum(data["throughputs"]) / len(data["throughputs"]) if data["throughputs"] else 0.0
        avg_score = (sum(data["scores"]) / len(data["scores"])) if data["scores"] else None
        avg_faith = (sum(data["faithfulness"]) / len(data["faithfulness"])) if data["faithfulness"] else None

        comb_summary.append({
            "model_name": m_name,
            "chunk_size": c_size,
            "top_k": tk,
            "count": n,
            "avg_cost_krw": avg_cost,
            "avg_tokens": avg_tokens,
            "avg_latency_sec": avg_lat,
            "avg_ttft_sec": avg_ttft,
            "avg_throughput": avg_tp,
            "avg_eval_score": avg_score,
            "avg_faithfulness": avg_faith,
        })

    return {
        "total_usd": total_usd,
        "total_krw": total_krw,
        "total_in_tokens": total_in,
        "total_out_tokens": total_out,
        "total_saved_krw": total_saved_krw,
        "total_cached_tokens": total_cached_tokens,
        "total_cache_hits": total_cache_hits,
        "total_calls": len(events),
        "events": events,
        "by_stage": by_stage,
        "by_model": by_model,
        "by_collection": by_collection,
        "by_combination": comb_summary,
    }


def reset_billing_events() -> None:
    """과금 카운터를 초기화합니다 (DB 레코드 삭제 및 세션 리셋)."""
    delete_all_billing_events_db()
    try:
        if "billing_events" in st.session_state:
            st.session_state["billing_events"] = []
    except Exception:
        pass


def render_billing_counter_widget(container: Any = None) -> None:
    """Streamlit UI에 과금 카운터 위젯을 렌더링합니다."""
    summary = get_billing_summary()
    c = container if container is not None else st.container()

    with c:
        st.markdown(
            f"""
            <div style="background-color: #1e222a; border: 1px solid #30363d; border-radius: 8px; padding: 12px 14px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 13px; font-weight: 600; color: #58a6ff;">💳 실시간 과금 카운터 <span style="font-size: 10px; color: #7ee787; font-weight: normal;">(DB 보존)</span></span>
                    <span style="font-size: 11px; color: #8b949e;">호출 {summary['total_calls']:,}회</span>
                </div>
                <div style="margin-top: 6px; display: flex; align-items: baseline; gap: 8px;">
                    <span style="font-size: 20px; font-weight: 700; color: #f0f6fc;">₩{summary['total_krw']:,.2f}</span>
                    <span style="font-size: 12px; color: #8b949e;">(${summary['total_usd']:,.5f})</span>
                </div>
                <div style="font-size: 11px; color: #8b949e; margin-top: 4px;">
                    누적 토큰: {summary['total_in_tokens'] + summary['total_out_tokens']:,} (입력 {summary['total_in_tokens']:,} / 출력 {summary['total_out_tokens']:,})
                </div>
                {"<div style='font-size: 11px; color: #7ee787; margin-top: 3px;'>⚡ 캐시 절감액: ₩" + f"{summary['total_saved_krw']:,.2f} ({summary['total_cache_hits']:,}회)</div>" if summary['total_saved_krw'] > 0 else ""}
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
                    st.write(f"- **{stage}**: ₩{data['krw']:,.2f} (${data['usd']:,.5f}) · {data['count']:,}회")

                st.caption("모델별 누적 비용")
                for model, data in summary["by_model"].items():
                    st.write(f"- **{model}**: ₩{data['krw']:,.2f} (${data['usd']:,.5f}) · {data['tokens']:,} 토큰")

                if summary.get("by_collection"):
                    st.caption("컬렉션별 누적 비용")
                    for coll, data in summary["by_collection"].items():
                        st.write(f"- **{coll}**: ₩{data['krw']:,.2f} (${data['usd']:,.5f}) · {data['count']:,}회")

                if st.button("🔄 과금 카운터 초기화", key="btn_reset_billing", width="stretch"):
                    reset_billing_events()
                    st.rerun()
