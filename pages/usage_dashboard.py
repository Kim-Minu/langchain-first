"""
[토큰 사용량 대시보드] 일별 · 주별 · 월별 × 모델별 토큰 · 응답 수 · 예상 비용 통계

- 데이터: chat_messages 테이블의 AI 답변 메타데이터 (답변마다 모델 · 입력/출력/추론 토큰 · 비용 기록)
- 날짜는 한국 시간 기준, 주는 월요일 시작
- 채팅 화면 사이드바의 "사용량 대시보드" 링크로 열고, 상단 링크로 돌아갑니다.
"""

from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from chat_app.config import KRW_PER_USD, MODEL_OPTIONS
from chat_app.db import get_checkpointer, get_usage_by_period, get_usage_date_bounds

st.set_page_config(page_title="토큰 사용량 대시보드", page_icon="📊", layout="wide")

PERIOD_LABELS = {"day": "일별", "week": "주별", "month": "월별"}
# 기간 단위별 기본 조회 범위 (일)
DEFAULT_RANGE_DAYS = {"day": 30, "week": 7 * 12, "month": 365}
PANDAS_FREQ = {"day": "D", "week": "W-MON", "month": "MS"}

# 모델 색: dataviz 기본 팔레트의 다크 모드 범주형 1~3번 (세 색은 색각 이상 기준 전체 쌍 검증 통과)
# 색은 MODEL_OPTIONS 순서로 모델에 고정 → 필터로 모델을 빼도 남은 모델의 색이 바뀌지 않음
SERIES_COLORS = ["#3987e5", "#d95926", "#199e70"]
OTHER_COLOR = "#8a8a85"
OTHER_LABEL = "기타"
SURFACE = "#131314"
GRID = "#2a2b2d"
TEXT_MUTED = "#9aa0a6"

METRICS = {
    "total_tokens": "총 토큰",
    "in_tokens": "입력 토큰",
    "out_tokens": "출력 토큰",
    "responses": "응답 수",
    "cost_usd": "예상 비용 (USD)",
}


def model_label(model_key: str) -> str:
    """모델 키 → 화면 이름. 지금 선택지에 없는 예전 모델은 '기타'로 묶음 (9번째 색을 만들지 않음)."""
    return MODEL_OPTIONS.get(model_key, OTHER_LABEL)


def model_color_scale() -> alt.Scale:
    labels = list(MODEL_OPTIONS.values())
    colors = (SERIES_COLORS + [OTHER_COLOR] * len(labels))[: len(labels)]
    return alt.Scale(domain=labels + [OTHER_LABEL], range=colors + [OTHER_COLOR])


def period_label(day: date, period: str) -> str:
    if period == "month":
        return f"{day:%Y-%m}"
    if period == "week":
        return f"{day:%m/%d} 주"
    return f"{day:%m/%d}"


def align_to_period(day: date, period: str) -> date:
    if period == "month":
        return day.replace(day=1)
    if period == "week":
        return day - timedelta(days=day.weekday())
    return day


def load_frame(period: str, start: date, end: date) -> pd.DataFrame:
    rows = get_usage_by_period(period, start, end)
    df = pd.DataFrame(rows, columns=["period", "model", "responses", "in_tokens", "out_tokens",
                                     "reasoning_tokens", "total_tokens", "cost_usd"])
    if df.empty:
        return df
    # PostgreSQL SUM(bigint)은 numeric(Decimal)으로 오므로 숫자형으로 바꿔야 pandas 합산에 포함됨
    value_columns = ["responses", "in_tokens", "out_tokens", "reasoning_tokens", "total_tokens", "cost_usd"]
    df[value_columns] = df[value_columns].apply(pd.to_numeric)
    df["model"] = df["model"].map(model_label)
    # 같은 '기타'로 묶인 예전 모델들을 합산
    return df.groupby(["period", "model"], as_index=False).sum(numeric_only=True)


def fill_missing_periods(df: pd.DataFrame, period: str, start: date, end: date, models: list[str]) -> pd.DataFrame:
    """사용 기록이 없는 기간도 0으로 채워 막대 사이 빈 기간이 그대로 보이게 합니다."""
    periods = pd.date_range(align_to_period(start, period), end, freq=PANDAS_FREQ[period]).date
    grid = pd.MultiIndex.from_product([periods, models], names=["period", "model"]).to_frame(index=False)
    filled = grid.merge(df, on=["period", "model"], how="left").fillna(0)
    filled["label"] = [period_label(p, period) for p in filled["period"]]
    return filled


def period_chart(df: pd.DataFrame, metric: str, models: list[str]) -> alt.Chart:
    order = list(MODEL_OPTIONS.values()) + [OTHER_LABEL]
    df = df.assign(model_order=df["model"].map(order.index))
    value_format = "$,.4f" if metric == "cost_usd" else ",.0f"
    return (
        alt.Chart(df)
        .mark_bar(stroke=SURFACE, strokeWidth=2, width={"band": 0.7})  # 쌓인 막대 조각 사이 2px 간격
        .encode(
            x=alt.X("label:O", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelOverlap=True)),
            y=alt.Y(f"{metric}:Q", stack="zero", title=None, axis=alt.Axis(format="~s" if metric != "cost_usd" else "$~g")),
            color=alt.Color("model:N", scale=model_color_scale(), title="모델",
                            legend=alt.Legend(orient="top", values=models)),
            order=alt.Order("model_order:Q"),
            tooltip=[
                alt.Tooltip("label:N", title="기간"),
                alt.Tooltip("model:N", title="모델"),
                alt.Tooltip(f"{metric}:Q", title=METRICS[metric], format=value_format),
                alt.Tooltip("responses:Q", title="응답 수", format=",.0f"),
                alt.Tooltip("in_tokens:Q", title="입력 토큰", format=",.0f"),
                alt.Tooltip("out_tokens:Q", title="출력 토큰", format=",.0f"),
            ],
        )
        .properties(height=340, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID, labelColor=TEXT_MUTED, titleColor=TEXT_MUTED)
        .configure_legend(labelColor="#e3e3e3", titleColor=TEXT_MUTED)
    )


def model_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("model", as_index=False).sum(numeric_only=True)
    summary["avg_tokens"] = summary["total_tokens"] / summary["responses"].where(summary["responses"] > 0)
    summary["share"] = summary["total_tokens"] / max(summary["total_tokens"].sum(), 1)
    order = list(MODEL_OPTIONS.values()) + [OTHER_LABEL]
    return summary.sort_values("model", key=lambda s: s.map(order.index)).reset_index(drop=True)


# ------------------------------------------------------------------------------
# 화면
# ------------------------------------------------------------------------------
try:
    get_checkpointer()  # 테이블이 없으면 생성 (대시보드를 먼저 연 경우)
except Exception as e:
    st.error(f"PostgreSQL 연결에 실패했습니다: {e}\n\n`docker compose up -d`로 DB를 실행하거나 `DATABASE_URL`을 확인하세요.")
    st.stop()

st.page_link("web_chat.py", label="채팅으로 돌아가기", icon=":material/arrow_back:")
st.title("토큰 사용량")
st.caption("AI 답변마다 기록된 모델 · 토큰 · 예상 비용을 기간별로 합산합니다. 한국 시간 기준 · 주는 월요일 시작")

bounds = get_usage_date_bounds()
if bounds["first_day"] is None:
    st.info("아직 사용 기록이 없습니다. 채팅에서 질문하면 여기에 통계가 쌓입니다.")
    st.stop()

# 필터: 한 줄에 기간 단위 · 날짜 범위 · 모델 · 지표
col_period, col_range, col_models, col_metric = st.columns([1.2, 1.6, 2.2, 1.2])
with col_period:
    period = st.segmented_control(
        "기간 단위", options=list(PERIOD_LABELS), format_func=PERIOD_LABELS.get, default="day", key="period"
    ) or "day"
today = date.today()
with col_range:
    # 기록이 없는 앞쪽 기간도 0으로 보여 주도록 첫 기록일로 자르지 않음
    default_start = today - timedelta(days=DEFAULT_RANGE_DAYS[period] - 1)
    picked = st.date_input("날짜 범위", value=(default_start, today), max_value=today, key=f"range_{period}")
start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (default_start, today))

frame = load_frame(period, start, end)
all_models = [m for m in list(MODEL_OPTIONS.values()) + [OTHER_LABEL] if m in set(frame.get("model", []))]
with col_models:
    models = st.multiselect("모델", options=all_models, default=all_models, key="models")
with col_metric:
    metric = st.selectbox("지표", options=list(METRICS), format_func=METRICS.get, key="metric")

frame = frame[frame["model"].isin(models)] if not frame.empty else frame
if frame.empty or not models:
    st.info("선택한 기간 · 모델에 사용 기록이 없습니다.")
    st.stop()

# KPI
total_tokens = int(frame["total_tokens"].sum())
responses = int(frame["responses"].sum())
cost_usd = float(frame["cost_usd"].sum())
k1, k2, k3, k4 = st.columns(4)
k1.metric("총 토큰", f"{total_tokens:,}", help=f"입력 {int(frame['in_tokens'].sum()):,} / 출력 {int(frame['out_tokens'].sum()):,}")
k2.metric("응답 수", f"{responses:,}")
k3.metric("예상 비용", f"${cost_usd:,.4f}", help=f"약 {cost_usd * KRW_PER_USD:,.1f}원 · Gemini Flash 요율 기준, 로컬 모델은 0")
k4.metric("응답당 평균 토큰", f"{total_tokens / responses:,.0f}" if responses else "-")

# 기간 × 모델 쌓인 막대
st.subheader(f"{PERIOD_LABELS[period]} {METRICS[metric]}")
filled = fill_missing_periods(frame, period, start, end, models)
st.altair_chart(period_chart(filled, metric, models), width="stretch")

# 모델별 요약
st.subheader("모델별 합계")
summary = model_summary(frame)
st.dataframe(
    summary[["model", "responses", "in_tokens", "out_tokens", "reasoning_tokens", "total_tokens",
             "avg_tokens", "share", "cost_usd"]],
    hide_index=True,
    width="stretch",
    column_config={
        "model": "모델",
        "responses": st.column_config.NumberColumn("응답 수", format="%d"),
        "in_tokens": st.column_config.NumberColumn("입력 토큰", format="localized"),
        "out_tokens": st.column_config.NumberColumn("출력 토큰", format="localized"),
        "reasoning_tokens": st.column_config.NumberColumn("추론 토큰", format="localized", help="출력 토큰에 포함된 값"),
        "total_tokens": st.column_config.NumberColumn("총 토큰", format="localized"),
        "avg_tokens": st.column_config.NumberColumn("응답당 평균", format="%.0f"),
        "share": st.column_config.ProgressColumn("토큰 비중", format="percent", min_value=0, max_value=1),
        "cost_usd": st.column_config.NumberColumn("예상 비용", format="$%.4f"),
    },
)

# 기간 × 모델 표 (차트의 표 보기, 오른쪽 위 메뉴에서 CSV 다운로드 가능)
st.subheader(f"{PERIOD_LABELS[period]} 상세")
pivot = (
    frame.assign(label=[period_label(p, period) for p in frame["period"]])
    .pivot_table(index=["period", "label"], columns="model", values=metric, aggfunc="sum", fill_value=0)
    .reindex(columns=models, fill_value=0)
    .sort_index(ascending=False)
    .reset_index(level="label")
    .reset_index(drop=True)
)
pivot["합계"] = pivot[models].sum(axis=1)
number_format = "$%.4f" if metric == "cost_usd" else "localized"
st.dataframe(
    pivot.rename(columns={"label": "기간"}),
    hide_index=True,
    width="stretch",
    column_config={m: st.column_config.NumberColumn(m, format=number_format) for m in [*models, "합계"]},
)
