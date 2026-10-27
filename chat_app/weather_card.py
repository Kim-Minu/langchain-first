"""날씨 카드 HTML (Gemini 날씨 카드 레이아웃: 장소 · 현재 기온 · 최고/최저 · 시간별 온도/강수량/바람 탭).

st.html은 DOMPurify(html 프로필)로 정리되어 인라인 <svg>가 지워지므로, 아이콘은 data URI <img>로 넣습니다.
탭은 라디오 입력 + CSS(:checked)로만 전환해 Streamlit 재실행 없이 동작합니다 (스타일은 styles.py의 .gm-wx-*).
"""

import base64
import html
import math
from datetime import datetime
from functools import cache

from chat_app.weather import condition, weekday_ko, wind_direction_ko

SUN = "#fdd663"
MOON = "#a8c7fa"
CLOUD = "#e3e3e3"
CLOUD_DIM = "#9aa0a6"
RAIN = "#8ab4f8"


def _sun(cx: float, cy: float, r: float) -> str:
    # Gemini 해 아이콘처럼 가장자리가 물결진 원: 중심 원 + 테두리를 따라 작은 원 12개
    bumps = "".join(
        f'<circle cx="{cx + r * math.cos(a):.2f}" cy="{cy + r * math.sin(a):.2f}" r="{r * 0.3:.2f}"/>'
        for a in (i * math.pi / 6 for i in range(12))
    )
    return f'<g fill="{SUN}"><circle cx="{cx}" cy="{cy}" r="{r}"/>{bumps}</g>'


def _moon(cx: float, cy: float, r: float) -> str:
    return (
        f'<path fill="{MOON}" d="M{cx + r * 0.35} {cy - r}a{r} {r} 0 1 0 {r * 0.65} {r * 1.55}'
        f'a{r * 0.8} {r * 0.8} 0 0 1 -{r * 0.65} -{r * 1.55}z"/>'
    )


def _cloud(x: float, y: float, s: float, fill: str = CLOUD) -> str:
    return (
        f'<path fill="{fill}" d="M{x + 12 * s} {y + 26 * s}h{26 * s}a{9 * s} {9 * s} 0 0 0 0-{18 * s}'
        f'a{12 * s} {12 * s} 0 0 0-{22 * s}-{3 * s}a{8 * s} {8 * s} 0 0 0-{4 * s} {21 * s}z"/>'
    )


@cache
def _icon_svg(kind: str, is_day: bool) -> str:
    body = {
        "clear": _sun(32, 32, 17) if is_day else _moon(30, 32, 18),
        "partly": (_sun(24, 24, 12) if is_day else _moon(22, 24, 13)) + _cloud(12, 22, 1.1),
        "cloudy": _cloud(4, 6, 1.0, CLOUD_DIM) + _cloud(10, 18, 1.1),
        "fog": _cloud(8, 4, 1.1) + "".join(
            f'<rect x="10" y="{y}" width="44" height="4" rx="2" fill="{CLOUD_DIM}"/>' for y in (40, 50)
        ),
        "rain": _cloud(8, 2, 1.1) + "".join(
            f'<path d="M{x} 42l-4 12" stroke="{RAIN}" stroke-width="4" stroke-linecap="round"/>' for x in (22, 34, 46)
        ),
        "snow": _cloud(8, 2, 1.1) + "".join(f'<circle cx="{x}" cy="{y}" r="3.5" fill="#ffffff"/>'
                                           for x, y in ((20, 46), (32, 52), (44, 46))),
        "thunder": _cloud(8, 2, 1.1) + f'<path fill="{SUN}" d="M34 36l-10 14h8l-4 12 12-16h-8l4-10z"/>',
    }[kind]
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">{body}</svg>'


def _icon(code: int, is_day: bool, size: int, label: str = "") -> str:
    svg = _icon_svg(condition(code)[1], bool(is_day))
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    alt = html.escape(label or condition(code)[0])
    return f'<img src="{uri}" width="{size}" height="{size}" alt="{alt}" title="{alt}">'


def _arrow(degrees: float) -> str:
    # 바람이 "불어가는" 방향을 가리키도록 풍향(불어오는 방향)에 180° 더함
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        f'<path fill="{CLOUD}" transform="rotate({(degrees + 180) % 360:.0f} 12 12)" d="M12 3l6 16-6-4-6 4z"/></svg>'
    )
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return f'<img src="{uri}" width="22" height="22" alt="{wind_direction_ko(degrees)}풍">'


def _hour_label(iso_time: str, first: bool) -> str:
    if first:
        return "지금"
    hour = datetime.fromisoformat(iso_time).hour
    return f"{'오전' if hour < 12 else '오후'} {hour % 12 or 12}시"


def _hourly_cells(hours: list[dict], tab: str) -> str:
    cells = []
    max_pop = max((h["pop"] for h in hours), default=0) or 1
    for i, h in enumerate(hours):
        label = f'<div class="gm-wx-time">{_hour_label(h["time"], i == 0)}</div>'
        if tab == "temp":
            body = (
                _icon(h["code"], h["is_day"], 36)
                + f'<div class="gm-wx-pop">{h["pop"]}%</div><div class="gm-wx-val">{round(h["temp"])}°</div>'
            )
        elif tab == "rain":
            height = max(4, round(48 * h["pop"] / max_pop)) if h["pop"] else 2
            body = (
                f'<div class="gm-wx-bar-wrap"><div class="gm-wx-bar" style="height:{height}px"></div></div>'
                f'<div class="gm-wx-pop">{h["pop"]}%</div><div class="gm-wx-val">{h["precip"]:g}mm</div>'
            )
        else:
            body = (
                f'<div class="gm-wx-arrow">{_arrow(h["wind_dir"])}</div>'
                f'<div class="gm-wx-pop">{wind_direction_ko(h["wind_dir"])}</div>'
                f'<div class="gm-wx-val">{h["wind_speed"]:.1f}<small>m/s</small></div>'
            )
        cells.append(f'<div class="gm-wx-hour">{label}{body}</div>')
    return "".join(cells)


def weather_card_html(data: dict, uid: str) -> str:
    """날씨 카드 HTML. uid는 카드마다 다른 값이어야 탭(라디오 그룹)이 서로 섞이지 않습니다."""
    cur = data["current"]
    today = data["daily"][0]
    name = f"wx-{uid}"
    tabs = [("temp", "온도"), ("rain", "강수량"), ("wind", "바람")]
    radios = "".join(
        f'<input type="radio" class="gm-wx-radio gm-wx-radio-{key}" name="{name}" id="{name}-{key}"'
        f'{" checked" if key == "temp" else ""}>'
        for key, _ in tabs
    )
    tab_labels = "".join(f'<label for="{name}-{key}" class="gm-wx-tab gm-wx-tab-{key}">{text}</label>' for key, text in tabs)
    panels = "".join(
        f'<div class="gm-wx-panel gm-wx-panel-{key}">{_hourly_cells(data["hourly"], key)}</div>' for key, _ in tabs
    )
    updated = cur["time"][11:16]
    return f"""
<div class="gm-wx">
  {radios}
  <div class="gm-wx-place">{weekday_ko(cur["time"])} · {html.escape(data["place"])}</div>
  <div class="gm-wx-now">
    {_icon(cur["code"], cur["is_day"], 76)}
    <span class="gm-wx-temp">{round(cur["temp"])}</span><span class="gm-wx-unit">°C</span>
    <div class="gm-wx-details">
      체감 {round(cur["feels_like"])}° · 습도 {cur["humidity"]}%<br>
      바람 {wind_direction_ko(cur["wind_dir"])}풍 {cur["wind_speed"]:.1f} m/s
    </div>
  </div>
  <div class="gm-wx-summary">
    <div>
      <div class="gm-wx-cond">{condition(cur["code"])[0]}</div>
      <div class="gm-wx-range">최고: {round(today["max"])}° &nbsp; 최저: {round(today["min"])}° &nbsp; 강수 확률: {today["pop"]}%</div>
    </div>
    <div class="gm-wx-tabs">{tab_labels}</div>
  </div>
  <div class="gm-wx-hourly">{panels}</div>
  <div class="gm-wx-source">
    <a href="{html.escape(data["source_url"])}" target="_blank" rel="noopener">{html.escape(data["source"])}</a>
    · {updated} 기준
  </div>
</div>
"""
