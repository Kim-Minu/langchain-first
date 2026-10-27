"""날씨 질문 감지 → 장소 추출(모델) → Open-Meteo 예보 조회 → 답변용 컨텍스트 (Streamlit에 의존하지 않는 순수 로직).

- 감지: 날씨 키워드가 있는 질문만 모델로 장소 · 좌표를 추출 (키워드가 없으면 추가 모델 호출 없음)
- 예보: Open-Meteo (https://open-meteo.com) — 키 없이 쓰는 공개 예보 API, 현재 · 시간별 · 일별 예보
- 컨텍스트: 조회한 예보를 질문 뒤에 붙여 모델이 카드와 같은 수치로 요약하도록 함
"""

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from chat_app.config import (
    WEATHER_DEFAULT_LATITUDE,
    WEATHER_DEFAULT_LONGITUDE,
    WEATHER_DEFAULT_PLACE,
)

WEATHER_PATTERN = re.compile(r"날씨|기온|일기\s*예보|강수|비\s*(가|와|오|올)|눈\s*(이|와|오|올)|우산|weather|forecast", re.IGNORECASE)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_SOURCE_URL = "https://open-meteo.com/"
HOURLY_COUNT = 24
REQUEST_TIMEOUT_SECONDS = 10

WEEKDAYS_KO = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
WIND_DIRECTIONS_KO = ["북", "북북동", "북동", "동북동", "동", "동남동", "남동", "남남동",
                      "남", "남남서", "남서", "서남서", "서", "서북서", "북서", "북북서"]

# WMO 날씨 코드 → (한국어 상태, 아이콘 종류)
WMO_CODES = {
    0: ("맑음", "clear"), 1: ("대체로 맑음", "clear"), 2: ("구름 조금", "partly"), 3: ("흐림", "cloudy"),
    45: ("안개", "fog"), 48: ("서리 안개", "fog"),
    51: ("약한 이슬비", "rain"), 53: ("이슬비", "rain"), 55: ("강한 이슬비", "rain"),
    56: ("어는 이슬비", "rain"), 57: ("강한 어는 이슬비", "rain"),
    61: ("약한 비", "rain"), 63: ("비", "rain"), 65: ("강한 비", "rain"),
    66: ("어는 비", "rain"), 67: ("강한 어는 비", "rain"),
    71: ("약한 눈", "snow"), 73: ("눈", "snow"), 75: ("강한 눈", "snow"), 77: ("싸락눈", "snow"),
    80: ("약한 소나기", "rain"), 81: ("소나기", "rain"), 82: ("강한 소나기", "rain"),
    85: ("약한 소낙눈", "snow"), 86: ("소낙눈", "snow"),
    95: ("뇌우", "thunder"), 96: ("우박 동반 뇌우", "thunder"), 99: ("강한 우박 동반 뇌우", "thunder"),
}

EXTRACT_PROMPT = (
    "다음 질문이 날씨(기온 · 강수 · 바람 · 일기예보)를 묻는지 판단하고, 묻는다면 대상 장소를 추출하세요.\n"
    "- place_name: 행정구역을 포함한 한국어 전체 이름 (예: 서울특별시 송파구 문정동, 일본 도쿄도 시부야구)\n"
    "- latitude / longitude: 그 장소의 위도 · 경도 (소수점 4자리)\n"
    "- 장소가 없으면 place_name을 빈 문자열로 두세요.\n\n질문: {question}"
)


class WeatherQuery(BaseModel):
    is_weather: bool = Field(description="날씨를 묻는 질문인지 여부")
    place_name: str = Field(default="", description="행정구역을 포함한 한국어 장소 이름, 없으면 빈 문자열")
    latitude: float | None = Field(default=None, description="위도")
    longitude: float | None = Field(default=None, description="경도")


def condition(code: int) -> tuple[str, str]:
    return WMO_CODES.get(code, ("알 수 없음", "cloudy"))


def wind_direction_ko(degrees: float) -> str:
    return WIND_DIRECTIONS_KO[round(degrees / 22.5) % 16]


def extract_weather_query(llm: BaseChatModel, question: str) -> tuple[WeatherQuery | None, dict | None]:
    """
    (날씨 질문이면 WeatherQuery, 아니면 None, 추출 호출의 usage_metadata)를 반환합니다.
    장소가 없으면 기본 장소(WEATHER_DEFAULT_*)로 채웁니다.
    """
    result = llm.with_structured_output(WeatherQuery, include_raw=True).invoke(
        EXTRACT_PROMPT.format(question=question)
    )
    usage = getattr(result.get("raw"), "usage_metadata", None)
    query: WeatherQuery | None = result.get("parsed")
    if query is None or not query.is_weather:
        return None, usage
    if not query.place_name or query.latitude is None or query.longitude is None:
        query = WeatherQuery(
            is_weather=True,
            place_name=WEATHER_DEFAULT_PLACE,
            latitude=WEATHER_DEFAULT_LATITUDE,
            longitude=WEATHER_DEFAULT_LONGITUDE,
        )
    return query, usage


def _get_json(url: str, params: dict) -> dict:
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        return json.load(resp)


def fetch_forecast(query: WeatherQuery) -> dict:
    """Open-Meteo 예보를 카드 · 컨텍스트용으로 정리합니다 (현재 + 지금부터 24시간 + 3일)."""
    raw = _get_json(FORECAST_URL, {
        "latitude": query.latitude,
        "longitude": query.longitude,
        "timezone": "auto",
        "forecast_days": 3,
        "wind_speed_unit": "ms",
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,weather_code,"
                   "wind_speed_10m,wind_direction_10m",
        "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code,is_day,"
                  "wind_speed_10m,wind_direction_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                 "precipitation_sum",
    })
    cur, hourly, daily = raw["current"], raw["hourly"], raw["daily"]

    # 시간별 예보는 현재 시각이 속한 시간부터 (시각 문자열은 현지 시간 ISO 형식이라 문자열 비교로 충분)
    current_hour = cur["time"][:13]
    start = next((i for i, t in enumerate(hourly["time"]) if t[:13] >= current_hour), 0)
    hours = [
        {
            "time": hourly["time"][i],
            "temp": hourly["temperature_2m"][i],
            "pop": hourly["precipitation_probability"][i] or 0,
            "precip": hourly["precipitation"][i] or 0,
            "code": hourly["weather_code"][i],
            "is_day": hourly["is_day"][i],
            "wind_speed": hourly["wind_speed_10m"][i],
            "wind_dir": hourly["wind_direction_10m"][i],
        }
        for i in range(start, min(start + HOURLY_COUNT, len(hourly["time"])))
    ]
    days = [
        {
            "date": daily["time"][i],
            "code": daily["weather_code"][i],
            "max": daily["temperature_2m_max"][i],
            "min": daily["temperature_2m_min"][i],
            "pop": daily["precipitation_probability_max"][i] or 0,
            "precip": daily["precipitation_sum"][i] or 0,
        }
        for i in range(len(daily["time"]))
    ]
    return {
        "place": query.place_name,
        "latitude": query.latitude,
        "longitude": query.longitude,
        "timezone": raw.get("timezone", ""),
        "current": {
            "time": cur["time"],
            "temp": cur["temperature_2m"],
            "feels_like": cur["apparent_temperature"],
            "humidity": cur["relative_humidity_2m"],
            "code": cur["weather_code"],
            "is_day": cur["is_day"],
            "wind_speed": cur["wind_speed_10m"],
            "wind_dir": cur["wind_direction_10m"],
        },
        "hourly": hours,
        "daily": days,
        "source": "Open-Meteo",
        "source_url": FORECAST_SOURCE_URL,
    }


def weekday_ko(iso_time: str) -> str:
    return WEEKDAYS_KO[datetime.fromisoformat(iso_time).weekday()]


def weather_context(data: dict) -> str:
    """조회한 예보를 질문 뒤에 붙일 참고 자료 문자열로 만듭니다 (모델이 카드와 같은 수치로 답하도록)."""
    cur = data["current"]
    lines = [
        f"[실시간 날씨 데이터 · 출처 {data['source']} · 기준 {cur['time']} ({data['timezone']})]",
        f"장소: {data['place']}",
        (
            f"현재: {condition(cur['code'])[0]}, {cur['temp']}°C (체감 {cur['feels_like']}°C), "
            f"습도 {cur['humidity']}%, 바람 {wind_direction_ko(cur['wind_dir'])}풍 {cur['wind_speed']} m/s"
        ),
    ]
    for day in data["daily"]:
        lines.append(
            f"{day['date']} ({weekday_ko(day['date'])}): {condition(day['code'])[0]}, "
            f"최고 {day['max']}°C / 최저 {day['min']}°C, 강수 확률 {day['pop']}%, 강수량 {day['precip']} mm"
        )
    hourly = ", ".join(
        f"{h['time'][11:16]} {condition(h['code'])[0]} {h['temp']}°C 강수 {h['pop']}%"
        for h in data["hourly"][:12]
    )
    lines.append(f"시간별(12시간): {hourly}")
    lines.append(
        "화면에 이 데이터로 만든 날씨 카드가 이미 표시되므로, 표를 다시 그리지 말고 "
        "현재 날씨 · 오늘 최고/최저 · 앞으로의 변화를 2~3문장으로 자연스럽게 요약하세요. "
        "다른 날짜를 물었다면 그 날짜의 예보를 중심으로 답하세요."
    )
    return "\n".join(lines)
