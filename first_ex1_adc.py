"""
API Key 없이 GCP ADC (gcloud auth application-default login) 인증만으로
Vertex AI Gemini 모델 및 LangChain 에이전트를 실행하는 예제 코드입니다.
"""

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime

import google.auth
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# 1. GCP ADC 인증 및 프로젝트 ID 설정 (API Key 불필요)
# 사전 준비: 터미널에서 `gcloud auth application-default login` 실행 완료 상태여야 합니다.
# ADC를 명시적으로 전달해 환경변수의 API 키와 혼용하지 않으며, gcloud 기본 프로젝트 ID도 함께 가져옵니다.
credentials, default_project = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or default_project
if not PROJECT_ID:
    raise ValueError(
        "Google Cloud 프로젝트 ID를 찾을 수 없습니다. "
        "`gcloud config set project 프로젝트ID`를 실행하거나 GOOGLE_CLOUD_PROJECT 환경변수를 설정하세요."
    )
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# 2. API Key 없이 Vertex AI Gemini 모델 초기화
# Gemini 3.8 실습에서는 low 추론 수준과 요청당 120초 제한을 사용합니다.
# 응답은 스트리밍 API로 받아 invoke가 최종 메시지로 모읍니다.
model = ChatGoogleGenerativeAI(
    model=MODEL_ID,
    vertexai=True,
    credentials=credentials,
    project=PROJECT_ID,
    location=LOCATION,
    reasoning_effort="low",
    streaming=True,
    timeout=120,
    max_retries=0,
)

# ------------------------------------------------------------------
# 테스트 1: 단일 LLM 호출 (제시해주신 예제)
# ------------------------------------------------------------------
print("=== [1. 단일 LLM 호출 (API Key 없음 / ADC 인증)] ===")
response = model.invoke("RAG의 원리를 한 문장으로 간단히 설명해줘.")
print(response.text)


# ------------------------------------------------------------------
# 테스트 2: Tool을 포함한 Agent 호출 (first_ex1.py와 동일한 기능)
# ------------------------------------------------------------------
def get_weather(city: str = "") -> str:
    """Get current real-time weather for a given city or the user's current location.

    Args:
        city: City name (e.g. 'Seoul', 'San Francisco', '서울').
              If empty, 'current', '내 위치', or '현재 위치', automatically detects the user's current location via IP.
    """
    try:
        lat, lon, location_name = None, None, None

        # 1. 위치 판단: 도시명이 비어있거나 '현재 위치'인 경우 IP 기반으로 내 현재 위치 자동 탐색
        is_current_location = not city or city.strip().lower() in [
            "current",
            "내 위치",
            "현재 위치",
            "현재",
            "내위치",
            "현재위치",
            "my location",
        ]

        if is_current_location:
            # IP 기반 현재 위치 Geolocation API (무료, 키 불필요)
            req = urllib.request.Request(
                "http://ip-api.com/json",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                geo = json.loads(resp.read().decode("utf-8"))
                lat = geo.get("lat")
                lon = geo.get("lon")
                city_name = geo.get("city", "")
                region = geo.get("regionName", "")
                location_name = f"{region} {city_name}".strip() or "현재 위치"
        else:
            # 특정 도시명이 전달된 경우 Open-Meteo Geocoding 검색
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city.strip())}&count=1&language=ko&format=json"
            req = urllib.request.Request(geo_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                geo = json.loads(resp.read().decode("utf-8"))
                if not geo.get("results"):
                    return f"'{city}' 위치를 찾을 수 없습니다."
                lat = geo["results"][0]["latitude"]
                lon = geo["results"][0]["longitude"]
                location_name = geo["results"][0].get("name", city)

        # 2. Open-Meteo 기상청 API로 실시간 날씨 및 강수량 조회 (무료, 키 불필요, 99.9% 가용성)
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,weather_code&timezone=auto"
        )
        req_w = urllib.request.Request(weather_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_w, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            current = data.get("current", {})
            temp = current.get("temperature_2m")
            humidity = current.get("relative_humidity_2m")
            precipitation = current.get("precipitation", 0.0)
            code = current.get("weather_code", 0)

            # WMO 기상 코드 한글 변환
            weather_desc = {
                0: "맑음",
                1: "대체로 맑음",
                2: "구름 조금",
                3: "흐림",
                45: "안개",
                48: "서리 안개",
                51: "약한 이슬비",
                53: "이슬비",
                55: "강한 이슬비",
                61: "약한 비",
                63: "보통 비",
                65: "강한 비",
                71: "약한 눈",
                73: "보통 눈",
                75: "강한 눈",
                80: "소나기",
                81: "강한 소나기",
                82: "폭우성 소나기",
                95: "뇌우",
            }.get(code, "맑거나 구름")

        # 3. Open-Meteo Air Quality API로 실시간 미세먼지(PM10, PM2.5) 조회 (무료)
        dust_info = "미세먼지 정보 없음"
        try:
            aq_url = (
                f"https://air-quality-api.open-meteo.com/v1/air-quality?"
                f"latitude={lat}&longitude={lon}&current=pm10,pm2_5&timezone=auto"
            )
            req_aq = urllib.request.Request(aq_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_aq, timeout=5) as resp_aq:
                aq_data = json.loads(resp_aq.read().decode("utf-8"))
                aq_current = aq_data.get("current", {})
                pm10 = aq_current.get("pm10")
                pm2_5 = aq_current.get("pm2_5")

                # 한국 환경부 기준 미세먼지 등급 산정
                if pm10 is not None:
                    if pm10 <= 30:
                        pm10_grade = "좋음"
                    elif pm10 <= 80:
                        pm10_grade = "보통"
                    elif pm10 <= 150:
                        pm10_grade = "나쁨"
                    else:
                        pm10_grade = "매우나쁨"
                    dust_info = f"{pm10_grade} (PM10: {pm10}µg/m³, PM2.5: {pm2_5}µg/m³)"
        except Exception:
            dust_info = "미세먼지 조회 일시 불가"

        return (
            f"{location_name}: 날씨 {weather_desc}, 기온 {temp}°C, 습도 {humidity}%, "
            f"강수량 {precipitation}mm, 미세먼지: {dust_info}"
        )

    except Exception as e:
        return f"{city or '현재 위치'}의 날씨 정보를 조회할 수 없습니다. (오류: {e})"


def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 3. Pydantic을 사용한 구조화된 출력(Structured Output) 스키마 정의
class WeatherAndTimeReport(BaseModel):
    location: str = Field(description="조회한 위치 또는 도시명")
    current_time: str = Field(description="현재 날짜 및 시각 (예: YYYY-MM-DD HH:MM:SS)")
    weather_condition: str = Field(description="현재 날씨 상태 (예: 맑음, 흐림, 비 등)")
    temperature: str = Field(description="현재 기온 (예: +18°C)")
    humidity: str = Field(description="현재 습도 (예: 65%)")
    precipitation: str = Field(description="현재 강수량 (예: 0.0mm)")
    fine_dust: str = Field(description="미세먼지 상태 및 수치 (예: 좋음 (PM10: 18µg/m³, PM2.5: 11µg/m³))")
    summary: str = Field(description="날씨, 대기질, 시각 종합 안내 한 줄 요약")


agent = create_agent(
    model=model,
    tools=[get_weather, get_current_time],
    response_format=WeatherAndTimeReport,
    system_prompt="You are a helpful assistant that provides weather, fine dust, and time information in the requested structured format.",
)

print("\n=== [2. Agent 도구 호출 및 구조화된 출력 (API Key 없음 / ADC 인증)] ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "지금 내 현재 위치의 날씨(기온, 습도, 강수량), 미세먼지 수치와 현재 시각을 알려줘."}]}
)

# 구조화된 출력 결과 확인
structured_data = result.get("structured_response")
if structured_data:
    print("✅ [구조화된 출력 결과 (Pydantic 모델)]:")
    print(f"위치     : {structured_data.location}")
    print(f"현재 시각: {structured_data.current_time}")
    print(f"날씨 상태: {structured_data.weather_condition}")
    print(f"기온     : {structured_data.temperature}")
    print(f"습도     : {structured_data.humidity}")
    print(f"강수량   : {structured_data.precipitation}")
    print(f"미세먼지 : {structured_data.fine_dust}")
    print(f"요약     : {structured_data.summary}")
    print("\n[JSON 변환 출력]:")
    print(structured_data.model_dump_json(indent=2))
else:
    print(result["messages"][-1].content)

