from datetime import datetime
import os
#from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# .env 파일이 있는 경우 환경변수 로드
#load_dotenv(override=True)

# 환경변수에서 GCP Vertex AI API 키 가져오기
google_api_key = os.getenv("GCP_API_KEY")
if not google_api_key or google_api_key == "your_gemini_api_key_here":
    raise ValueError(
        "GCP_API_KEY가 설정되지 않았습니다.\n"
        "환경변수 GCP_API_KEY를 설정하거나 .env 파일에 실제 발급받은 API 키를 입력해 주세요."
    )


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# GCP Vertex AI의 Gemini 모델을 사용하도록 설정 (vertexai=True)
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    vertexai=True,
    google_api_key=google_api_key,
    project="iceu-songpa05",
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
)

agent = create_agent(
    model=model,
    tools=[get_weather, get_current_time],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "지금 몇 시야? 샌프란시스코 날씨도 알려줘."}]}
)
print(result["messages"][-1].content_blocks)