from datetime import datetime
from langchain.agents import create_agent
from langchain_ollama import ChatOllama  # 1) ChatOllama 임포트


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 2) Ollama 모델 설정
model = ChatOllama(
    model="gemma4:e2b",  # Ollama에 설치된 모델 태그명 입력 (예: gemma2:2b)
    temperature=0,
)

# 3) Agent에 Ollama 모델 연결
agent = create_agent(
    model=model,
    tools=[get_weather, get_current_time],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "지금 몇 시야? 샌프란시스코 날씨도 알려줘."}]}
)

print(result["messages"][-1].content_blocks)