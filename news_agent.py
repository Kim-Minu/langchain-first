import os
import urllib.error
import urllib.request
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# 1. GCP Vertex AI API 키 확인
google_api_key = os.getenv("GCP_API_KEY")
if not google_api_key or google_api_key == "your_gemini_api_key_here":
    raise ValueError(
        "GCP_API_KEY가 설정되지 않았습니다.\n"
        "환경변수 GCP_API_KEY를 설정하거나 .env 파일에 실제 발급받은 API 키를 입력해 주세요."
    )

# 2. 요청하신 설정대로 모델 정의
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    vertexai=True,
    google_api_key=google_api_key,
    project="iceu-songpa05",
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
)


# 3. Pydantic을 사용한 구조화된 출력(Structured Output) 스키마 정의
class NewsArticleOutput(BaseModel):
    category: str = Field(description="뉴스 기사의 카테고리 (예: 정치, 경제, 사회, 문화, IT/과학 등)")
    title: str = Field(description="뉴스 기사의 정확한 제목")


# 4. 웹 페이지 기사 본문을 가져오는 Tool 함수 정의
@tool
def fetch_web_page(url: str) -> str:
    """Fetch the HTML/text content of a web page given its URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            return content
    except urllib.error.URLError as e:
        return f"웹 페이지를 가져오는 중 오류가 발생했습니다: {e}"


# 5. response_format 인자에 Pydantic 스키마를 전달하여 구조화된 출력 에이전트 생성
agent = create_agent(
    model=model,
    tools=[fetch_web_page],
    response_format=NewsArticleOutput,
    system_prompt=(
        "You are an expert news analyst. Given a news article URL, "
        "use the `fetch_web_page` tool to read the article content. "
        "Extract the news category and title accurately according to the specified output format."
    ),
)

# 6. 에이전트 실행
if __name__ == "__main__":
    article_url = "https://www.nocutnews.co.kr/news/6595366"
    print(f"기사 분석 시작: {article_url}\n", flush=True)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"다음 기사 링크에서 뉴스 카테고리와 제목을 추출해줘:\n{article_url}",
                }
            ]
        }
    )

    # 7. 구조화된 응답(structured_response) 출력
    structured_data = result.get("structured_response")

    print("=== [구조화된 출력 결과] ===")
    if structured_data:
        print(f"카테고리 : {structured_data.category}")
        print(f"제목     : {structured_data.title}")
        print("\n전체 Pydantic 모델 객체:")
        print(structured_data)
        print("\nJSON 변환 결과:")
        print(structured_data.model_dump_json(indent=2))
    else:
        # 응답이 messages 형태로 반환된 경우 대비
        print(result["messages"][-1].content)

