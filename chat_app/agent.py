"""모델 생성 및 검색 방식(Vertex AI Grounding)별 에이전트 빌드."""

import google.auth
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.cache.base import BaseCache
from langgraph.checkpoint.base import BaseCheckpointSaver

from chat_app.config import (
    GEMINI_MODEL,
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_CLOUD_PROJECT,
    MAX_OUTPUT_TOKENS,
    SEARCH_CACHE_TTL_SECONDS,
    SEARCH_GROUNDING,
    is_gemini,
)
from chat_app.content_normalize import NormalizeContentMiddleware
from chat_app.grounding_cache import GroundingCacheMiddleware
from chat_app.structured import OutputFormat

BASE_SYSTEM_PROMPT = (
    "당신은 Google Gemini 스타일의 유능하고 지적이며 친절한 AI 어시스턴트입니다. "
    "사용자의 이전 대화 맥락을 기억하여 일관성 있고 풍부하게 답변하세요."
)
GROUNDING_PROMPT = (
    " 실시간 구글 검색(Google Search) 기능이 활성화되어 있으므로, "
    "최신 뉴스, 실시간 데이터, 웹 정보가 필요한 질문에는 검색 결과를 충실히 반영하여 답변하세요."
)
URL_CONTEXT_PROMPT = (
    " 사용자 메시지에 웹 페이지 링크(URL)가 있으면 해당 페이지를 직접 읽고, "
    "그 페이지에 실제로 적힌 내용을 근거로 답변하세요. 페이지에 없는 내용은 지어내지 마세요."
)
CODE_EXECUTION_PROMPT = (
    " 계산, 데이터 처리, 통계, 코드 작성 요청에는 코드 실행 도구로 Python 코드를 직접 실행해 결과를 확인한 뒤 "
    "그 결과를 근거로 답하세요. 실행한 코드와 실행 결과는 화면에 자동으로 표시되므로, "
    "답변 본문에는 코드를 다시 적지 말고 결과의 의미와 해석에 집중하세요."
)


def _create_gemini_llm(max_output_tokens: int):
    credentials, default_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        vertexai=True,
        credentials=credentials,
        project=GOOGLE_CLOUD_PROJECT or default_project,
        location=GOOGLE_CLOUD_LOCATION,
        reasoning_effort="low",
        max_output_tokens=max_output_tokens,
        streaming=True,
        timeout=120,
        max_retries=0,
    )


def _create_ollama_llm(max_output_tokens: int):
    # langchain-ollama는 Gemma 선택 시에만 필요하므로 지연 import (미설치 시 ImportError → 화면에서 안내)
    from langchain_ollama import ChatOllama

    return ChatOllama(model="gemma4:e2b", temperature=0, num_predict=max_output_tokens)


def create_agent_for_model(
    model_name: str,
    search_mode: str,
    checkpointer: BaseCheckpointSaver,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    cache: BaseCache | None = None,
    output_format: OutputFormat | None = None,
    url_context: bool = False,
    code_execution: bool = False,
):
    """
    선택한 모델에 맞춰 LLM 객체를 생성하고, 검색 방식에 따라 에이전트를 빌드합니다.
    - Grounding: Gemini 내장 google_search 도구 (Google 서버에서 검색까지 처리, Gemini 전용)
    - url_context: Gemini 내장 url_context 도구 (메시지 속 링크 페이지를 Gemini가 직접 읽음, Gemini 전용)
    - code_execution: Gemini 내장 code_execution 도구 (Google 샌드박스에서 Python 실행 후 결과 확인, Gemini 전용)
    - max_output_tokens: 답변 1회당 출력 토큰 상한 (Gemini: max_output_tokens, Ollama: num_predict)
    - cache: LangGraph 캐시. 내장 도구(검색 · 링크 읽기) 사용 시 같은 맥락의 답변을 TTL 동안 재사용
    - output_format: 구조화된 출력 형식(원샷 JSON 예시 또는 '라벨 : 값' 텍스트 템플릿). 주어지면 예시에서 만든
      JSON Schema로 ProviderStrategy(모델 네이티브 구조화 출력)를 적용하고, 예시를 시스템 프롬프트에 원샷으로 넣음
    """
    use_grounding = search_mode == SEARCH_GROUNDING and is_gemini(model_name)
    use_url_context = url_context and is_gemini(model_name)

    if is_gemini(model_name):
        llm = _create_gemini_llm(max_output_tokens)
    else:
        llm = _create_ollama_llm(max_output_tokens)

    system_prompt = BASE_SYSTEM_PROMPT
    tools: list[dict] = []
    middleware = []
    if use_grounding:
        # Vertex AI Gemini의 내장 구글 검색 도구 (dict 도구는 ToolNode 없이 모델에만 전달됨)
        # bind_tools로 미리 묶지 않고 create_agent에 넘겨야 구조화된 출력(response_format)과 함께 바인딩됨
        tools.append({"google_search": {}})
        system_prompt += GROUNDING_PROMPT
    if use_url_context:
        tools.append({"url_context": {}})
        system_prompt += URL_CONTEXT_PROMPT
    if code_execution and is_gemini(model_name):
        # 로컬 Python 실행(@tool)은 이 컴퓨터에서 임의 코드가 돌고, 함수 도구라 내장 검색 도구와 섞을 수도 없음
        # → Google 샌드박스에서 실행되는 Gemini 내장 도구 사용
        tools.append({"code_execution": {}})
        system_prompt += CODE_EXECUTION_PROMPT
    if tools and cache is not None and SEARCH_CACHE_TTL_SECONDS > 0:
        middleware.append(GroundingCacheMiddleware(cache, SEARCH_CACHE_TTL_SECONDS, GEMINI_MODEL, max_output_tokens))
    if tools:
        # 미들웨어는 앞쪽이 바깥쪽 → 모델에 가장 가까운 안쪽에 두어 캐시 · 체크포인트 모두 정규화된 메시지를 저장
        middleware.append(NormalizeContentMiddleware())

    response_format = None
    if output_format is not None:
        # ToolStrategy(출력용 함수 도구)는 Gemini에서 내장 도구와 한 요청에 섞을 수 없으므로 ProviderStrategy 사용
        response_format = ProviderStrategy(output_format.json_schema())
        system_prompt += output_format.system_prompt()

    return create_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        middleware=middleware,
        system_prompt=system_prompt,
        response_format=response_format,
    )
