# langchain-first

LangChain v1의 `create_agent`와 `deepagents`의 `create_deep_agent`를 사용해 도구(tool)를 호출하는 에이전트를 만들어 보는 예제 모음입니다.

## 요구 사항

- Python 3.14 이상
- [uv](https://docs.astral.sh/uv/)
- 예제에 따라 다음 중 하나
  - Google Gemini API 키 (Vertex AI 또는 Gemini API)
  - 로컬 [Ollama](https://ollama.com/)

## 설치

```bash
uv sync
```

## 환경 변수

| 변수 | 사용하는 예제 | 설명 |
| --- | --- | --- |
| `GCP_API_KEY` | `first_ex1.py`, `third.py`, `news_agent.py` | Vertex AI용 API 키 |
| `GOOGLE_CLOUD_LOCATION` | 위와 동일 | Vertex AI 리전 (기본값 `us-central1`) |
| `GOOGLE_API_KEY` | `main.py` | Gemini API 키 (`init_chat_model`의 `google-genai` 공급자) |
| `GOOGLE_CSE_API_KEY` | `web_chat.py` | Custom Search JSON API 키 (검색 방식 "Custom Search (CSE)" 선택 시) |
| `GOOGLE_CSE_ID` | `web_chat.py` | Programmable Search Engine ID (`cx`) |

예제 코드는 `.env`를 자동으로 읽지 않으므로(`load_dotenv` 주석 처리) 셸에서 직접 export 하거나 uv의 `--env-file` 옵션을 사용합니다.

```bash
export GCP_API_KEY=...
# 또는
uv run --env-file .env python news_agent.py
```

Vertex AI 예제는 GCP 프로젝트 ID가 코드에 `project="iceu-songpa05"`로 지정되어 있으니, 본인 프로젝트로 바꿔서 사용하세요.

## 예제

| 파일 | 모델 | 내용 |
| --- | --- | --- |
| [first_ex1.py](first_ex1.py) | Gemini 2.5 Flash (Vertex AI) | 날씨·현재 시각 도구를 가진 가장 기본적인 에이전트 |
| [second_ex.py](second_ex.py) | Ollama `gemma4:e2b` | 같은 에이전트를 로컬 모델로 실행 |
| [third.py](third.py) | Gemini 2.5 Flash (Vertex AI) | URL에서 『위대한 개츠비』 원문을 가져와 분석. `create_agent`와 `create_deep_agent` 결과 비교, `InMemorySaver` 체크포인터 사용 |
| [main.py](main.py) | Gemini (`init_chat_model`) | `third.py`와 같은 실험을 `init_chat_model`로 모델을 초기화해 수행 |
| [news_agent.py](news_agent.py) | Gemini 2.5 Flash (Vertex AI) | 뉴스 기사 URL을 읽고 카테고리·제목을 Pydantic 스키마(`response_format`)로 구조화해 추출 |

## 실행

```bash
uv run python first_ex1.py
uv run python second_ex.py     # 사전에: ollama pull gemma4:e2b
uv run python third.py
uv run python main.py
uv run python news_agent.py
```
