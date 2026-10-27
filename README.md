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
| `SEARCH_CACHE_TTL_SECONDS` | `web_chat.py` | Vertex AI Grounding 답변 캐시 유효 시간(초, 기본 `600`, `0`이면 끔). LangGraph `BaseCache`(PostgreSQL) + `wrap_model_call` 미들웨어 |
| `MAX_OUTPUT_TOKENS` | `web_chat.py` | 출력 토큰 상한의 기본값 (기본 `8192`, 화면의 모델 메뉴에서 조정 가능 · Gemini는 추론 토큰 포함) |
| `DATABASE_URL` | `web_chat.py` | 대화 내역 저장용 PostgreSQL (기본값: `docker-compose.yml`의 로컬 DB `localhost:5433`) |

예제 코드는 `.env`를 자동으로 읽지 않으므로(`load_dotenv` 주석 처리) 셸에서 직접 export 하거나 uv의 `--env-file` 옵션을 사용합니다.

```bash
export GCP_API_KEY=...
# 또는
uv run --env-file .env python news_agent.py
```

### 웹 챗봇 (web_chat.py) 실행

대화 내역을 PostgreSQL에 저장하므로 DB를 먼저 띄웁니다.

```bash
docker compose up -d                 # 로컬 PostgreSQL (127.0.0.1:5433)
uv run streamlit run web_chat.py
```

- 에이전트 대화 기억: `PostgresSaver` 체크포인터 (`checkpoints*` 테이블)
- 사이드바 대화 목록 · 화면 메시지 · 토큰/비용: `chat_threads`, `chat_messages` 테이블
- 현재 대화는 URL `?t=<thread_id>`에 남아 새로고침이나 앱 재시작 후에도 이어집니다.

### PDF RAG 실습 (rag_app/) 실행

PDF를 인덱싱(문서 읽기 → 청크 → 임베딩 → pgvector 저장)하고 질문에 답하는(질문 임베딩 → 검색 → 프롬프트 → Gemini 답변) 과정을 단계별 출력과 함께 보여 줍니다. 임베딩 · 답변은 Gemini(Vertex AI), 벡터 스토어는 위 도커 PostgreSQL(pgvector)을 사용합니다.

```bash
docker compose up -d
gcloud auth application-default login   # Vertex AI 인증이 만료된 경우
uv run streamlit run rag_app/app.py
```

청크 크기 · 겹침 · top-k · 지시문 등은 사이드바에서 바꿀 수 있습니다. 기획 · 설계 문서: [docs/rag-pdf-pipeline/](docs/rag-pdf-pipeline/01-plan.md)

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
