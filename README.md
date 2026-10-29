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
| `SEARCH_CACHE_TTL_SECONDS` | `chat_app/web_chat.py` | Vertex AI Grounding 답변 캐시 유효 시간(초, 기본 `600`, `0`이면 끔). LangGraph `BaseCache`(PostgreSQL) + `wrap_model_call` 미들웨어 |
| `MAX_OUTPUT_TOKENS` | `chat_app/web_chat.py` | 출력 토큰 상한의 기본값 (기본 `8192`, 화면의 모델 메뉴에서 조정 가능 · Gemini는 추론 토큰 포함) |
| `DATABASE_URL` | `chat_app/web_chat.py` | 대화 내역 저장용 PostgreSQL (기본값: `docker-compose.yml`의 로컬 DB `localhost:5433`) |

예제 코드는 `.env`를 자동으로 읽지 않으므로(`load_dotenv` 주석 처리) 셸에서 직접 export 하거나 uv의 `--env-file` 옵션을 사용합니다.

```bash
export GCP_API_KEY=...
# 또는
uv run --env-file .env python exercises/news_agent.py
```

### 웹 챗봇 (`chat_app/web_chat.py`) 실행

대화 내역을 PostgreSQL에 저장하므로 DB를 먼저 띄웁니다.

```bash
docker compose up -d                 # 로컬 PostgreSQL (127.0.0.1:5433)
uv run streamlit run chat_app/web_chat.py
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

| 화면 | 파일 | 내용 |
| --- | --- | --- |
| PDF RAG 실습 | `rag_app/app.py` | PDF 업로드 → 인덱싱 4단계 · 질문 4단계의 단계별 출력 확인, DeepSeek / Gemma / Gemini / Claude Haiku / Ollama 지원 |
| OCR · RAG 평가 | `rag_app/pages/ocr_eval.py` | 스캔 PDF를 Gemini OCR로 읽어 인덱싱 → 질문 · 정답 세트로 정답률 · 검색 순위 측정 → 설정별 비교 |
| LLM 성능 벤치마크 | `rag_app/pages/benchmark.py` | RAG 및 단독 질의 환경에서 5대 모델의 TTFT, 처리속도(t/s), 1회 예상 비용 실측 비교 |
| 과금 대시보드 | `rag_app/pages/billing_dashboard.py` | Google Cloud Model Garden 공식 단가 기준 PostgreSQL 영구 저장 실시간 과금 추적 |

사이드바 네비게이션 메뉴로 위 4개 화면을 자유롭게 전환할 수 있습니다.

#### CLI 성능 벤치마크
터미널에서 직접 모델별 속도와 비용을 측정할 수도 있습니다:
```bash
uv run python rag_app/benchmark_cli.py --rag --top-k 4
```

#### OCR · RAG 평가 화면 사용 순서

1. 평가 문서를 `rag_app/eval/data/`에 둡니다 (`han-gang_1-20.pdf`, `han-gang_1-100.pdf`). 이 폴더는 저작물(책 스캔본)과 OCR 결과를 담아 git에서 제외되어 있습니다. 사이드바 **평가 문서**에서 고르면 PDF · 질문 세트 · 컬렉션이 함께 바뀌고, 다른 PDF는 화면에서 업로드해도 됩니다.
2. **1. OCR · 인덱싱** 탭: `OCR 실행` → 원본 페이지와 OCR 결과를 나란히 확인 → `인덱싱 실행`. OCR 결과는 **페이지 이미지** · 모델 · 프롬프트별로 `eval/data/ocr_pages/`에 캐시되어, 다시 실행하거나 같은 페이지를 담은 다른 PDF(1~20쪽 → 1~100쪽)를 OCR해도 API를 호출하지 않습니다. 실패한 페이지만 다음 실행 때 이어서 OCR합니다.
3. **2. 질문 · 정답** 탭: 평가 세트(`rag_app/eval/han_gang_qa.json` 1~20쪽, `han_gang_100_qa.json` 1~100쪽)를 확인 · 수정합니다. Gemini로 새로 생성할 수도 있습니다.
4. **3. RAG 평가** 탭: 질문 세트(원문 질문 / 사용자 말투 질문)를 고르고 `평가 실행`. 점수 · 정답 페이지 검색률 · MRR과 문항별 판정이 나옵니다.
5. 사이드바 설정(청크 크기, 페이지 연결, top-k 등)을 바꾸고 다시 인덱싱 · 평가하면 **4. 실행 비교** 탭에 결과가 쌓입니다.

평가 결과와 개선 기록: [04-evaluation.md](docs/rag-pdf-pipeline/04-evaluation.md) (1~20쪽) · [05-evaluation-100p.md](docs/rag-pdf-pipeline/05-evaluation-100p.md) (1~100쪽, 규모 효과)

Vertex AI 예제는 GCP 프로젝트 ID가 코드에 `project="iceu-songpa05"`로 지정되어 있으니, 본인 프로젝트로 바꿔서 사용하세요.

## 초기 실습 및 예제 스크립트 (`exercises/`)

| 파일 | 모델 | 내용 |
| --- | --- | --- |
| [first_ex1.py](exercises/first_ex1.py) | Gemini 2.5 Flash (Vertex AI) | 날씨·현재 시각 도구를 가진 가장 기본적인 에이전트 |
| [second_ex.py](exercises/second_ex.py) | Ollama `gemma4:e2b` | 같은 에이전트를 로컬 모델로 실행 |
| [third.py](exercises/third.py) | Gemini 2.5 Flash (Vertex AI) | URL에서 『위대한 개츠비』 원문을 가져와 분석. `create_agent`와 `create_deep_agent` 결과 비교, `InMemorySaver` 체크포인터 사용 |
| [main.py](exercises/main.py) | Gemini (`init_chat_model`) | `third.py`와 같은 실험을 `init_chat_model`로 모델을 초기화해 수행 |
| [news_agent.py](exercises/news_agent.py) | Gemini 2.5 Flash (Vertex AI) | 뉴스 기사 URL을 읽고 카테고리·제목을 Pydantic 스키마(`response_format`)로 구조화해 추출 |
| [first_ex1_adc.py](exercises/first_ex1_adc.py) | Gemini (ADC 인증) | API Key 없이 GCP ADC로 구동하는 에이전트 |
| [test_inmemory_saver.py](exercises/test_inmemory_saver.py) | Gemini (Vertex AI) | LangGraph `InMemorySaver` 체크포인터 상태 영속화 테스트 |

```bash
uv run python exercises/first_ex1.py
uv run python exercises/second_ex.py     # 사전에: ollama pull gemma4:e2b
uv run python exercises/third.py
uv run python exercises/news_agent.py
```
