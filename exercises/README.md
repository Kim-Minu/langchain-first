# 🧪 LangChain 학습 및 실습 예제 (Exercises & Playground)

초기 LangChain 에이전트 생성, 도구(Tool) 연동, 메모리 세이버, LLM 호출 패턴 등을 학습하기 위해 작성된 독립 실행 스크립트 모음입니다.

---

### 📂 파일 목록 및 설명

| 파일명 | 설명 |
| :--- | :--- |
| **`first_ex1.py`** | GCP API Key 환경변수를 이용한 기본 Google GenAI 에이전트 호출 예제 |
| **`first_ex1_adc.py`** | API Key 없이 GCP ADC (`gcloud auth application-default login`) 기반 인증으로 Vertex AI Gemini 모델 및 에이전트 실행 예제 |
| **`second_ex.py`** | 로컬 Ollama 모델(`langchain_ollama.ChatOllama`)과 커스텀 함수 도구(`get_weather`, `get_current_time`)를 바인딩한 에이전트 예제 |
| **`third.py`** | `InMemorySaver` 세션 체크포인터 및 DeepAgents / URL 텍스트 추출 도구 연동 예제 |
| **`news_agent.py`** | 뉴스 요약 및 웹 API 데이터 조회를 수행하는 도구 기반 에이전트 예제 |
| **`test_inmemory_saver.py`** | LangGraph의 `InMemorySaver`를 이용한 멀티턴 대화 히스토리 및 상태 영속화 테스트 |
| **`main.py`** | URL 문서 읽기(`fetch_text_from_url`) 및 문학 데이터 분석 보조 에이전트 예제 |

---

### 🚀 실행 방법
프로젝트 루트에서 `uv run`으로 개별 스크립트를 실행할 수 있습니다:
```bash
uv run python exercises/first_ex1_adc.py
uv run python exercises/second_ex.py
```

