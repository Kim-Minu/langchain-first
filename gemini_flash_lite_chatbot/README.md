# Gemini 3.5 Flash Lite Chatbot (LangChain)

Google Cloud Platform(GCP) Vertex AI 환경의 **`gemini-3.5-flash-lite`** 모델을 활용하여, **LangChain(랭체인)** 생태계에서 초고속 응답과 경량 추론을 제공하는 대화형 챗봇 패키지입니다.

- **디렉토리**: `gemini_flash_lite_chatbot/`
- **대상 모델**: `gemini-3.5-flash-lite`
- **인증 방식**: GCP ADC (`gcloud auth application-default login`) 또는 Google API Key 지원

---

## 📁 디렉토리 구조

```text
gemini_flash_lite_chatbot/
├── __init__.py           # 패키지 익스포트 (GeminiFlashLiteService, GeminiFlashLiteChatbot 등)
├── config.py             # GCP 프로젝트 ID, 리전, 모델 파라미터(reasoning_effort 등) 설정
├── gemini_service.py     # BaseChatModel 기반 gemini-3.5-flash-lite 래퍼 서비스
├── chatbot.py            # LangChain LCEL, 대화 메모리 보존, 실시간 스트리밍 대화 엔진
├── cli_chat.py           # 터미널 대화형 인터랙티브 CLI 챗봇
├── streamlit_chat.py     # Streamlit 기반 모던 웹 챗봇 UI
├── test_service.py       # LangChain 연동 자동화 단위 테스트
└── README.md             # 사용 및 실행 가이드
```

---

## ⚙️ 환경변수 설정 (`.env`)

필요에 따라 프로젝트 루트 또는 `gemini_flash_lite_chatbot/.env`에 설정할 수 있습니다:

```bash
# GCP 프로젝트 및 리전
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
export GOOGLE_CLOUD_LOCATION="global"  # 또는 us-central1

# 모델 식별자 (기본값: gemini-3.5-flash-lite)
export GEMINI_FLASH_LITE_MODEL="gemini-3.5-flash-lite"

# 추론 수준 (low, medium, high)
export GEMINI_REASONING_EFFORT="low"
```

---

## 🚀 빠른 시작 및 테스트

### 1. 자동화 단위 테스트
```bash
python3 gemini_flash_lite_chatbot/test_service.py
```

### 2. 터미널 대화형 CLI 챗봇
```bash
# 실시간 GCP Vertex AI 모드 실행
python3 gemini_flash_lite_chatbot/cli_chat.py

# 로컬 Mock 테스트 모드로 실행 (오프라인 환경)
python3 gemini_flash_lite_chatbot/cli_chat.py --mock
```
> 터미널에서 `/info`(설정 조회), `/reset`(대화 초기화), `/history`(기록 보기), `/mode`(모드 전환), `/exit` 지원

### 3. Streamlit 웹 챗봇
```bash
streamlit run gemini_flash_lite_chatbot/streamlit_chat.py
```
- 사이드바에서 Reasoning Effort(low, medium, high), Temperature, Max Output Tokens 실시간 조절
- 스트리밍 타이핑 효과 및 깔끔한 대화 화면

---

## 💻 Python 코드에서 LangChain 활용 예시

```python
from gemini_flash_lite_chatbot import GeminiFlashLiteService, GeminiFlashLiteChatbot
from langchain_core.messages import HumanMessage

# 1. 모델 서비스 인스턴스 초기화
llm = GeminiFlashLiteService(
    model_name="gemini-3.5-flash-lite",
    project_id="iceu-songpa05",
    location="global",
    reasoning_effort="low",
)

# 2. 스트리밍 질의
for chunk in llm.stream([HumanMessage(content="Flash Lite 모델의 장점은?")]):
    print(chunk.content, end="", flush=True)

# 3. 대화 메모리 챗봇
bot = GeminiFlashLiteChatbot()
reply = bot.ask("안녕하세요! 오늘 기분 좋은 하루네요.")
print(reply)
```

