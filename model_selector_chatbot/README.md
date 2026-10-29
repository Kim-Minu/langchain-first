# GCP Vertex AI MaaS 모델 선택 가능 챗봇 (LangChain)

Google Cloud Platform(GCP) Vertex AI MaaS (Model-as-a-Service) 인프라 기반의 오픈/파트너 모델들을 **LangChain(랭체인)** 상에서 자유롭게 선택하고 전환하며 대화할 수 있는 챗봇 패키지입니다.

- **디렉토리**: `model_selector_chatbot/`
- **핵심 클래스**: `GcpMaasChatModel` (LangChain `BaseChatModel` 상속)
- **호환 래퍼**: `DeepSeekV4FlashApiService` (공식 명칭 호환)

---

## 🎯 지원 모델 및 GCP curl 사양

### 1. DeepSeek V4.1 Flash (MaaS)
- **공식 서비스 명칭**: `DeepSeek V4.1 Flash API Service`
- **모델 식별자**: `deepseek-ai/deepseek-v4.1-flash-maas`
- **GCP curl 요청 사양**:
```bash
curl \
  -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" https://${ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/openapi/chat/completions \
  -d '{"model":"deepseek-ai/deepseek-v4.1-flash-maas", "stream":true, "messages":[{"role": "user", "content": "Summer travel plan to Paris"}]}'
```

### 2. Google Gemma 4 26B A4B IT (MaaS)
- **공식 서비스 명칭**: `Gemma 4 26B IT API Service`
- **모델 식별자**: `google/gemma-4-26b-a4b-it-maas`
- **특징**: `max_tokens` 최대 128,000 토큰 지원, Thinking(사고) 템플릿 지원
- **GCP curl 요청 사양**:
```bash
curl \
  -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" https://${ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/openapi/chat/completions \
  -d '{"model":"google/gemma-4-26b-a4b-it-maas", "stream":false, "max_tokens":128000, "messages":[{"role": "user", "content": "Summer travel plan to Paris"}], "chat_template_kwargs": {"enable_thinking": true}}'
```

---

## 📁 디렉토리 구조

```text
model_selector_chatbot/
├── __init__.py           # 패키지 익스포트 (GcpMaasChatModel, ModelSelectorChatbot 등)
├── config.py             # 모델 카탈로그 (SUPPORTED_MODELS), ENDPOINT, REGION, PROJECT_ID 관리
├── model_service.py      # LangChain BaseChatModel 기반 GCP MaaS 코어 (Thinking 옵션 지원)
├── deepseek_service.py   # 공식 명칭 DeepSeek V4.1 Flash API Service 하위 호환 래퍼
├── chatbot.py            # 대화 메모리 유지 및 실시간 모델 동적 전환 (switch_model) 지원 챗봇
├── cli_chat.py           # 터미널 대화형 인터랙티브 CLI 챗봇 (/model, /thinking 명령어 지원)
├── streamlit_chat.py     # Streamlit 기반 모던 웹 챗봇 (모델 선택 Selectbox 및 UI 파라미터)
├── test_service.py       # 모델별 엔드포인트 URL 및 Thinking 옵션 자동화 단위 테스트
└── README.md             # 사용 및 연동 가이드
```

---

## 🚀 실행 및 테스트 방법

### 1. 자동화 단위 테스트 실행
```bash
python3 model_selector_chatbot/test_service.py
```

### 2. Streamlit 웹 챗봇 실행
```bash
streamlit run model_selector_chatbot/streamlit_chat.py
```
- 사이드바 상단 셀렉트박스에서 **DeepSeek V4.1 Flash** 또는 **Gemma 4 26B A4B IT**를 즉시 변경 가능
- Gemma 4 선택 시 **Enable Thinking (사고 모드)** 토글 활성화
- 대화 중간에 모델을 바꾸어도 이전 대화 맥락이 그대로 보존됨

### 3. 터미널 대화형 CLI 챗봇 실행
```bash
# 기본 실행 (DeepSeek Flash)
python3 model_selector_chatbot/cli_chat.py

# Gemma 4 모델 및 Thinking 모드로 실행
python3 model_selector_chatbot/cli_chat.py --model gemma-4-26b --thinking

# 오프라인 Mock 테스트 모드로 실행
python3 model_selector_chatbot/cli_chat.py --mock
```

**CLI 명령어 안내:**
- `/model` : 대화 도중 모델 목록을 보고 다른 모델로 즉시 전환
- `/thinking` : Gemma 4의 Thinking(사고) 모드 On/Off 토글
- `/info` : 현재 선택된 모델, URL, 엔드포인트 확인
- `/reset` : 대화 내역 초기화
- `/history` : 대화 기록 보기
- `/exit` : 종료

---

## 💻 Python 코드에서 LangChain 활용 예시

```python
from model_selector_chatbot import GcpMaasChatModel, ModelSelectorChatbot
from langchain_core.messages import HumanMessage

# 1. Gemma 4 26B 모델 인스턴스 (Thinking 활성화)
gemma_llm = GcpMaasChatModel(
    model_name="google/gemma-4-26b-a4b-it-maas",
    max_tokens=128000,
    enable_thinking=True,
)

# 2. DeepSeek V4.1 Flash 모델 인스턴스
deepseek_llm = GcpMaasChatModel(
    model_name="deepseek-ai/deepseek-v4.1-flash-maas",
    max_tokens=4096,
)

# 3. 대화 도중 모델 전환 챗봇
bot = ModelSelectorChatbot()

# DeepSeek로 질의
res1 = bot.ask("내 이름은 민우야.")
print("DeepSeek:", res1)

# 대화 맥락을 유지한 채 Gemma 4로 모델 전환 후 질의
bot.switch_model("gemma-4-26b", enable_thinking=True)
res2 = bot.ask("내 이름이 뭐라고 했지?")
print("Gemma 4:", res2)
```
