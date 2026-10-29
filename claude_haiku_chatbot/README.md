# Anthropic Claude Haiku 5.5 LangChain 챗봇 (GCP Vertex AI)

Google Cloud Platform(GCP) Vertex AI 환경에서 호스팅되는 Anthropic의 초경량·고성능 모델 **Claude Haiku 5.5 (`claude-haiku-5-5`)**를 LangChain 표준 LCEL(LangChain Expression Language) 인터페이스로 통합한 챗봇 패키지입니다.

`anthropic.AnthropicVertex` SDK를 기반으로 구현되어 기업 보안 환경 및 GCP Vertex AI 엔드포인트(`region="global"`)와 완벽하게 호환됩니다.

---

## 📁 디렉토리 구조

```text
claude_haiku_chatbot/
├── __init__.py           # 패키지 익스포트
├── config.py             # 모델 및 GCP Vertex AI 설정 관리
├── claude_service.py     # BaseChatModel 상속 커스텀 LangChain 서비스 (AnthropicVertex 연동)
├── chatbot.py            # LangChain LCEL 체인 및 멀티턴 대화 메모리 엔진
├── cli_chat.py           # 터미널 기반 대화형 CLI 애플리케이션
├── streamlit_chat.py     # 브라우저 기반 Streamlit 웹 챗봇
├── test_service.py       # 단위 테스트 (5개 테스트 케이스 검증)
└── README.md             # 프로젝트 안내 문서
```

---

## ⚙️ 주요 기능 및 사양

- **모델명**: `claude-haiku-5-5`
- **GCP Vertex AI 리전**: `global` (Anthropic Vertex 표준)
- **LangChain 네이티브 통합**: `BaseChatModel` 상속 및 `ChatPromptTemplate | service | StrOutputParser` LCEL 체인 구현
- **스트리밍 지원**: 실시간 토큰 스트리밍 (`stream()`, `stream_ask()`)
- **시스템 프롬프트 자동 분리**: Anthropic API 규격(최상위 `system` 파라미터 + `messages` 내 `user`/`assistant` 턴)에 맞게 `SystemMessage`를 자동 변환
- **모의(Mock) 모드 내장**: 외부 네트워크 연결이나 GCP 권한이 없는 격리/테스트 환경에서도 전체 파이프라인 단위 테스트 가능

---

## 🔑 환경 설정 및 인증

### 1. 필수 패키지 설치
```bash
pip install 'anthropic[vertex]' langchain-core streamlit
```

### 2. GCP 인증 설정
GCP Vertex AI 권한이 부여된 Application Default Credentials(ADC)를 생성합니다.
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 3. 환경 변수 설정 (`.env`)
프로젝트 루트 또는 `claude_haiku_chatbot/.env`에 다음 설정을 지정할 수 있습니다.
```env
PROJECT_ID=YOUR_PROJECT_ID
REGION=global
CLAUDE_MODEL=claude-haiku-5-5
CLAUDE_MAX_TOKENS=1024
CLAUDE_MOCK_MODE=false
```

> [!NOTE]
> `claude-haiku-5-5` 모델은 Anthropic Vertex AI 사양상 `temperature` 파라미터가 deprecated되어 요청 파라미터에서 제외됩니다.

---

## 🚀 실행 방법

### 1. 인터랙티브 CLI 챗봇 실행
터미널에서 실시간 스트리밍 대화를 나눌 수 있습니다:

```bash
# 실시간 Vertex AI 모드
python3 claude_haiku_chatbot/cli_chat.py --project YOUR_PROJECT_ID

# 오프라인 Mock 테스트 모드
python3 claude_haiku_chatbot/cli_chat.py --mock
```

**CLI 특수 명령어**:
- `/reset` : 대화 내역 초기화
- `/history` : 누적된 대화 기록 확인
- `/system [프롬프트]` : 시스템 프롬프트 확인 또는 변경
- `/info` : 현재 프로젝트, 리전, 모델 파라미터 확인
- `/mode` : 실시간 API 모드 ↔ Mock 모드 즉시 전환
- `/exit` : 프로그램 종료

---

### 2. Streamlit 웹 챗봇 실행
브라우저 환경에서 대화할 수 있는 Web UI를 구동합니다:

```bash
streamlit run claude_haiku_chatbot/streamlit_chat.py
```
- 사이드바에서 Project ID, Region, Temperature, Max Output Tokens, System Prompt를 실시간으로 조정할 수 있습니다.
- Mock 모드 토글을 통해 오프라인 테스트가 가능합니다.

---

### 3. 단위 테스트 실행
```bash
python3 claude_haiku_chatbot/test_service.py
```

---

## 💻 Python 코드에서 직접 사용하기

### 1. 기본 LangChain LCEL 파이프라인
```python
from claude_haiku_chatbot.claude_service import ClaudeHaikuVertexChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 모델 인스턴스 생성
llm = ClaudeHaikuVertexChatModel(
    project_id="YOUR_PROJECT_ID",
    region="global",
    model_name="claude-haiku-5-5",
    max_tokens=1024,
)

# 2. LCEL 체인 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 지능형 AI 비서입니다."),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()

# 3. 호출 (동기 또는 스트리밍)
# 동기 호출
response = chain.invoke({"question": "파리 여행 3일 코스를 추천해줘."})
print(response)

# 실시간 스트리밍
for chunk in chain.stream({"question": "간단한 파이썬 팁을 알려줘."}):
    print(chunk, end="", flush=True)
```

### 2. 멀티턴 챗봇 엔진 (`ClaudeHaikuChatbot`) 활용
```python
from claude_haiku_chatbot.chatbot import ClaudeHaikuChatbot

bot = ClaudeHaikuChatbot()

# 첫 번째 질문
print(bot.ask("내 이름은 민우야."))

# 두 번째 질문 (이전 대화 기억)
print(bot.ask("내 이름이 뭐라고 했지?"))
```

