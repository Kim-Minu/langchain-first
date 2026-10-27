# PDF RAG 실습 앱 설계

기획 배경과 요구사항은 [01-plan.md](01-plan.md) 참고.

## 1. 전체 흐름

인덱싱(A)은 자료를 등록 · 변경할 때 한 번, 런타임(B)은 질문마다 실행된다. 두 흐름은 PostgreSQL(pgvector)에 저장된 벡터로만 연결된다.

```mermaid
flowchart LR
    subgraph A["A. 미리 준비하기 · 인덱싱 (자료 등록 · 변경 시)"]
        direction LR
        A1["A-1 문서 읽기<br/>PDF → 텍스트"] --> A2["A-2 청크로 나누기<br/>작은 원문 조각"]
        A2 --> A3["A-3 임베딩하기<br/>청크 → 숫자 배열"]
        A3 --> A4["A-4 벡터 스토어 저장<br/>벡터 + 원문 + 출처"]
    end

    DB[("PostgreSQL + pgvector<br/>langchain_pg_embedding")]
    A4 --> DB

    subgraph B["B. 질문에 답하기 · 런타임 (질문마다)"]
        direction LR
        B1["B-1 질문 입력<br/>질문도 임베딩"] --> B2["B-2 관련 청크 검색<br/>벡터 비교 → 원문"]
        B2 --> B3["B-3 프롬프트 구성<br/>질문 + 원문 + 지시문"]
        B3 --> B4["B-4 LLM → 답변<br/>근거를 읽고 생성"]
        B1 -. "원래 질문도 그대로 전달" .-> B3
    end

    DB --> B2
```

### A-1 문서 읽기 상세 (텍스트 + 이미지)

```mermaid
flowchart LR
    PDF["PDF 바이트"] --> T["PyPDFLoader<br/>페이지별 텍스트"]
    PDF --> I["pypdf page.images<br/>페이지별 이미지 (Pillow로 디코딩)"]
    I --> F{"짧은 변 64px 이상?<br/>단색 아님?"}
    F -- 아니오 --> W["건너뜀<br/>(단색은 경고 목록에 기록)"]
    F -- 예 --> G["GeminiImageReader<br/>글자 받아 적기 + 도표 설명<br/>4개씩 동시 요청 · 실패 시 1회 재시도"]
    G --> M["페이지 본문 + [이미지 설명] 텍스트"]
    T --> M
    M --> OUT["페이지별 Document"]
```

## 2. 구성 요소

```mermaid
flowchart TB
    UI["rag_app/app.py<br/>Streamlit 화면 · 사이드바 설정 · 단계별 출력"]
    P["rag_app/pipeline.py<br/>단계별 함수 + LCEL 체인"]
    C["rag_app/config.py<br/>기본값 · 범위 · 환경변수"]

    UI --> P
    UI --> C
    P --> C

    P -->|PyPDFLoader| PDF["pypdf"]
    P -->|RecursiveCharacterTextSplitter| TS["langchain-text-splitters"]
    P -->|GoogleGenerativeAIEmbeddings<br/>ChatGoogleGenerativeAI| V["Vertex AI Gemini<br/>(ADC 인증)"]
    P -->|ChatOllama · 답변 선택 시| O["로컬 Ollama<br/>gemma4:e2b"]
    P -->|PGVector| PG[("docker-compose PostgreSQL<br/>pgvector/pgvector:pg17-trixie")]
```

| 파일 | 역할 |
| --- | --- |
| [`rag_app/config.py`](../../rag_app/config.py) | 모든 기본값과 사이드바 입력 범위, 환경변수 읽기 |
| [`rag_app/pipeline.py`](../../rag_app/pipeline.py) | 그림의 8단계를 함수 하나씩으로 구현. 화면 코드와 분리해 단독 테스트 가능 |
| [`rag_app/app.py`](../../rag_app/app.py) | PDF RAG 실습 화면. 단계 실행, 결과를 `st.session_state`에 보관하고 단계별로 렌더링 |
| [`rag_app/ui_common.py`](../../rag_app/ui_common.py) | 두 화면이 공유하는 사이드바 설정(화면을 옮겨도 유지) · 캐시된 모델/스토어 · 오류 안내 |
| [`rag_app/evaluation.py`](../../rag_app/evaluation.py) | OCR(파일 캐시) · 질문/정답 생성 · Gemini 채점 · 병렬 평가(할당량 재시도) · 지표 집계 |
| [`rag_app/pages/ocr_eval.py`](../../rag_app/pages/ocr_eval.py) | OCR · RAG 평가 화면 (OCR · 인덱싱 / 질문 · 정답 / RAG 평가 / 실행 비교) |
| [`rag_app/eval/han_gang_qa.json`](../../rag_app/eval/han_gang_qa.json) | 평가 세트 20쌍. 원본 PDF · OCR 캐시는 저작물이라 git 제외 폴더 `rag_app/eval/data/`에 둠 |

## 3. 단계별 입출력

| 단계 | 함수 | 입력 | 출력 | LangChain 구성 요소 |
| --- | --- | --- | --- | --- |
| A-1 | `load_pdf` | PDF 바이트, 파일명, (선택) `GeminiImageReader` | `list[Document]` (페이지당 1개, 이미지 설명 포함) | `PyPDFLoader` + `ChatGoogleGenerativeAI`(이미지 입력) |
| A-2 | `split_documents` | 페이지 Document, 청크 크기 · 겹침 | `list[Document]` (청크) | `RecursiveCharacterTextSplitter` |
| A-3 | `embed_chunks` | 청크 | `list[list[float]]` | `GoogleGenerativeAIEmbeddings.embed_documents` |
| A-4 | `save_to_store` | 청크, 벡터 | 저장된 id 목록 | `PGVector.add_embeddings` |
| B-1 | `embed_question` | 질문 | `list[float]` | `GoogleGenerativeAIEmbeddings.embed_query` |
| B-2 | `search_chunks` | 질문 벡터, k | `list[(Document, 코사인 거리)]` | `PGVector.similarity_search_with_score_by_vector` |
| B-3 | `build_prompt` | 지시문, 청크, 질문 | `list[BaseMessage]` | `ChatPromptTemplate` |
| B-4 | `generate_answer` | 메시지 | `AIMessage` (본문 + 토큰) | `ChatGoogleGenerativeAI` 또는 `ChatOllama` |
| 체인 | `build_rag_chain` | 질문 | 답변 문자열 | `retriever \| prompt \| llm \| StrOutputParser` |

### 메타데이터 (출처 연결)

각 청크에는 아래 metadata가 붙어 벡터 · 원문과 같은 행에 저장된다. 검색 결과와 프롬프트의 `[번호] (파일명 p.페이지)` 표기는 이 값을 쓴다.

```json
{"source": "보고서.pdf", "page": 3, "total_pages": 12, "start_index": 235, "chunk_index": 7}
```

- `page`: PyPDFLoader의 0부터 시작하는 값을 사람이 읽는 1부터로 변환
- `source`: 업로드를 임시 파일로 읽으므로 임시 경로 대신 원래 파일명으로 교체
- `start_index`: 페이지 안에서 청크가 시작하는 글자 위치 (`add_start_index=True`)

### 같은 문서 판단 (인덱싱 건너뛰기)

```mermaid
flowchart TD
    U["PDF 업로드 · 인덱싱 실행"] --> K["index_key = sha256(파일 내용 해시 + 저장 결과를 바꾸는 설정)"]
    K --> F{"강제 재인덱싱 체크?"}
    F -- 예 --> RUN["A-1 ~ A-4 실행"]
    F -- 아니오 --> Q{"컬렉션 metadata의 index_key 같음?<br/>이미지 실패 0개?<br/>저장된 청크 수 = 기록된 청크 수?"}
    Q -- 모두 예 --> SKIP["건너뜀 · API 호출 없음<br/>A-4 DB 내용만 표시"]
    Q -- 하나라도 아니오 --> RUN
    RUN --> SAVE["청크 저장 완료 후<br/>컬렉션 metadata에 index_key 기록"]
```

- **키에 들어가는 설정:** 청크 크기 · 겹침, 임베딩 모델 · 차원, 이미지 읽기 on/off · 프롬프트 · 모델. 파일명은 넣지 않습니다. 이름만 바뀐 같은 파일은 건너뛰고, 이름이 같아도 내용이 바뀌면 다시 인덱싱합니다.
- **컬렉션 metadata 예:**

```json
{"index_key": "9c1f…", "file_sha256": "2037…", "file_name": "보고서.pdf", "chunks": 253,
 "image_failures": 0, "indexed_at": "2026-10-27T15:39:30", "settings": {"chunk_size": 1000, "...": "..."}}
```

## 4. 런타임 시퀀스

```mermaid
sequenceDiagram
    actor U as 사용자
    participant App as app.py
    participant Emb as Gemini 임베딩
    participant DB as PostgreSQL (pgvector)
    participant LLM as Gemini LLM

    U->>App: 질문 입력
    App->>Emb: embed_query(질문) · RETRIEVAL_QUERY
    Emb-->>App: 질문 벡터 (B-1 표시)
    App->>DB: 코사인 거리 상위 k개 검색
    DB-->>App: 청크 원문 + metadata + 거리 (B-2 표시)
    App->>App: 지시문 + 청크 + 원래 질문으로 프롬프트 구성 (B-3 표시)
    App->>LLM: invoke(messages)
    LLM-->>App: 답변 + 토큰 사용량 (B-4 표시)
```

## 5. 데이터 모델

`langchain-postgres`가 관리하는 테이블 2개. `web_chat.py`의 `chat_threads` · `chat_messages` · `checkpoints*`와 같은 DB에 있지만 서로 참조하지 않는다.

```mermaid
erDiagram
    langchain_pg_collection ||--o{ langchain_pg_embedding : "컬렉션별 청크"
    langchain_pg_collection {
        uuid uuid PK
        varchar name "컬렉션 이름 (사이드바)"
        json cmetadata "마지막 인덱싱 정보 (index_key, file_sha256, chunks ...)"
    }
    langchain_pg_embedding {
        varchar id PK
        uuid collection_id FK
        vector embedding "임베딩 벡터"
        varchar document "청크 원문"
        jsonb cmetadata "출처 (source, page, chunk_index ...)"
    }
```

## 6. 주요 설계 결정

| 결정 | 이유 |
| --- | --- |
| 임베딩 저장에 `add_documents` 대신 `add_embeddings` 사용 | `add_documents`는 내부에서 다시 임베딩한다. A-3에서 보여준 벡터를 그대로 저장해 API 호출을 절반으로 줄이고, 화면에 보인 벡터와 DB 벡터가 같음을 보장 |
| 이미지는 PyPDFLoader의 `extract_images` 대신 pypdf `page.images`로 직접 추출 | PyPDFLoader는 JPEG · 일부 무손실 형식만 꺼내고, 흑백 스캔에 흔한 `CCITTFaxDecode` · `JBIG2Decode`는 경고만 남기고 건너뛴다. `page.images`는 형식 해석을 Pillow에 맡겨 흑백 스캔도 처리하고, 이미지를 모아 병렬로 Gemini에 보낼 수 있다 |
| LangChain `LLMImageBlobParser` 대신 자체 `GeminiImageReader` | 기본 파서는 응답 content가 문자열이라고 가정(`assert`)하는데 Gemini 3 계열은 블록 리스트로 줄 수 있다. `msg.text`로 읽고, 결과에 `[이미지 설명]` 표시를 붙여 본문과 구분 |
| 이미지 하나의 실패로 인덱싱을 멈추지 않음 | 실제로 일시적인 504가 발생해 PDF 전체가 실패한 적이 있음. 1회 재시도 후에도 실패하면 그 이미지만 건너뛰고 화면에 경고. 빈 응답도 일시적으로 발생한 적이 있어 같은 방식으로 재시도 · 경고 |
| 단색 이미지는 Gemini에 보내지 않고 경고 | 특수 형식이 잘못 디코딩되면 까만 이미지가 나와 Gemini가 빈 응답을 주고, 오류 없이 내용만 사라진다. 이를 화면에 드러내기 위함 |
| 답변 모델은 Gemini / Ollama Gemma 선택, 이미지 읽기는 항상 Gemini | Gemma(`gemma4:e2b`)는 로컬 · 무료로 답변 비교용. e2b는 작은 모델이라 스캔 한글 받아 적기 정확도를 기대하기 어렵고, 답변 모델 선택에 따라 인덱싱 결과가 달라지지 않도록 이미지 읽기는 Gemini로 고정 |
| OCR은 기존 `GeminiImageReader`를 OCR 프롬프트로 재사용, 결과는 파일 캐시 | 스캔 페이지 = 이미지 1장이라 이미지 읽기와 같은 경로. `[이미지 설명]` 표시만 생략(`mark=None`). 20쪽 OCR에 약 3분이 걸려 파일 · 모델 · 프롬프트별로 캐시하고, 실패한 쪽이 있으면 저장하지 않음 |
| 평가 세트에 사용자 말투 질문을 함께 둠 | LLM이 만든 질문은 문서 문장을 그대로 따라 써서 검색이 실제보다 쉬움(원문 1.00 vs 말투 0.90). 같은 정답으로 두 세트를 평가 |
| 채점은 LLM, 검색은 정답 쪽 포함 여부 · 순위로 따로 측정 | 답변 실패가 검색 때문인지 생성 때문인지 구분하기 위함 |
| 쪽을 넘어 이어서 청크 나누기 (옵션, 기본 꺼짐) | 쪽 경계에서 끊긴 문장(욱/리하 등)을 한 청크로. 앞 쪽이 문장 끝으로 끝나지 않으면 띄어 쓰지 않고 붙임. 효과는 미검증이라 기본값은 유지 |
| 청크마다 따로 임베딩 요청 + 스레드 동시 요청 (`EMBEDDING_CONCURRENCY`, 기본 8) | Vertex AI의 `gemini-embedding-2`는 요청 한 번에 텍스트 하나만 받는다(배치 시 "only supports one content at a time" 오류). 순서를 유지한 채 동시에 보내 인덱싱 시간을 줄임 |
| 문서용 · 질문용 임베딩 분리 (`embed_documents` / `embed_query`) | Gemini 임베딩은 `task_type`(RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY)에 따라 검색에 최적화된 벡터를 만든다. LangChain이 메서드별로 자동 지정 |
| 지시문을 템플릿 문자열이 아니라 변수(`{instructions}`)로 주입 | 사용자가 고친 지시문에 `{ }`가 들어가도 템플릿 변수로 해석되지 않음 |
| 같은 문서 판단은 파일 내용 해시 + 설정 (컬렉션 metadata에 저장) | 파일명으로는 이름만 바뀐 같은 파일이나 내용이 바뀐 같은 이름 파일을 구분할 수 없음. 설정이 다르면 청크 · 벡터가 달라지므로 키에 포함. 키는 청크 저장이 끝난 뒤 기록해, 저장 도중 실패하면 다음에 잘못 건너뛰지 않음. 이미지 변환 실패 이력이 있거나 청크 수가 맞지 않으면 건너뛰지 않음 |
| 재인덱싱 시 컬렉션 삭제 후 재생성 | 같은 문서를 다시 넣거나 청크 설정을 바꿨을 때 이전 청크가 섞이지 않게 함 (FR-7) |
| 도커 이미지 `pgvector/pgvector:pg17-trixie` | 기존 `postgres:17`(Debian trixie, glibc 2.41)과 같은 OS 기반이라 기존 `pgdata` 볼륨을 collation 불일치 없이 그대로 사용. bookworm 기반 `pg17` 태그는 glibc 2.36이라 collation 버전 경고 발생 |
| 임베딩 · LLM 모두 `global` 엔드포인트 | `gemini-embedding-2`는 `global`에서만 제공된다 (`us-central1`은 404 확인). LLM도 `web_chat.py`와 같은 설정. 각각 `RAG_EMBEDDING_LOCATION` · `RAG_LLM_LOCATION`으로 변경 |
| `temperature` 대신 `reasoning_effort="low"` | Gemini 3 계열은 temperature를 낮추면 반복 · 품질 저하가 생길 수 있어 기본값 유지 권장 |
| 단계 함수와 LCEL 체인을 둘 다 제공 | 단계 함수는 중간값 확인용, 체인은 실제 서비스에 쓰는 형태. 화면에서 두 결과를 비교 |

## 7. 환경변수

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://langchain:langchain@localhost:5433/langchain_chat` | `web_chat.py`와 공유 |
| `GOOGLE_CLOUD_PROJECT` | ADC 기본 프로젝트 | Vertex AI 프로젝트 |
| `RAG_EMBEDDING_MODEL` | `gemini-embedding-2` | 임베딩 모델 |
| `RAG_EMBEDDING_LOCATION` | `global` | 임베딩 엔드포인트 리전 (`gemini-embedding-2`는 `global` 전용) |
| `RAG_EMBEDDING_CONCURRENCY` | `8` | 임베딩 동시 요청 수 (할당량 초과 시 낮춤) |
| `RAG_IMAGE_CONCURRENCY` | `4` | 이미지 변환 동시 요청 수 |
| `RAG_EVAL_CONCURRENCY` | `2` | 평가 동시 실행 문항 수 (문항마다 답변 + 채점 2회 호출) |
| `RAG_LLM_MODEL` | `GEMINI_MODEL` 또는 `gemini-3.8-flash` | 답변 모델 |
| `RAG_LLM_LOCATION` | `GOOGLE_CLOUD_LOCATION` 또는 `global` | 답변 모델 엔드포인트 |
| `RAG_COLLECTION_NAME` | `rag_pdf` | 기본 컬렉션 이름 |
