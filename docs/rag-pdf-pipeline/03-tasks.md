# PDF RAG 실습 앱 작업 목록

브랜치: `feature/rag-pdf-pipeline` · 기획 [01-plan.md](01-plan.md) · 설계 [02-design.md](02-design.md)

## 인프라 · 의존성

- [x] `docker-compose.yml` 이미지를 `pgvector/pgvector:pg17-trixie`로 교체 (기존 볼륨 · 대화 데이터 유지 확인)
- [x] 의존성 추가: `langchain-community`, `langchain-text-splitters`, `pypdf`, `langchain-postgres`

## 구현

- [x] `rag_app/config.py` — 기본값 · 입력 범위 · 환경변수
- [x] `rag_app/pipeline.py` — A-1 ~ A-4 인덱싱 함수
- [x] `rag_app/pipeline.py` — B-1 ~ B-4 런타임 함수, `build_rag_chain` (LCEL)
- [x] `rag_app/pipeline.py` — 저장 결과 확인용 `inspect_collection`
- [x] `rag_app/app.py` — 사이드바 설정 (청크 크기 · 겹침 · 임베딩 모델 · 차원 · 컬렉션 · top-k · 답변 모델 · 토큰 · 지시문)
- [x] `rag_app/app.py` — A 탭: PDF 업로드, 인덱싱 실행, 단계별 출력
- [x] `rag_app/app.py` — B 탭: 질문, 단계별 출력, LCEL 체인 비교
- [x] 오류 안내 (ADC 재인증 · DB 미실행 · pgvector 없음)
- [x] PDF 속 이미지 처리: `GeminiImageReader` + pypdf `page.images`, 사이드바 on/off, 병렬 요청 · 재시도
- [x] 같은 문서 건너뛰기: `file_sha256` · `make_index_key` · `find_same_index`, 컬렉션 metadata 기록, '같은 문서여도 다시 인덱싱' 체크박스, B 탭에 파일명 표시
- [x] 이미지 읽기 프롬프트 사이드바 입력 (비우면 기본값, 바꾸면 재인덱싱 경고), 빈 응답 재시도 · 경고
- [x] 답변 모델 선택: Gemini Flash / Ollama Gemma 4 (로컬), Ollama 오류 안내
- [x] A-1 경고: 글자 없는 페이지, 변환 실패 · 단색 이미지

## OCR · RAG 평가 (브랜치 `feature/rag-ocr-eval`)

- [x] 평가 PDF를 `rag_app/eval/data/`로 가져오기 (공개 저장소라 git 제외)
- [x] OCR 프롬프트(인쇄 텍스트만 · 사진/그림 제외) · `GeminiImageReader` 재사용 · OCR 파일 캐시
- [x] 질문 · 정답 20쌍: Gemini 생성 → 원문 대조 검수 · 3문항 교체 → 사용자 말투 질문 추가
- [x] Gemini 채점 · 정답 쪽 검색률 · MRR · 할당량(429) 재시도 · 병렬 평가
- [x] 평가 화면 4탭 · 사이드바 공통 모듈(`ui_common.py`)로 분리 · 두 화면 설정 공유
- [x] 쪽을 넘어 이어서 청크 나누기 옵션 (기본 꺼짐)
- [x] 임베딩 캐시: `CachedEmbeddings` + `embedding_cache` 테이블 (모델 · 차원 · task_type · 텍스트 해시), A-3 · 평가 인덱싱에 캐시 사용 수 표시
- [x] 기준선 평가: 원문 1.00 / 말투 0.90 → [04-evaluation.md](04-evaluation.md)
- [x] 평가 확장 (RAG 평가 관례 반영): 근거 충실도(faithfulness, 주장 단위 판정) · 답이 없는 질문 4개(거절 정확도) · 채점 모델 분리(`gemini-3.1-pro-preview`) · 사람 판정 입력과 LLM 채점 일치율
- [ ] 확장된 평가 세트(24문항)로 기준선 재측정 결과를 04-evaluation.md에 반영
- [ ] 개선안 효과 측정 (작은 청크 + k 증가, 쪽 연결 등) — 실험 중단, 화면에서 이어서 측정

## 검증

- [x] 가짜 임베딩 · LLM으로 실제 pgvector에 대해 전체 흐름 스모크 테스트
  - 비 PDF 거부 / 페이지 · 청크 metadata / 재인덱싱 시 중복 없음 / 같은 문장 검색 시 거리 0 / 지시문 `{ }` 처리 / 체인 실행
- [x] Streamlit `AppTest`로 첫 화면 렌더링 · 겹침 ≥ 크기 검증 메시지 확인
- [x] `ruff check` (기존 코드와 같은 `BLE001` 외 경고 없음)
- [x] Vertex AI 실제 호출로 임베딩 · 답변 확인 — `gemini-embedding-2`(global, 768차원) 검색 → Gemini 답변 · LCEL 체인 결과 일치
- [x] `gemini-embedding-2` 배치 미지원 대응 (청크별 요청 + 동시 요청), `us-central1` 미제공 → 임베딩 리전 `global`

- [x] 스캔 PDF 실제 Gemini 변환: JPEG(DCT) · 흑백 1비트(CCITTFax) 정상, 텍스트 전용 PDF는 호출 없음, 40px 아이콘 건너뜀, 팔레트 형식이 단색으로 풀리는 경우 경고 표시
- [x] Ollama `gemma4:e2b` 답변: Gemini 임베딩 검색 결과로 근거 번호 포함 답변 · LCEL 체인 동일, 화면에서 선택 시 오류 없음
- [x] 같은 문서 판단 7가지 경우 (실제 pgvector): 같은 문서 · 이름만 바뀐 파일 → 건너뜀 / 청크 설정 변경 · 내용 변경 · 이미지 실패 이력 · 청크 누락 · 키 없는 기존 컬렉션 → 재인덱싱
- [x] 임베딩 캐시: 배치 내 중복 1회 요청 · 재실행 전부 캐시 · 일부 변경 시 변경분만 요청 · 질문용/차원별 분리 (가짜 임베딩), 실제 Gemini 26청크 2.2초 → 0.0초 · 캐시 벡터와 새 벡터 차이 0 · 검색 거리 동일
- [x] 평가 화면 흐름 (AppTest): OCR(캐시) → 인덱싱(같은 설정 건너뜀) → 2문항 평가 · 진행 표시 · 실행 비교 기록

## 다음 후보 (범위 밖)

- [ ] 여러 PDF를 한 컬렉션에 누적 저장 (현재는 재인덱싱 시 교체) — 청크 metadata의 `file_sha256`으로 문서별 교체 · 삭제 가능
- [ ] MMR 검색(`max_marginal_relevance_search`)으로 중복 청크 줄이기 옵션
- [ ] 답변 스트리밍
- [ ] 질문 다시 쓰기(query rewriting) · 키워드 결합 검색(hybrid) · 소제목 붙인 청크 — [04-evaluation.md](04-evaluation.md) 개선안 3~5
- [ ] 페이지 전체를 이미지로 렌더링하는 폴백 (pypdfium2) — 이미지 형식 해석 실패 · 벡터 도형으로 그린 차트까지 처리
