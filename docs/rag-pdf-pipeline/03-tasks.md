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
- [x] A-1 경고: 글자 없는 페이지, 변환 실패 · 단색 이미지

## 검증

- [x] 가짜 임베딩 · LLM으로 실제 pgvector에 대해 전체 흐름 스모크 테스트
  - 비 PDF 거부 / 페이지 · 청크 metadata / 재인덱싱 시 중복 없음 / 같은 문장 검색 시 거리 0 / 지시문 `{ }` 처리 / 체인 실행
- [x] Streamlit `AppTest`로 첫 화면 렌더링 · 겹침 ≥ 크기 검증 메시지 확인
- [x] `ruff check` (기존 코드와 같은 `BLE001` 외 경고 없음)
- [x] Vertex AI 실제 호출로 임베딩 · 답변 확인 — `gemini-embedding-2`(global, 768차원) 검색 → Gemini 답변 · LCEL 체인 결과 일치
- [x] `gemini-embedding-2` 배치 미지원 대응 (청크별 요청 + 동시 요청), `us-central1` 미제공 → 임베딩 리전 `global`

- [x] 스캔 PDF 실제 Gemini 변환: JPEG(DCT) · 흑백 1비트(CCITTFax) 정상, 텍스트 전용 PDF는 호출 없음, 40px 아이콘 건너뜀, 팔레트 형식이 단색으로 풀리는 경우 경고 표시

## 다음 후보 (범위 밖)

- [ ] 여러 PDF를 한 컬렉션에 누적 저장 (현재는 재인덱싱 시 교체)
- [ ] MMR 검색(`max_marginal_relevance_search`)으로 중복 청크 줄이기 옵션
- [ ] 답변 스트리밍
- [ ] 페이지 전체를 이미지로 렌더링하는 폴백 (pypdfium2) — 이미지 형식 해석 실패 · 벡터 도형으로 그린 차트까지 처리
