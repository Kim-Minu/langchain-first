# arXiv 논문 검색 · 저장 · 질의 — 태스크

### Phase 1: 설정 · 의존성
- [ ] `arxiv` 의존성, `pytest` dev 의존성 추가
- [ ] `config.py`에 `ARXIV_COLLECTION` · `ARXIV_DATA_DIR` · `ARXIV_MAX_RESULTS`
- [ ] `.gitignore`에 `rag_app/data/`

### Phase 2: 서비스 (`rag_app/arxiv_service.py`)
- [ ] `Paper` · `search_papers` (R1)
- [ ] `download_pdf` — 있으면 재사용, `.part` → 이름 변경 (R2)
- [ ] `index_paper` — advisory lock, 재확인, 삭제 → 저장 → 목록 기록 (R3 · R6)
- [ ] `list_indexed_papers`
- [ ] `ask` — `arxiv_id` 필터 검색 → 답변 (R4)

### Phase 3: 화면 (`rag_app/pages/arxiv_qa.py`)
- [ ] 검색 → 체크 → 다운로드 · 저장 (논문별 성공 · 건너뜀 · 실패 표시)
- [ ] 저장된 논문 선택 → 질문 → 답변 + 근거 (R5)

### Phase 4: 테스트 (`tests/test_arxiv_service.py`)
- [ ] 정상: 검색 결과를 `Paper`로 변환 → `test_search_papers_maps_results`
- [ ] 정상: PDF 다운로드 후 저장 → `test_download_pdf_saves_file`
- [ ] 정상: 저장하면 청크 · 목록이 생기고 `arxiv_id`가 붙음 → `test_index_paper_stores_chunks_and_registry`
- [ ] 정상: 두 논문 저장 후 필터 질의는 고른 논문 청크만 근거로 씀 → `test_ask_filters_by_arxiv_id`
- [ ] 경계: 이미 받은 PDF는 다시 받지 않음 → `test_download_pdf_reuses_existing_file`
- [ ] 경계: 같은 논문 재저장은 건너뜀 → `test_index_paper_skips_same_index`
- [ ] 경계: 검색 결과 0건 → `test_search_papers_empty`
- [ ] 실패: PDF가 아닌 응답이면 오류, 파일이 남지 않음 → `test_download_pdf_rejects_non_pdf`
- [ ] 실패: pdf_url 없음 → `test_download_pdf_without_url`
- [ ] 실패/롤백: 저장 도중 실패 후 재시도하면 정상 저장, 다른 논문은 그대로 → `test_index_paper_recovers_after_failure`
- [ ] 동시성: 같은 논문을 두 스레드가 동시에 저장해도 청크 중복 없음 → `test_index_paper_concurrent_same_paper`
