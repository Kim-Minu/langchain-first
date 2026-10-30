# arXiv 논문 검색 · 저장 · 질의 — 태스크

### Phase 1: 설정 · 의존성
- [x] `arxiv` 의존성, `pytest` dev 의존성 추가 (+ `[tool.pytest.ini_options]` testpaths · pythonpath)
- [x] `config.py`에 `ARXIV_COLLECTION` · `ARXIV_DATA_DIR` · `ARXIV_MAX_RESULTS`
- [x] `.gitignore`에 `rag_app/data/`

### Phase 2: 서비스 (`rag_app/arxiv_service.py`)
- [x] `Paper` · `search_papers` (R1)
- [x] `download_pdf` — 있으면 재사용, 임시 `.part` → 이름 변경 (R2)
- [x] `index_paper` — advisory lock, 재확인, 새 세대 저장 → 기록 교체 → 옛 세대 삭제 (R3 · R6)
- [x] `list_indexed_papers`
- [x] `ask` — `arxiv_id` 필터 검색 → 답변 (R4)

### Phase 3: 화면 (`rag_app/pages/arxiv_qa.py`)
- [x] 검색 → 체크 → 다운로드 · 저장 (논문별 성공 · 건너뜀 · 실패 표시)
- [x] 저장된 논문 선택 → 질문 → 답변 + 근거 (R5)
- [x] 사이드바 링크 (`ui_common.render_left_menu`)
- [x] PDF 미리보기 — 검색 결과 · 저장된 논문, 받지 않은 논문은 다운로드 후 `st.pdf` (R7)

### Phase 3-1: 코드 리뷰 반영
- [x] MAJOR-1 새 검색 시 이전 체크가 남음 → 검색마다 data_editor key 변경
- [x] MAJOR-2 재저장 실패 시 기존 버전 유실 → 청크에 `index_key`(세대), 새 세대 저장 → 기록 교체 → 옛 세대 삭제, 질의는 기록된 세대만
- [x] MAJOR-3 임베딩 모델 혼재 → 컬렉션 이름에 모델 · 차원
- [x] 확인 필요: 기존 PDF RAG 화면에서 논문 컬렉션을 선택 · 삭제 목록에서 제외, 저장 거부
- [x] 확인 필요: `save_to_store`에서 논문 컬렉션 거부 (OCR 평가 화면 포함 모든 경로), 판별을 이름 접두사 → metadata 논문 기록으로
- [x] 재리뷰 M1 config f-string 백슬래시 → Python 3.11 문법 오류, 슬러그를 밖에서 계산
- [x] 재리뷰 m5 미리보기가 고른 논문과 다른 PDF를 보여 줌 → 논문 ID와 함께 저장, 일치할 때만 표시
- [x] 재리뷰 m6 저장된 논문의 PDF가 없으면 미리보기 불가 (R7 위반) → 기록에 `pdf_url`, `paper_from_record`로 다시 받기

### Phase 4: 테스트 (`tests/test_arxiv_service.py`)
- [x] 정상: 검색 결과를 `Paper`로 변환 → `test_search_papers_maps_results`
- [x] 정상: PDF 다운로드 후 저장 → `test_download_pdf_saves_file`
- [x] 정상: 저장하면 청크 · 목록이 생기고 `arxiv_id`가 붙음 → `test_index_paper_stores_chunks_and_registry`
- [x] 정상: 두 논문 저장 후 필터 질의는 고른 논문 청크만 근거로 씀 → `test_ask_filters_by_arxiv_id`
- [x] 경계: 이미 받은 PDF는 다시 받지 않음 → `test_download_pdf_reuses_existing_file`
- [x] 경계: 옛 형식 ID(`hep-th/9901001v1`)의 `/`를 파일명에서 치환 → `test_pdf_path_for_old_style_id`
- [x] 경계: 같은 논문 재저장은 건너뜀 → `test_index_paper_skips_same_index`
- [x] 경계: 검색 결과 0건 · 빈 검색어 → `test_search_papers_empty`
- [x] 실패: PDF가 아닌 응답이면 오류, 파일이 남지 않음 → `test_download_pdf_rejects_non_pdf`
- [x] 실패: pdf_url 없음 → `test_download_pdf_without_url`
- [x] 실패/롤백: 저장 도중 실패 후 재시도하면 정상 저장, 다른 논문은 그대로 → `test_index_paper_recovers_after_failure`
- [x] 동시성: 같은 논문을 두 스레드가 동시에 저장해도 청크 중복 없음 → `test_index_paper_concurrent_same_paper` (잠금을 빼면 실패함을 확인)
- [x] 실패/롤백: 이미 저장된 논문의 재저장이 도중 실패해도 기존 버전 기록 · 질의 유지, 재시도로 교체 → `test_index_paper_reindex_failure_keeps_old_version`
- [x] 동시성: 서로 다른 논문 기록을 동시에 병합해도 유실 없음 → `test_record_paper_concurrent_different_papers`
- [x] 경계: 저장된 논문이 없으면 LLM을 부르지 않고 안내 → `test_ask_without_papers`
- [x] 정상: 컬렉션 이름에 임베딩 모델 · 차원, arXiv 컬렉션 판별 → `test_arxiv_collection_is_separated_by_embedding`
- [x] 정상: 화면에서 검색 → 미리보기 (받지 않은 논문 다운로드 후 PDF 뷰어) → `test_page_search_and_preview` (R5 · R7)
- [x] 정상: 저장 기록 → Paper, PDF 주소 보충 → `test_paper_from_record_fills_pdf_url` (R7)
- [x] 실패: 논문 컬렉션에 `save_to_store`하면 거부, 청크 그대로 · 이름만 비슷한 컬렉션은 해당 없음 → `test_save_to_store_refuses_paper_collection`
- [x] 정상: 보호 장치 후에도 일반 컬렉션은 기존처럼 교체 저장 → `test_save_to_store_still_works_for_normal_collection`
