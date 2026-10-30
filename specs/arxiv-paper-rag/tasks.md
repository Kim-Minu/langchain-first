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

## 2차: R8 ~ R12

### Phase 5: 설정 · 서비스
- [x] `rank-bm25` 의존성, config 상수 (`ARXIV_TOP_K` · `ARXIV_CANDIDATES` · `ARXIV_HYBRID_WEIGHTS` · `ARXIV_OCR_MIN_CHARS` · `ARXIV_OCR_PROMPT` · `ARXIV_KEYWORD_PROMPT`)
- [x] `analyze_pdf` · 페이지별 OCR 로딩 (캐시 · 실패 기록) → `index_paper(ocr_llm=)` (R8)
- [x] BM25 말뭉치 캐시 · 키워드 변환 · 하이브리드 `ask` (top-k 5) (R9 · R10)
- [x] `delete_paper` (R11)
- [x] `migrate_legacy_collection` + 실제 `arxiv_papers` 이전 실행 (R12) — 2편 · 139청크 이전, 재저장 시 건너뜀 확인

### Phase 5-1: 2차 리뷰 반영
- [x] M1 저장마다 `generation` id — 잔여 청크 정리 · 필터 · 청크 수 · BM25 캐시 키를 세대 기준으로
- [x] M2 LLM OCR 실패(`ocr_failures`, 재시도)와 추출 경고(`ocr_warnings`, 표시만) 분리
- [x] 확인 필요: 실제 컬렉션 `backfill_generation` 실행 — 2편 · 117 / 22청크, 재저장 시 건너뜀 확인
- [x] 확인 필요: `ARXIV_OCR_MODEL` · OCR 출력 한도 `IMAGE_MAX_OUTPUT_TOKENS`
- [x] Minor(선택): 키워드 변환 실패 시 원문 질문으로 BM25
- [x] 재리뷰 확인 필요 1: `delete_paper`가 세대 없는 옛 청크까지 삭제 (R11)
- [x] 검색 결과 표에 PDF 링크 열 (`LinkColumn`, 새 탭) (R1 추가 요청)
- [x] 미리보기 메뉴를 저장된 논문 목록과 합침 — 표에서 행 선택 → 미리보기 · 삭제, 지운 논문의 미리보기 닫기 (3차)
- [x] 미리보기 닫기 버튼 (R7 추가 요청) · 화면 변수 `preview`가 요약 함수 이름을 가리던 것 정리

### Phase 6: 화면
- [x] 저장 결과에 PDF 유형 · OCR 쪽수, 근거에 검색 경로(의미 · BM25 순위) 표시
- [x] 저장된 논문 목록 · 삭제 버튼

### Phase 7: 테스트
- [x] 정상: 텍스트 PDF는 OCR 호출 없이 `text` → `test_analyze_text_pdf_skips_ocr`
- [x] 정상: 스캔 PDF는 모든 페이지 OCR, `image` → `test_index_scanned_pdf_runs_ocr`
- [x] 경계: 혼합 PDF는 이미지 페이지만 OCR, `mixed` → `test_index_mixed_pdf_ocr_only_image_pages`
- [x] 실패/롤백: OCR 실패 페이지가 있으면 기록에 남기고 다음 저장에서 다시 OCR → `test_ocr_failure_is_retried_next_time`
- [x] 실패: OCR이 필요한데 LLM이 없으면 오류 → `test_index_scanned_pdf_requires_ocr_llm`
- [x] 정상: 의미 검색이 놓친 정확한 키워드를 BM25가 찾아 상위 5에 포함 → `test_hybrid_finds_keyword_missed_by_vector`
- [x] 경계: 기본 top-k 5 → `test_ask_returns_top_5`
- [x] 정상: 한글 질문은 영어 키워드로 BM25, 영어 질문은 변환 호출 없음 → `test_bm25_query_translation`
- [x] 경계: 재저장으로 세대가 바뀌면 BM25 캐시도 새 청크로 → `test_bm25_cache_follows_generation`
- [x] 정상: 논문 삭제 후 목록 · 질의 · 청크에서 사라지고 다른 논문은 그대로 → `test_delete_paper`
- [x] 정상: 기존 컬렉션 이전 후 재임베딩 없이 "이미 저장됨" · 질의 가능 → `test_migrate_legacy_collection`
- [x] 실패/롤백: 새 이름이 이미 있으면 이전하지 않고 원본 그대로 → `test_migrate_refuses_existing_target`
- [x] 실패/롤백: OCR 재시도 저장이 도중 실패해도 지금 검색되는 세대 유지 → `test_ocr_retry_failure_keeps_live_generation`
- [x] 정상: OCR 재시도 후 BM25 캐시가 새 세대 청크로 → `test_ocr_retry_refreshes_bm25_cache`
- [x] 경계: 단색 이미지 등 추출 경고는 건너뛰기를 막지 않음 → `test_extraction_warning_does_not_block_skip`
- [x] 실패: 키워드 변환 실패 시 원문으로 BM25 → `test_keyword_query_falls_back_on_error`
- [x] 정상: 세대 없는 기록 · 청크에 세대 채우기, 다시 실행해도 그대로 → `test_backfill_generation`
- [x] 동시성: 해당 없음 — 새 경로는 기존 논문 잠금(`_paper_lock`)을 그대로 쓰며 동시 저장 테스트가 이미 있음. 삭제도 같은 잠금 사용
- [x] 정상: 검색 결과 PDF 링크, 저장된 논문 표 · 행 선택 전 버튼 없음 · 선택 후 미리보기(없으면 다시 받음) · 닫기 · 삭제 시 미리보기 닫힘 → `test_page_search_and_preview` (R1 · R7 · R11)
- [x] 정상: 세대 없는 옛 청크도 논문 삭제 시 제거 → `test_delete_paper_removes_legacy_chunks` (R11)
