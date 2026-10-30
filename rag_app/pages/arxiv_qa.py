"""
[arXiv 논문 질의 - arXiv에서 논문을 찾아 받고, pgvector에 저장한 뒤 논문 내용을 질문]

1. 검색: 키워드 → arXiv 논문 목록 (제목 · 저자 · 게시일 · 요약)
2. 저장: 고른 논문 PDF 다운로드 → 청크 → 임베딩 → 공용 컬렉션(config.ARXIV_COLLECTION)에 누적
3. 저장된 논문: 목록 · PDF 미리보기 · 삭제 (검색 결과는 표의 PDF 링크로 새 탭에서)
4. 질의: 저장된 논문 전체 또는 고른 논문만 검색해 답변 + 근거 청크

로직은 rag_app/arxiv_service.py, 이 파일은 화면만 담당한다.
실행: uv run streamlit run rag_app/app.py → 사이드바 "📚 arXiv 논문 질의" 링크
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from rag_app import arxiv_service as ax
from rag_app import config
from rag_app import pipeline as rag
from rag_app.ui_common import get_embeddings, get_llm, preview, render_left_menu, show_error, step_header

PDF_KIND_LABEL = {"text": "텍스트 PDF", "image": "이미지(스캔) PDF", "mixed": "혼합 PDF"}

st.set_page_config(page_title="arXiv 논문 질의", page_icon="📚", layout="wide")
render_left_menu()

st.title("📚 arXiv 논문 질의")
st.caption(
    f"답변 `{config.LLM_MODEL}` · 임베딩 `{config.EMBEDDING_MODEL}` {config.EMBEDDING_DIMENSION}차원 · "
    f"청크 {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP} · 컬렉션 `{config.ARXIV_COLLECTION}`"
)

embeddings = get_embeddings(config.EMBEDDING_MODEL, config.EMBEDDING_DIMENSION)

# ---------------------------------------------------------------------------
# 1. 검색
# ---------------------------------------------------------------------------
step_header("1", "논문 검색", "arXiv API로 검색합니다 (요청 사이 3초 간격 · 실패 시 3회 재시도).")
with st.form("arxiv_search"):
    col_query, col_count = st.columns([5, 1])
    query = col_query.text_input("검색어", placeholder="예: retrieval augmented generation")
    max_results = col_count.number_input("결과 수", 1, 50, config.ARXIV_MAX_RESULTS)
    searched = st.form_submit_button("검색", type="primary")

if searched:
    try:
        with st.spinner("arXiv 검색 중..."):
            st.session_state["arxiv_results"] = ax.search_papers(query, int(max_results))
        # data_editor는 체크 상태를 행 번호로 기억하므로 검색마다 key를 바꿔 이전 체크가 새 목록에 남지 않게 함
        st.session_state["arxiv_search_no"] = st.session_state.get("arxiv_search_no", 0) + 1
    except Exception as e:
        show_error("arXiv 검색", e)

papers: list[ax.Paper] = st.session_state.get("arxiv_results", [])
indexed_ids = {p["arxiv_id"] for p in ax.list_indexed_papers()}

# ---------------------------------------------------------------------------
# 2. 저장
# ---------------------------------------------------------------------------
if papers:
    step_header("2", "다운로드 · 저장", "저장할 논문을 체크하세요. 이미 같은 내용으로 저장된 논문은 건너뜁니다.")
    table = pd.DataFrame([
        {
            "선택": False,
            "저장됨": "✅" if p.arxiv_id in indexed_ids else "",
            "arXiv ID": p.arxiv_id,
            "제목": p.title,
            "저자": ", ".join(p.authors[:3]) + (" 외" if len(p.authors) > 3 else ""),
            "게시일": p.published,
            "요약": preview(p.summary, 200),
            "PDF": p.pdf_url,  # 저장하지 않고 원문을 새 탭에서 바로 확인
        }
        for p in papers
    ])
    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=[c for c in table.columns if c != "선택"],
        column_config={"PDF": st.column_config.LinkColumn("PDF", display_text="열기 ↗", help="arXiv PDF를 새 탭에서 엽니다")},
        key=f"arxiv_pick_{st.session_state.get('arxiv_search_no', 0)}",
    )
    picked = [p for p, chosen in zip(papers, edited["선택"]) if chosen]

    if st.button(f"선택한 논문 {len(picked)}편 다운로드 · 저장", disabled=not picked):
        # 논문마다 따로 처리해 하나가 실패해도 나머지는 계속 (R6)
        for paper in picked:
            with st.status(f"{paper.arxiv_id} · {paper.title}", expanded=False) as status:
                try:
                    st.write("PDF 다운로드")
                    pdf_path = ax.download_pdf(paper)
                    st.write("페이지 분석 (스캔 페이지는 OCR) → 청크 → 임베딩 → 저장")
                    result = ax.index_paper(
                        paper, pdf_path, embeddings, ocr_llm=get_llm(config.ARXIV_OCR_MODEL, config.IMAGE_MAX_OUTPUT_TOKENS),
                    )
                    kind = PDF_KIND_LABEL[result.pdf_kind]
                    if result.ocr_pages:
                        kind += f" · OCR {len(result.ocr_pages)}쪽"
                    label = "이미 저장됨 · 건너뜀" if result.skipped else f"저장 완료 · 청크 {result.chunks}개"
                    if result.ocr_failures:
                        st.warning(f"OCR 실패 {len(result.ocr_failures)}건 — 다시 저장하면 실패한 페이지만 다시 OCR합니다.")
                        st.caption("\n".join(result.ocr_failures[:5]))
                    if result.ocr_warnings:  # 디코딩 실패 · 단색 이미지 등은 다시 해도 같으므로 안내만
                        st.info(f"읽지 못한 이미지 {len(result.ocr_warnings)}건 (다시 저장해도 같음)")
                        st.caption("\n".join(result.ocr_warnings[:5]))
                    status.update(label=f"{paper.arxiv_id} — {label} · {kind}", state="complete")
                except Exception as e:
                    status.update(label=f"{paper.arxiv_id} — 실패", state="error", expanded=True)
                    show_error(f"{paper.arxiv_id} 저장", e)

stored = ax.list_indexed_papers()

# ---------------------------------------------------------------------------
# 3. 저장된 논문 (미리보기 · 삭제)
# ---------------------------------------------------------------------------
# 검색 결과는 표의 PDF 링크로 새 탭에서, 저장된 논문은 여기서 화면 안에 미리봄
step_header(
    "3", "저장된 논문",
    f"{len(stored)}편. 삭제하면 청크와 기록이 지워지고, 받아 둔 PDF는 남아 다시 저장할 때 내려받지 않습니다.",
)
if not stored:
    st.info("저장된 논문이 없습니다. 위에서 검색해 저장하세요.")
    st.stop()

for p in stored:
    col_info, col_view, col_del = st.columns([6, 1, 1])
    kind = PDF_KIND_LABEL.get(p.get("pdf_kind", "text"), "")
    col_info.markdown(f"**{p['arxiv_id']}** · {p['title']}  \n청크 {p['chunks']}개 · {kind} · {p.get('indexed_at', '')}")
    if col_view.button("📄 미리보기", key=f"arxiv_view_{p['arxiv_id']}", width="stretch"):
        try:
            with st.spinner("PDF 준비 중..."):
                path = ax.download_pdf(ax.paper_from_record(p))  # 받아 둔 파일이 있으면 그대로, 없으면 다시 받음
            # 다른 위젯을 눌러도 유지되게 저장하되, 어떤 논문의 파일인지 함께 기록
            st.session_state["arxiv_preview"] = {"arxiv_id": p["arxiv_id"], "path": str(path)}
        except Exception as e:
            show_error(f"{p['arxiv_id']} 미리보기", e)
    if col_del.button("🗑️ 삭제", key=f"arxiv_del_{p['arxiv_id']}", width="stretch"):
        try:
            ax.delete_paper(p["arxiv_id"])
        except Exception as e:
            show_error(f"{p['arxiv_id']} 삭제", e)
        else:
            if st.session_state.get("arxiv_preview", {}).get("arxiv_id") == p["arxiv_id"]:
                st.session_state.pop("arxiv_preview")  # 지운 논문의 미리보기도 닫음
            st.rerun()

# 변수 이름을 shown으로 — preview는 ui_common에서 가져온 요약 함수 이름
shown = st.session_state.get("arxiv_preview")
stored_titles = {p["arxiv_id"]: p["title"] for p in stored}
if shown and shown["arxiv_id"] in stored_titles and Path(shown["path"]).exists():
    col_title, col_close = st.columns([6, 2])
    col_title.markdown(f"**미리보기** · {shown['arxiv_id']} · {stored_titles[shown['arxiv_id']]}")
    if col_close.button("✕ 미리보기 닫기", key="arxiv_preview_close", width="stretch"):
        st.session_state.pop("arxiv_preview", None)
        st.rerun()
    st.pdf(shown["path"], height=800)

# ---------------------------------------------------------------------------
# 4. 질의
# ---------------------------------------------------------------------------
step_header(
    "4", "논문 질의",
    "대상을 고르지 않으면 전체에서 찾습니다. "
    f"의미 검색 · BM25 키워드 검색 각 {config.ARXIV_CANDIDATES}개를 합쳐 상위 {config.ARXIV_TOP_K}개를 근거로 씁니다.",
)

titles = {p["arxiv_id"]: f"{p['arxiv_id']} · {p['title']}" for p in stored}
targets = st.multiselect("대상 논문", list(titles), format_func=titles.get)
question = st.text_area("질문", placeholder="예: 이 논문이 제안한 방법의 핵심 아이디어는?")

if st.button("질문하기", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("검색 · 답변 생성 중..."):
            answer, hits = ax.ask(
                question, get_llm(config.LLM_MODEL, config.MAX_OUTPUT_TOKENS), embeddings, arxiv_ids=targets or None,
            )
    except Exception as e:
        show_error("질의", e)
    else:
        st.markdown(answer)
        for i, (doc, ranks) in enumerate(hits, start=1):
            # 어느 검색에서 몇 위로 찾았는지 (— 는 그 검색의 후보에 없었음)
            found = " · ".join(
                f"{name} {ranks[key]}위" if ranks[key] else f"{name} —"
                for name, key in (("의미", "vector_rank"), ("BM25", "bm25_rank"))
            )
            with st.expander(f"[{i}] {doc.metadata.get('source', '?')} p.{rag.page_label(doc)} · {found}"):
                st.text(doc.page_content)
