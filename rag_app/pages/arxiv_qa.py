"""
[arXiv 논문 질의 - arXiv에서 논문을 찾아 받고, pgvector에 저장한 뒤 논문 내용을 질문]

1. 검색: 키워드 → arXiv 논문 목록 (제목 · 저자 · 게시일 · 요약)
2. 저장: 고른 논문 PDF 다운로드 → 청크 → 임베딩 → 공용 컬렉션(config.ARXIV_COLLECTION)에 누적
3. 미리보기: 검색 결과 · 저장된 논문의 PDF를 화면에서 확인 (받지 않은 논문은 먼저 다운로드)
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
        }
        for p in papers
    ])
    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=[c for c in table.columns if c != "선택"],
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
                    st.write("청크 → 임베딩 → 저장")
                    result = ax.index_paper(paper, pdf_path, embeddings)
                    label = "이미 저장됨 · 건너뜀" if result.skipped else f"저장 완료 · 청크 {result.chunks}개"
                    status.update(label=f"{paper.arxiv_id} — {label}", state="complete")
                except Exception as e:
                    status.update(label=f"{paper.arxiv_id} — 실패", state="error", expanded=True)
                    show_error(f"{paper.arxiv_id} 저장", e)

stored = ax.list_indexed_papers()

# ---------------------------------------------------------------------------
# 3. PDF 미리보기
# ---------------------------------------------------------------------------
# 검색 결과와 저장된 논문을 함께 후보로 (검색 결과 쪽 정보를 우선)
candidates = {p["arxiv_id"]: ax.paper_from_record(p) for p in stored}
candidates.update({p.arxiv_id: p for p in papers})
if candidates:
    step_header("3", "PDF 미리보기", "저장 전에 내용을 확인할 수 있습니다. 아직 받지 않은 논문은 먼저 다운로드합니다.")
    col_pick, col_btn = st.columns([5, 1])
    preview_id = col_pick.selectbox(
        "미리볼 논문", list(candidates), format_func=lambda i: f"{i} · {candidates[i].title}", key="arxiv_preview_pick",
    )
    col_btn.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
    if col_btn.button("미리보기", width="stretch"):
        try:
            with st.spinner("PDF 준비 중..."):
                path = ax.download_pdf(candidates[preview_id])  # 받아 둔 파일이 있으면 그대로 씀
            # 다른 위젯을 눌러도 유지되게 저장하되, 어떤 논문의 파일인지 함께 기록
            st.session_state["arxiv_preview"] = {"arxiv_id": preview_id, "path": str(path)}
        except Exception as e:
            show_error(f"{preview_id} 미리보기", e)
    preview = st.session_state.get("arxiv_preview")
    # 다른 논문을 고르면 이전 논문의 PDF를 그 이름 아래 보여 주지 않음
    if preview and preview["arxiv_id"] == preview_id and Path(preview["path"]).exists():
        st.pdf(preview["path"], height=800)

# ---------------------------------------------------------------------------
# 4. 질의
# ---------------------------------------------------------------------------
step_header("4", "논문 질의", f"저장된 논문 {len(stored)}편. 대상을 고르지 않으면 전체에서 찾습니다.")
if not stored:
    st.info("저장된 논문이 없습니다. 위에서 검색해 저장하세요.")
    st.stop()

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
        for i, (doc, distance) in enumerate(hits, start=1):
            with st.expander(f"[{i}] {doc.metadata.get('source', '?')} p.{rag.page_label(doc)} · 거리 {distance:.3f}"):
                st.text(doc.page_content)
