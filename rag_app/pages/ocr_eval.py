"""
[OCR · RAG 평가 - 스캔 PDF를 OCR로 읽고, 질문 · 정답 세트로 RAG 답변 품질을 측정]

1. OCR · 인덱싱: 스캔 PDF → Gemini로 페이지마다 인쇄 텍스트만 받아 적기(사진 · 그림 제외) → 청크 → 임베딩 → pgvector
2. 질문 · 정답: 평가 세트(rag_app/eval/han_gang_qa.json) 확인 · 수정 · 저장, Gemini로 새로 생성
3. RAG 평가: 질문마다 검색 → 답변 → Gemini 채점(correct / partial / incorrect) + 정답 페이지 검색 순위
4. 실행 비교: 설정을 바꿔 가며 실행한 결과를 나란히 비교

사이드바 설정 · 모델 · 벡터 스토어 · 단계 함수는 PDF RAG 실습 화면(app.py)과 공유한다.
실행: uv run streamlit run rag_app/app.py → 사이드바 맨 위 "🔎 OCR · RAG 평가" 링크
"""

import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pypdf
import streamlit as st

from rag_app import billing, config
from rag_app import evaluation as ev
from rag_app import pipeline as rag
from rag_app.ui_common import (
    get_answer_llm,
    get_llm,
    get_store,
    preview,
    render_header_toolbar,
    render_left_menu,
    show_error,
    step_header,
)

st.set_page_config(page_title="OCR · RAG 평가", page_icon="🔎", layout="wide")

# 왼쪽 사이드바: 네비게이션 메뉴 및 실시간 과금 카운터 위젯
render_left_menu()

st.title("🔎 OCR · RAG 평가")

# 상단 툴바: 빠른 설정(평가 문서, 답변 모델, top-k) 및 팝오버 상세 설정
settings = render_header_toolbar(page="eval")

doc = config.EVAL_DOCS[settings.eval_doc]
QA_KEY = f"eval_qa_{settings.eval_doc}"  # 문서마다 질문 세트를 따로 보관

VERDICT_MARK = {"correct": "✅ 정답", "partial": "🟡 부분", "incorrect": "❌ 오답", "error": "⚠️ 오류"}

st.caption(
    "스캔 PDF를 Gemini OCR로 읽어 인덱싱하고, 질문 · 정답 세트로 RAG 답변이 잘 나오는지 측정합니다. "
    f"{doc.label} · OCR `{settings.llm_model}` · 답변 `{settings.answer_model}` · 채점 `{settings.judge_model}` · 컬렉션 `{settings.collection}`"
)


@st.cache_data(show_spinner=False, max_entries=40)
def page_image(data: bytes, page_no: int):
    """원본 페이지의 가장 큰 이미지 (스캔 PDF는 페이지 전체). OCR 결과와 나란히 비교하는 용도."""
    images = list(pypdf.PdfReader(io.BytesIO(data)).pages[page_no - 1].images)
    return max((img.image for img in images), key=lambda im: im.size[0] * im.size[1], default=None)


def run_label(question_set: str) -> str:
    return (
        f"{doc.label} · 컬렉션 {settings.collection} · {settings.chunk_size}/{settings.chunk_overlap}"
        f"{' 페이지연결' if settings.across_pages else ''} · k={settings.top_k} · 답변 {settings.answer_model}"
        f" · 채점 {settings.judge_model} · {question_set}"
    )


tab_ocr, tab_qa, tab_eval, tab_compare = st.tabs(["1. OCR · 인덱싱", "2. 질문 · 정답", "3. RAG 평가", "4. 실행 비교"])

# ---------------------------------------------------------------------------
# 1. OCR · 인덱싱
# ---------------------------------------------------------------------------
with tab_ocr:
    uploaded = st.file_uploader("PDF 파일 (비우면 기본 평가 문서 사용)", type=["pdf"])
    if uploaded is not None:
        pdf_name, pdf_data = uploaded.name, uploaded.getvalue()
    elif doc.pdf.exists():
        pdf_name, pdf_data = doc.pdf.name, doc.pdf.read_bytes()
        st.caption(f"평가 문서: {doc.label} · `{doc.pdf.relative_to(config.EVAL_DIR.parent.parent)}`")
    else:
        pdf_name, pdf_data = None, None
        st.warning(f"평가 문서가 없습니다. `{doc.pdf}`에 PDF를 두거나 업로드하세요.")

    left, right = st.columns(2)
    refresh_ocr = left.checkbox(
        "캐시 무시하고 다시 OCR", help="OCR 결과는 페이지 이미지 · 모델 · 프롬프트별로 eval/data/ocr_pages/에 저장해 재사용 (다른 PDF의 같은 페이지도 공유)",
    )
    if left.button("OCR 실행", type="primary", disabled=pdf_data is None):
        with st.status("OCR 중… (페이지가 많으면 몇 분 걸림)", expanded=True) as status:
            try:
                if not pdf_data.startswith(rag.PDF_MAGIC):
                    raise ValueError("PDF 파일만 사용할 수 있습니다.")
                progress = st.progress(0.0, text="페이지 이미지 준비 중…")
                pages, failures, cached = ev.ocr_pdf(
                    pdf_data, pdf_name, get_llm(settings.llm_model, config.IMAGE_MAX_OUTPUT_TOKENS),
                    settings.llm_model, settings.ocr_prompt, use_cache=not refresh_ocr,
                    on_page=lambda done, total, page, from_cache: progress.progress(
                        done / total, text=f"{done}/{total}쪽 · {page}쪽 {'캐시' if from_cache else 'OCR 완료'}"
                    ),
                )
                progress.empty()
                st.session_state["eval_ocr"] = {
                    "file_name": pdf_name, "sha": rag.file_sha256(pdf_data), "prompt": settings.ocr_prompt,
                    "pages": pages, "failures": failures, "cached": cached,
                }
                status.update(label=f"OCR 완료 (캐시 {cached}쪽 · 새로 OCR {len(pages) - cached}쪽)", state="complete", expanded=False)
                new_ocr = len(pages) - cached
                if new_ocr > 0:
                    billing.record_billing_event(
                        stage="문서 이미지 읽기 (OCR)",
                        model_id=settings.llm_model,
                        input_tokens=new_ocr * 258,
                        output_tokens=new_ocr * 300,
                        collection_name=settings.collection,
                    )
            except Exception as e:
                status.update(label="OCR 실패", state="error")
                show_error("OCR", e)

    ocr = st.session_state.get("eval_ocr")
    if ocr:
        if ocr["file_name"] != pdf_name or ocr["prompt"] != settings.ocr_prompt:
            st.warning("파일이나 OCR 프롬프트가 마지막 OCR 때와 다릅니다. 반영하려면 다시 OCR 하세요.")
        pages = ocr["pages"]
        st.divider()
        step_header("1-1", "OCR 결과", f"{ocr['file_name']} · 사진 · 그림 · 지도 안의 글자와 그림 설명은 제외, 캡션은 포함")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("페이지", len(pages))
        c2.metric("총 글자 수", f"{sum(len(p.page_content) for p in pages):,}")
        c3.metric("실패한 페이지", len(ocr["failures"]))
        c4.metric("캐시에서 가져온 쪽", f"{ocr['cached']} / {len(pages)}", help="페이지 이미지 단위 캐시 (eval/data/ocr_pages/)")
        if ocr["failures"]:
            st.warning("OCR에 실패한 페이지는 캐시에 저장하지 않았습니다. 다시 실행하면 그 페이지만 이어서 OCR 합니다:\n\n"
                       + "\n".join(f"- {f}" for f in ocr["failures"]))
        st.dataframe(
            pd.DataFrame(
                [{"page": p.metadata["page"], "글자 수": len(p.page_content), "미리보기": preview(p.page_content)} for p in pages]
            ),
            hide_index=True, width="stretch", height=240,
        )
        page_no = st.number_input("원본과 비교할 페이지", 1, len(pages), 1, key="ocr_page_view")
        img_col, text_col = st.columns(2)
        with img_col:
            image = page_image(pdf_data, page_no) if pdf_data and ocr["file_name"] == pdf_name else None
            if image is not None:
                st.image(image, caption=f"원본 {page_no}쪽", width="stretch")
            else:
                st.caption("원본 이미지를 표시할 수 없습니다.")
        with text_col:
            st.text(pages[page_no - 1].page_content or "(인쇄된 텍스트 없음)")

        st.divider()
        step_header(
            "1-2", "인덱싱",
            f"청크 {settings.chunk_size}/{settings.chunk_overlap}{' · 페이지 연결' if settings.across_pages else ''} "
            f"→ {settings.embedding_model} {settings.embedding_dim}차원 → pgvector `{settings.collection}`",
        )
        content_settings = {
            "ocr_prompt": ocr["prompt"],
            "ocr_model": settings.llm_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "across_pages": settings.across_pages,
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
        }
        index_key = rag.make_index_key(ocr["sha"], content_settings)
        force = st.checkbox("같은 문서 · 설정이어도 다시 인덱싱", key="eval_force_index")
        if st.button("인덱싱 실행", disabled=not settings.chunk_ok):
            try:
                same = None if force else rag.find_same_index(settings.collection, index_key)
                if same:
                    st.success(f"같은 OCR 결과 · 같은 설정으로 {same['indexed_at']}에 인덱싱돼 있어 건너뛰었습니다.")
                else:
                    with st.spinner("청크 · 임베딩 · 저장 중…"):
                        chunks = rag.split_documents(pages, settings.chunk_size, settings.chunk_overlap, settings.across_pages)
                        store = get_store(settings.embedding_model, settings.embedding_dim, settings.collection)
                        vectors, cache_hits = rag.embed_chunks(store.embeddings, chunks)
                        # 과금 카운터: 신규 임베딩(캐시 미적용분) 과금 기록
                        api_chunks = len(chunks) - cache_hits
                        if api_chunks > 0:
                            total_chars = sum(len(c.page_content) for c in chunks[:api_chunks])
                            in_tokens = max(1, int(total_chars / 2.2))
                            billing.record_billing_event(
                                stage="문서 청크 임베딩",
                                model_id=settings.embedding_model,
                                input_tokens=in_tokens,
                                output_tokens=0,
                                collection_name=settings.collection,
                                chunk_size=settings.chunk_size,
                                chunk_overlap=settings.chunk_overlap,
                            )
                        if cache_hits > 0:
                            cached_chars = sum(len(c.page_content) for c in chunks[api_chunks:])
                            cached_tokens = max(1, int(cached_chars / 2.2))
                            saved_cost = billing.calculate_billing_cost(settings.embedding_model, cached_tokens, 0)
                            billing.record_billing_event(
                                stage="문서 청크 임베딩 (캐시 적중)",
                                model_id=settings.embedding_model,
                                input_tokens=0,
                                output_tokens=0,
                                collection_name=settings.collection,
                                cache_hit=True,
                                cache_hit_count=cache_hits,
                                cached_tokens=cached_tokens,
                                saved_cost_krw=saved_cost["cost_krw"],
                                chunk_size=settings.chunk_size,
                                chunk_overlap=settings.chunk_overlap,
                            )
                        rag.save_to_store(
                            store, chunks, vectors,
                            {"index_key": index_key, "file_sha256": ocr["sha"], "file_name": ocr["file_name"],
                             "settings": content_settings, "image_failures": len(ocr["failures"])},
                        )
                    multi = sum(c.metadata.get("page_end", c.metadata["page"]) != c.metadata["page"] for c in chunks)
                    st.success(
                        f"청크 {len(chunks)}개 저장 (여러 페이지에 걸친 청크 {multi}개) · "
                        f"임베딩 캐시 사용 {cache_hits}개 / 새로 계산 {len(chunks) - cache_hits}개"
                    )
            except Exception as e:
                show_error("인덱싱", e)
        try:
            info = rag.get_collection_info(settings.collection)
            if info and info.get("stored_chunks"):
                with st.expander(f"컬렉션 `{settings.collection}` · 청크 {info['stored_chunks']}개 · {info.get('indexed_at', '')}"):
                    st.json({k: v for k, v in info.items() if k != "settings"} | {"settings": info.get("settings")})
        except Exception as e:
            show_error("컬렉션 조회", e)

# ---------------------------------------------------------------------------
# 2. 질문 · 정답
# ---------------------------------------------------------------------------
with tab_qa:
    if QA_KEY not in st.session_state:
        st.session_state[QA_KEY] = ev.load_qa(doc.qa_file) if doc.qa_file.exists() else []
    st.caption(
        f"평가 세트: `{doc.qa_file.relative_to(config.EVAL_DIR.parent.parent)}` "
        f"({'없음 — 생성 후 저장하세요' if not doc.qa_file.exists() else f'{len(st.session_state[QA_KEY])}문항'}) · "
        "paraphrase는 같은 질문을 문서 표현을 피해 사용자 말투로 바꾼 것 (생성된 질문은 문서 문장을 그대로 써서 검색이 쉬움)"
    )
    edited = st.data_editor(
        pd.DataFrame([asdict(item) for item in st.session_state[QA_KEY]],
                     columns=["id", "answerable", "page", "question", "paraphrase", "answer", "note"]),
        num_rows="dynamic", width="stretch", height=420, key=f"qa_editor_{settings.eval_doc}",
        column_config={
            "id": st.column_config.NumberColumn(width="small"),
            "answerable": st.column_config.CheckboxColumn("답 있음", width="small", default=True,
                                                          help=f"끄면 문서에 답이 없는 질문 (정답 '{config.UNANSWERABLE}', page 0)"),
            "page": st.column_config.NumberColumn(width="small"),
        },
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("파일에 저장"):
        items = [
            ev.QAItem(id=int(r.id), question=str(r.question), answer=str(r.answer), page=int(r.page),
                      note=str(r.note or ""), paraphrase=str(r.paraphrase or ""), answerable=bool(r.answerable))
            for r in edited.dropna(subset=["question", "answer", "page"]).itertuples()
        ]
        ev.save_qa(doc.qa_file, items)
        st.session_state[QA_KEY] = items
        st.success(f"{len(items)}개 저장")
    count = c2.number_input("생성 개수", 1, 50, config.EVAL_QUESTION_COUNT, label_visibility="collapsed")
    if c3.button(f"OCR 결과로 Gemini가 {count}개 새로 생성", disabled="eval_ocr" not in st.session_state):
        with st.spinner("질문 · 정답 생성 중…"):
            try:
                st.session_state[QA_KEY] = ev.generate_qa(
                    get_llm(settings.llm_model, 16384), st.session_state["eval_ocr"]["pages"], count,
                )
                st.session_state.pop(f"qa_editor_{settings.eval_doc}", None)
                st.rerun()
            except Exception as e:
                show_error("질문 생성", e)
    st.caption("생성 결과는 화면에만 반영됩니다. 정답 · 페이지를 원문과 대조해 검수한 뒤 '파일에 저장'하세요.")

# ---------------------------------------------------------------------------
# 3. RAG 평가
# ---------------------------------------------------------------------------
with tab_eval:
    items: list[ev.QAItem] = st.session_state.get(QA_KEY, [])
    try:
        info = rag.get_collection_info(settings.collection) or {}
    except Exception as e:
        info = {}
        show_error("컬렉션 조회", e)
    if not info.get("stored_chunks"):
        st.warning(f"컬렉션 `{settings.collection}`이 비어 있습니다. 1번 탭에서 OCR · 인덱싱을 먼저 하세요.")
    else:
        stored = info.get("settings", {})
        st.caption(
            f"컬렉션 `{settings.collection}` · 청크 {info['stored_chunks']}개 · 인덱싱 설정 "
            f"{stored.get('chunk_size')}/{stored.get('chunk_overlap')}{' 페이지연결' if stored.get('across_pages') else ''}"
        )
    question_set = st.radio(
        "질문 세트", ["원문 질문", "사용자 말투 질문"], horizontal=True,
        help="원문 질문은 생성 당시 문서 표현을 그대로 써서 검색이 쉬움. 사용자 말투 질문이 실제 상황에 가까움",
    )
    if settings.judge_model == settings.answer_model:
        st.warning("채점 모델이 답변 모델과 같습니다. 자기 답에 후한 편향이 생길 수 있어 사이드바 C에서 다른 모델을 권장합니다.")
    if st.button(f"{len(items)}개 질문 평가 실행", type="primary", disabled=not items or not info.get("stored_chunks")):
        progress = st.progress(0.0, text="평가 중…")
        try:
            results = ev.evaluate(
                items,
                get_store(settings.embedding_model, settings.embedding_dim, settings.collection),
                get_answer_llm(settings.llm_choice, settings.llm_model, settings.max_output_tokens),
                get_llm(settings.judge_model, 8192),
                settings.instructions,
                settings.top_k,
                use_paraphrase=question_set == "사용자 말투 질문",
                on_done=lambda done, r: progress.progress(done / len(items), text=f"{done}/{len(items)} · {r.id}번 {VERDICT_MARK[r.verdict]}"),
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            run = {"label": run_label(question_set), "summary": ev.summarize(results), "results": results,
                   "settings": {"chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
                                "across_pages": settings.across_pages, "top_k": settings.top_k,
                                "answer_model": settings.answer_model, "judge_model": settings.judge_model,
                                "question_set": question_set}}
            st.session_state.setdefault("eval_runs", []).append(run)
            progress.empty()
        except Exception as e:
            progress.empty()
            show_error("평가", e)

    runs = st.session_state.get("eval_runs", [])
    if runs:
        run = runs[-1]
        summary = run["summary"]
        st.divider()
        step_header("3-1", "결과 요약", run["label"])
        fmt = lambda v, f: "-" if v is None else format(v, f)
        st.markdown("**답변**")
        c = st.columns(5)
        c[0].metric("점수 (답 있는 질문)", f"{summary['score']:.2f}", help="정답 1 · 부분 0.5 · 오답 0의 평균")
        c[1].metric("정답 / 부분 / 오답", f"{summary['correct']} / {summary['partial']} / {summary['incorrect']}")
        c[2].metric("거절 정확도", fmt(summary.get("refusal_accuracy"), ".0%"),
                    help="문서에 답이 없는 질문에 '찾을 수 없다'고 답한 비율 (지어내지 않는지)")
        c[3].metric("근거 충실도", fmt(summary.get("faithfulness"), ".2f"),
                    help="답변의 사실 주장 중 검색된 문맥에 근거한 비율의 평균 (환각 측정, 정답 데이터 불필요)")
        c[4].metric("근거 없는 주장이 있는 답변", summary.get("unsupported_answers", 0))
        st.markdown("**검색 · 실행**")
        c = st.columns(5)
        c[0].metric("정답 페이지 검색률", f"{summary['page_hit_rate']:.0%}")
        c[1].metric("MRR", f"{summary['mrr']:.3f}", help="정답 페이지 청크 순위의 역수 평균 (1위 1, 2위 0.5, 못 찾음 0)")
        c[2].metric("평균 답변 시간", f"{summary['avg_latency']}초")
        c[3].metric("오류", summary["error"])
        agreement_slot = c[4].empty()

        # 사람 확인: '사람 판정' 칸을 채우면 LLM 채점과의 일치율을 계산 (채점을 믿어도 되는지 확인)
        table = pd.DataFrame(
            [{"id": r.id, "유형": "답 있음" if r.answerable else "답 없음", "판정": VERDICT_MARK[r.verdict],
              "사람 판정": None, "근거 충실도": r.faithfulness, "정답 쪽": r.gold_page or "-",
              "검색 순위": r.gold_rank or "-",
              "검색된 쪽": ", ".join(f"{x['page']}" + (f"~{x['page_end']}" if x["page_end"] != x["page"] else "") for x in r.retrieved),
              "질문": r.question, "답변": preview(r.answer, 80), "판정 이유": r.reason}
             for r in run["results"]]
        )
        reviewed = st.data_editor(
            table, hide_index=True, width="stretch", key=f"human_review_{len(runs)}",
            disabled=[col for col in table.columns if col != "사람 판정"],
            column_config={"사람 판정": st.column_config.SelectboxColumn(
                options=list(VERDICT_MARK.values())[:3], help="직접 판정해 LLM 채점과 비교 (일부만 채워도 됨)")},
        )
        filled = reviewed[reviewed["사람 판정"].notna()]
        if len(filled):
            agree = (filled["사람 판정"] == filled["판정"]).mean()
            agreement_slot.metric("사람 · LLM 채점 일치율", f"{agree:.0%}", help=f"사람이 판정한 {len(filled)}문항 기준")
        else:
            agreement_slot.metric("사람 · LLM 채점 일치율", "-", help="표의 '사람 판정' 칸을 채우면 계산")

        for r in run["results"]:
            if r.verdict != "correct" or r.unsupported:
                title = f"{VERDICT_MARK[r.verdict]} {r.id}번 · {r.question}" + (" · ⚠️ 근거 없는 주장" if r.unsupported else "")
                with st.expander(title):
                    st.markdown(f"**정답 (p.{r.gold_page})** {r.gold}\n\n**답변** {r.answer}\n\n**판정 이유** {r.reason}")
                    if r.unsupported:
                        st.markdown("**검색된 문맥에 근거가 없는 주장**\n" + "\n".join(f"- {c}" for c in r.unsupported))
                    st.json(r.retrieved)
        st.download_button(
            "결과 JSON 내려받기",
            json.dumps({**run, "results": [asdict(r) for r in run["results"]]}, ensure_ascii=False, indent=1),
            file_name="rag_eval_result.json", mime="application/json",
        )

# ---------------------------------------------------------------------------
# 4. 실행 비교
# ---------------------------------------------------------------------------
with tab_compare:
    runs = st.session_state.get("eval_runs", [])
    if not runs:
        st.info("3번 탭에서 평가를 실행하면 여기에 실행별 결과가 쌓입니다. 사이드바 설정을 바꿔 가며 비교해 보세요.")
    else:
        table = pd.DataFrame([{"실행": i, "설정": run["label"], **run["summary"]} for i, run in enumerate(runs, start=1)])
        st.dataframe(table, hide_index=True, width="stretch")
        st.bar_chart(table.set_index("실행")[["score", "page_hit_rate", "mrr"]], height=260)
        if st.button("비교 기록 지우기"):
            st.session_state.pop("eval_runs")
            st.rerun()
