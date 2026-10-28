"""
[PDF RAG 실습 - 단계별 출력 확인 · Streamlit & LangChain]

A. 미리 준비하기 (인덱싱): 문서 읽기 → 청크로 나누기 → 임베딩하기 → 벡터 스토어 저장
B. 질문에 답하기 (런타임): 질문 입력 → 관련 청크 검색 → 프롬프트 구성 → LLM 답변
각 단계가 끝날 때마다 그 단계의 출력값(페이지 · 청크 · 벡터 · DB 행 · 검색 점수 · 프롬프트 · 답변)을 화면에 표시한다.

- 문서: PDF만 허용 (업로드 확장자 + 파일 헤더 %PDF- 검사)
- 임베딩 · LLM: Google Gemini (Vertex AI, ADC 인증)
- 벡터 스토어: docker-compose.yml의 PostgreSQL + pgvector (langchain-postgres PGVector)
- 청크 크기 · 겹침 · top-k · 임베딩 차원 · 지시문 등은 왼쪽 사이드바에서 변경 (ui_common.py, 평가 화면과 공유)
- 평가 화면: pages/ocr_eval.py (스캔 PDF OCR → 질문 · 정답 세트로 RAG 정답률 측정)

실행 방법:
    docker compose up -d                     # PostgreSQL(pgvector) 실행
    gcloud auth application-default login    # Vertex AI 인증 (만료 시)
    uv run streamlit run rag_app/app.py
"""

import sys
from pathlib import Path

# streamlit run은 스크립트 폴더(rag_app/)만 import 경로에 넣으므로 프로젝트 루트를 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from rag_app import config
from rag_app import pipeline as rag
from rag_app.ui_common import (
    get_answer_llm,
    get_embeddings,
    get_llm,
    get_store,
    preview,
    render_sidebar,
    show_error,
    step_header,
)

st.set_page_config(page_title="PDF RAG 실습", page_icon="📄", layout="wide")


settings = render_sidebar()
# 아래 코드가 쓰는 이름으로 풀어 둠
image_mode, image_prompt = settings.image_mode, settings.image_prompt
chunk_size, chunk_overlap, across_pages = settings.chunk_size, settings.chunk_overlap, settings.across_pages
embedding_model, embedding_dim, collection_name = settings.embedding_model, settings.embedding_dim, settings.collection
top_k, llm_choice, llm_model = settings.top_k, settings.llm_choice, settings.llm_model
max_output_tokens, instructions = settings.max_output_tokens, settings.instructions
chunk_settings_ok, answer_model = settings.chunk_ok, settings.answer_model


def run_index_steps(result: dict, data: bytes, file_name: str, index_info: dict) -> None:
    """A-1 ~ A-4를 차례로 실행하며 단계 결과를 result에 쌓음 (중간에 실패해도 그 전 단계 결과는 화면에 남음)."""
    use_images = image_mode == config.IMAGE_GEMINI
    st.write("1) 문서 읽기" + (f" (이미지는 {llm_model}로 변환 — 이미지가 많으면 오래 걸림)" if use_images else ""))
    # 실패 기록(failures)이 이전 인덱싱과 섞이지 않게 매번 새로 만듦 (LLM 객체는 캐시 재사용)
    image_reader = (
        rag.GeminiImageReader(get_llm(llm_model, config.IMAGE_MAX_OUTPUT_TOKENS), image_prompt)
        if use_images else None
    )
    result["pages"] = rag.load_pdf(data, file_name, image_reader)
    result["image_failures"] = image_reader.failures if image_reader else []
    st.write("2) 청크로 나누기")
    result["chunks"] = rag.split_documents(result["pages"], chunk_size, chunk_overlap, across_pages)
    if not result["chunks"]:
        raise ValueError(
            "추출된 텍스트가 없습니다. 이미지 안의 글자도 찾지 못했습니다."
            if use_images
            else "추출된 텍스트가 없습니다. 스캔 PDF라면 사이드바 'PDF 속 이미지'를 Gemini로 켜고 다시 인덱싱하세요."
        )
    st.write(f"3) 임베딩하기 ({len(result['chunks'])}개 청크)")
    embeddings = get_embeddings(embedding_model, embedding_dim)
    result["vectors"], result["embedding_cache_hits"] = rag.embed_chunks(embeddings, result["chunks"])
    st.write("4) 벡터 스토어 저장")
    store = get_store(embedding_model, embedding_dim, collection_name)
    result["ids"] = rag.save_to_store(
        store, result["chunks"], result["vectors"],
        {**index_info, "image_failures": len(result["image_failures"])},
    )


st.title("📄 PDF RAG 실습")
st.caption(
    "LangChain으로 구성한 RAG 파이프라인을 단계별로 실행하고 각 단계의 출력값을 확인합니다. "
    f"임베딩 `{embedding_model}` · 답변 `{answer_model}` · 벡터 스토어 PostgreSQL(pgvector) `{collection_name}`"
)

tab_index, tab_query = st.tabs(["A. 미리 준비하기 (인덱싱)", "B. 질문에 답하기 (런타임)"])

# ---------------------------------------------------------------------------
# A. 미리 준비하기 (인덱싱)
# ---------------------------------------------------------------------------
with tab_index:
    st.info("자료를 등록하거나 변경할 때 한 번 실행합니다. 결과는 PostgreSQL에 저장되어 앱을 다시 켜도 남습니다.")
    uploaded = st.file_uploader("PDF 파일", type=["pdf"], accept_multiple_files=False)

    # 저장 결과를 바꾸는 설정 — 파일 내용 해시와 합쳐 "같은 문서" 판단 키가 됨
    content_settings = {
        "image_mode": image_mode,
        "image_prompt": image_prompt,
        "image_model": llm_model if image_mode == config.IMAGE_GEMINI else None,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "across_pages": across_pages,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
    }
    index_settings = {"file": uploaded.name if uploaded else None, "collection": collection_name, **content_settings}

    force_reindex = st.checkbox(
        "같은 문서여도 다시 인덱싱",
        help="파일 내용과 설정이 마지막 인덱싱과 같으면 API 호출 없이 건너뜀. 체크하면 단계별 출력을 다시 만듦",
    )
    run_index = st.button(
        "인덱싱 실행", type="primary", disabled=uploaded is None or not chunk_settings_ok,
    )

    if run_index and uploaded is not None:
        result: dict = {"settings": index_settings}
        st.session_state.pop("index_result", None)
        with st.status("인덱싱 중…", expanded=True) as status:
            try:
                if not rag.pdf_name_is_valid(uploaded.name):
                    raise ValueError("PDF 파일만 업로드할 수 있습니다.")
                data = uploaded.getvalue()
                file_hash = rag.file_sha256(data)
                index_key = rag.make_index_key(file_hash, content_settings)
                same = None if force_reindex else rag.find_same_index(collection_name, index_key)
                if same:
                    result["skipped"] = same
                    status.update(label="이미 인덱싱된 문서 — 건너뜀", state="complete", expanded=False)
                else:
                    run_index_steps(
                        result, data, uploaded.name,
                        {
                            "index_key": index_key,
                            "file_sha256": file_hash,
                            "file_name": uploaded.name,
                            "settings": content_settings,
                        },
                    )
                    status.update(label="인덱싱 완료", state="complete", expanded=False)
            except Exception as e:
                status.update(label="인덱싱 실패", state="error")
                show_error("인덱싱", e)
        st.session_state["index_result"] = result

    result = st.session_state.get("index_result")
    if result:
        if result["settings"] != index_settings:
            st.warning("사이드바 설정이나 파일이 마지막 인덱싱 때와 다릅니다. 반영하려면 다시 인덱싱하세요.")
        if "skipped" in result:
            info = result["skipped"]
            st.success(
                f"같은 내용(sha256 `{info['file_sha256'][:12]}…`) · 같은 설정으로 {info['indexed_at']}에 "
                f"인덱싱된 문서라 건너뛰었습니다. 임베딩 · 이미지 읽기 API를 호출하지 않았습니다.\n\n"
                "단계별 출력을 다시 보려면 '같은 문서여도 다시 인덱싱'을 체크하고 실행하세요."
            )

        # 1) 문서 읽기
        if "pages" in result:
            st.divider()
            pages = result["pages"]
            step_header(
                "A-1", "문서 읽기",
                "PDF → 텍스트 · PyPDFLoader가 페이지마다 Document 하나를 만듦"
                + (f" · 이미지는 Gemini가 변환해 {rag.IMAGE_TEXT_MARK} 아래에 붙임"
                   if result["settings"]["image_mode"] == config.IMAGE_GEMINI else ""),
            )
            total_chars = sum(len(p.page_content) for p in pages)
            image_counts = [p.page_content.count(rag.IMAGE_TEXT_MARK) for p in pages]
            empty_pages = [p.metadata["page"] for p in pages if not p.page_content.strip()]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("페이지 수", len(pages))
            c2.metric("총 글자 수", f"{total_chars:,}")
            c3.metric("변환한 이미지", sum(image_counts))
            c4.metric("파일", pages[0].metadata["source"] if pages else "-")
            if result.get("image_failures"):
                st.warning(
                    f"이미지 {len(result['image_failures'])}개를 변환하지 못해 건너뛰었습니다. "
                    "일시적인 API 오류라면 다시 인덱싱하면 대부분 해결됩니다.\n\n" + "\n".join(f"- {f}" for f in result["image_failures"])
                )
            if empty_pages:
                hint = (
                    "사이드바 'PDF 속 이미지'를 Gemini로 켜면 스캔된 페이지도 읽을 수 있습니다."
                    if result["settings"]["image_mode"] == config.IMAGE_OFF
                    else "빈 페이지이거나, 이미지가 PDF 안에 특수한 형식으로 들어 있어 꺼내지 못한 경우입니다."
                )
                st.warning(f"글자가 없는 페이지 {len(empty_pages)}개는 검색 대상에서 빠집니다: {empty_pages}\n\n{hint}")
            st.dataframe(
                pd.DataFrame(
                    [{"page": p.metadata["page"], "글자 수": len(p.page_content), "이미지": n,
                      "미리보기": preview(p.page_content)}
                     for p, n in zip(pages, image_counts, strict=True)]
                ),
                hide_index=True, width="stretch",
            )
            page_no = st.number_input("전체 텍스트 볼 페이지", 1, max(len(pages), 1), 1, key="page_view")
            with st.expander(f"{page_no} 페이지 원문 · metadata"):
                st.json(pages[page_no - 1].metadata)
                st.text(pages[page_no - 1].page_content)

        # 2) 청크로 나누기
        if "chunks" in result:
            st.divider()
            chunks = result["chunks"]
            step_header(
                "A-2", "청크로 나누기",
                f"작은 원문 조각 · RecursiveCharacterTextSplitter(chunk_size={result['settings']['chunk_size']}, "
                f"chunk_overlap={result['settings']['chunk_overlap']})",
            )
            lengths = [len(c.page_content) for c in chunks]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("청크 수", len(chunks))
            c2.metric("평균 길이", f"{sum(lengths) / len(lengths):.0f}" if lengths else "-")
            c3.metric("최소 길이", min(lengths) if lengths else "-")
            c4.metric("최대 길이", max(lengths) if lengths else "-")
            if lengths:
                st.bar_chart(pd.DataFrame({"글자 수": lengths}), x_label="청크 번호", y_label="글자 수", height=200)
                chunk_no = st.number_input("자세히 볼 청크 번호", 0, len(chunks) - 1, 0, key="chunk_view")
                left, right = st.columns(2)
                with left:
                    st.markdown(f"**청크 {chunk_no}**")
                    st.json(chunks[chunk_no].metadata)
                    st.text(chunks[chunk_no].page_content)
                with right:
                    if chunk_no + 1 < len(chunks):
                        st.markdown(f"**다음 청크 {chunk_no + 1}** (앞부분이 겹침 영역)")
                        st.json(chunks[chunk_no + 1].metadata)
                        st.text(chunks[chunk_no + 1].page_content)

        # 3) 임베딩하기
        if "vectors" in result:
            st.divider()
            vectors = result["vectors"]
            step_header(
                "A-3", "임베딩하기",
                f"청크 → 숫자 배열 · {result['settings']['embedding_model']} (task_type=RETRIEVAL_DOCUMENT)",
            )
            hits = result.get("embedding_cache_hits", 0)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("벡터 수", len(vectors))
            c2.metric("차원", len(vectors[0]) if vectors else "-")
            c3.metric("캐시 사용", hits, help="같은 모델 · 차원 · 텍스트의 벡터를 embedding_cache 테이블에서 재사용")
            c4.metric("새로 계산 (API 호출)", len(vectors) - hits)
            st.dataframe(
                pd.DataFrame(
                    [{"chunk": i, **{f"v{j}": round(v, 5) for j, v in enumerate(vec[:8])}, "…": "…",
                      "원문 미리보기": preview(result["chunks"][i].page_content, 60)}
                     for i, vec in enumerate(vectors)]
                ),
                hide_index=True, width="stretch",
            )

        # 4) 벡터 스토어 저장
        if "ids" in result or "skipped" in result:
            st.divider()
            step_header(
                "A-4", "벡터 스토어 저장",
                f"벡터 + 원문 + 출처 연결 · PostgreSQL langchain_pg_embedding 테이블 (컬렉션 `{result['settings']['collection']}`)",
            )
            try:
                total, rows = rag.inspect_collection(result["settings"]["collection"])
                st.metric("저장된 행 수", total)
                st.dataframe(
                    pd.DataFrame(
                        [{"id": r.id, "document (원문)": preview(r.document, 60), "cmetadata (출처)": r.metadata,
                          "vector_dims": r.dimension, "embedding": r.vector_preview} for r in rows]
                    ),
                    hide_index=True, width="stretch",
                )
                with st.expander("컬렉션 metadata (같은 문서 판단에 쓰는 인덱싱 정보)"):
                    st.json(rag.get_collection_info(result["settings"]["collection"]))
            except Exception as e:
                show_error("저장 결과 조회", e)

# ---------------------------------------------------------------------------
# B. 질문에 답하기 (런타임)
# ---------------------------------------------------------------------------
with tab_query:
    st.info("질문이 들어올 때마다 실행합니다. A에서 저장한 컬렉션에서 관련 청크를 찾아 답변의 근거로 씁니다.")

    try:
        collection_info = rag.get_collection_info(collection_name) or {}
    except Exception as e:
        collection_info = {}
        show_error("컬렉션 조회", e)
    stored_total = collection_info.get("stored_chunks", 0)
    if stored_total == 0:
        st.warning(f"컬렉션 `{collection_name}`에 저장된 청크가 없습니다. A 탭에서 먼저 인덱싱하세요.")
    else:
        file_label = f" · {collection_info['file_name']}" if collection_info.get("file_name") else ""
        st.caption(f"컬렉션 `{collection_name}`{file_label} · 청크 {stored_total}개")

    with st.form("question_form"):
        question = st.text_area("질문", placeholder="문서 내용에 대해 질문하세요", height=80)
        run_chain_too = st.checkbox(
            "LCEL 체인으로도 한 번에 실행해 비교", value=False,
            help="단계별 실행과 같은 흐름을 체인 하나(retriever | prompt | llm | parser)로 실행",
        )
        asked = st.form_submit_button("질문하기", type="primary", disabled=stored_total == 0)

    if asked and question.strip():
        qa: dict = {"question": question.strip(), "k": top_k, "answer_model": answer_model}
        st.session_state.pop("qa_result", None)
        with st.status("답변 생성 중…", expanded=True) as status:
            try:
                embeddings = get_embeddings(embedding_model, embedding_dim)
                store = get_store(embedding_model, embedding_dim, collection_name)
                st.write("1) 질문 임베딩")
                qa["vector"] = rag.embed_question(embeddings, qa["question"])
                st.write("2) 관련 청크 검색")
                qa["hits"] = rag.search_chunks(store, qa["vector"], top_k)
                st.write("3) 프롬프트 구성")
                qa["messages"] = rag.build_prompt(instructions, [doc for doc, _ in qa["hits"]], qa["question"])
                st.write("4) LLM 답변")
                llm = get_answer_llm(llm_choice, llm_model, max_output_tokens)
                qa["answer"] = rag.generate_answer(llm, qa["messages"])
                if run_chain_too:
                    st.write("+) LCEL 체인 실행")
                    qa["chain_answer"] = rag.build_rag_chain(store, llm, instructions, top_k).invoke(qa["question"])
                status.update(label="답변 완료", state="complete", expanded=False)
            except Exception as e:
                status.update(label="실패", state="error")
                show_error("질문 처리", e)
        st.session_state["qa_result"] = qa

    qa = st.session_state.get("qa_result")
    if qa:
        # 1) 질문 입력
        if "vector" in qa:
            st.divider()
            step_header("B-1", "질문 입력", "질문도 임베딩 · 같은 모델, task_type=RETRIEVAL_QUERY")
            st.markdown(f"> {qa['question']}")
            c1, c2 = st.columns([1, 3])
            c1.metric("차원", len(qa["vector"]))
            c2.code("[" + ", ".join(f"{v:.5f}" for v in qa["vector"][:10]) + ", …]", language=None)

        # 2) 관련 청크 검색
        if "hits" in qa:
            st.divider()
            step_header(
                "B-2", "관련 청크 검색",
                f"숫자 비교 → 원문 반환 · 코사인 거리 상위 {qa['k']}개 (거리가 작을수록 질문과 가까움, 유사도 = 1 - 거리)",
            )
            st.dataframe(
                pd.DataFrame(
                    [{"순위": i, "코사인 거리": round(score, 4), "유사도": round(1 - score, 4),
                      "page": doc.metadata.get("page"), "chunk": doc.metadata.get("chunk_index"),
                      "원문 미리보기": preview(doc.page_content, 80)}
                     for i, (doc, score) in enumerate(qa["hits"], start=1)]
                ),
                hide_index=True, width="stretch",
            )
            for i, (doc, score) in enumerate(qa["hits"], start=1):
                with st.expander(f"[{i}] p.{doc.metadata.get('page')} · 거리 {score:.4f}"):
                    st.text(doc.page_content)

        # 3) 프롬프트 구성
        if "messages" in qa:
            st.divider()
            step_header("B-3", "프롬프트 구성", "질문 + 원문 + 지시문 · 원래 질문은 검색과 별도로 그대로 전달됨")
            for msg in qa["messages"]:
                st.markdown(f"**{msg.type}** · {len(msg.content):,}자")
                st.code(msg.content, language=None, wrap_lines=True)

        # 4) LLM → 답변
        if "answer" in qa:
            st.divider()
            answer = qa["answer"]
            step_header("B-4", "LLM → 답변", f"근거를 읽고 생성 · {qa['answer_model']}")
            st.markdown(answer.text)
            usage = answer.usage_metadata or {}
            c1, c2, c3 = st.columns(3)
            c1.metric("입력 토큰", f"{usage.get('input_tokens', 0):,}")
            c2.metric("출력 토큰", f"{usage.get('output_tokens', 0):,}")
            c3.metric("합계", f"{usage.get('total_tokens', 0):,}")

        if "chain_answer" in qa:
            st.divider()
            step_header("체인", "찾는 일과 쓰는 일을 연결한 전체 흐름", "build_rag_chain(): 위 B-1~B-4를 LCEL 체인 하나로 실행한 결과")
            st.code(
                '{"context": retriever | format_context, "question": RunnablePassthrough()}\n'
                "| prompt.partial(instructions=...) | llm | StrOutputParser()",
                language="python",
            )
            st.markdown(qa["chain_answer"])
