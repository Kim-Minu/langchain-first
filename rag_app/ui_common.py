"""두 화면(app.py · pages/ocr_eval.py)이 함께 쓰는 사이드바 설정 · 캐시된 모델/스토어 · 화면 도우미."""

from dataclasses import dataclass

import streamlit as st

from rag_app import billing, config
from rag_app import pipeline as rag

# 사이드바 위젯의 기본값. 위젯은 key로만 만들고 값은 session_state에 둠
# (Streamlit은 화면을 옮기면 그 화면에 없는 위젯 값을 지우므로, 매 실행마다 다시 대입해 두 화면이 같은 설정을 공유)
DEFAULTS = {
    "cfg_image_mode": config.IMAGE_MODE,
    "cfg_image_prompt": config.IMAGE_PROMPT,
    "cfg_ocr_prompt": config.OCR_PROMPT,
    "cfg_chunk_size": config.CHUNK_SIZE,
    "cfg_chunk_overlap": config.CHUNK_OVERLAP,
    "cfg_across_pages": config.CHUNK_ACROSS_PAGES,
    "cfg_embedding_model": config.EMBEDDING_MODEL,
    "cfg_embedding_dim": config.EMBEDDING_DIMENSION,
    "cfg_collection": config.COLLECTION_NAME,
    "cfg_eval_doc": config.EVAL_DOC,
    "cfg_eval_collection": config.EVAL_DOCS[config.EVAL_DOC].collection,
    "cfg_top_k": config.TOP_K,
    "cfg_llm_choice": config.LLM_DEEPSEEK_FLASH,
    "cfg_llm_model": config.LLM_MODEL,
    "cfg_judge_model": config.JUDGE_MODEL,
    "cfg_max_output_tokens": config.MAX_OUTPUT_TOKENS,
    "cfg_instructions": config.SYSTEM_PROMPT,
}


@dataclass
class Settings:
    image_mode: str
    image_prompt: str | None  # 이미지 읽기를 끄면 None
    ocr_prompt: str
    chunk_size: int
    chunk_overlap: int
    across_pages: bool
    embedding_model: str
    embedding_dim: int
    collection: str
    top_k: int
    llm_choice: str
    llm_model: str  # Gemini 모델 ID (Gemini 답변 · 이미지 읽기 · OCR · 채점)
    max_output_tokens: int
    instructions: str
    judge_model: str  # 평가 화면의 채점 · 근거 충실도 판정 모델
    eval_doc: str  # 평가 화면의 평가 문서 (config.EVAL_DOCS 키)

    @property
    def chunk_ok(self) -> bool:
        return self.chunk_overlap < self.chunk_size

    @property
    def answer_model(self) -> str:
        if self.llm_choice == config.LLM_DEEPSEEK_FLASH:
            return config.DEEPSEEK_FLASH_MODEL
        if self.llm_choice == config.LLM_GEMMA_26B:
            return config.GEMMA_26B_MODEL
        if self.llm_choice == config.LLM_GEMINI_FLASH_LITE:
            return config.GEMINI_FLASH_LITE_MODEL
        if self.llm_choice == config.LLM_CLAUDE_HAIKU:
            return config.CLAUDE_HAIKU_MODEL
        if self.llm_choice == config.LLM_OLLAMA_GEMMA:
            return config.OLLAMA_MODEL
        return self.llm_model


def _follow_eval_doc() -> None:
    """평가 문서를 바꾸면 컬렉션 이름도 그 문서의 기본 컬렉션으로."""
    st.session_state["cfg_eval_collection"] = config.EVAL_DOCS[st.session_state["cfg_eval_doc"]].collection


def render_embedding_cache_widget() -> None:
    """왼쪽 사이드바에 PostgreSQL 임베딩 캐시 상태(활성화 여부, 캐시된 청크/질문 수)를 표출합니다."""
    with st.container(border=True):
        st.markdown("##### ⚡ 임베딩 캐시 상태")
        stats = rag.get_embedding_cache_stats()
        if stats.get("enabled"):
            st.caption("🟢 **캐시 활성화됨** (`embedding_cache`)")
            total = stats.get("total", 0)
            doc_cnt = stats.get("doc_count", 0)
            query_cnt = stats.get("query_count", 0)
            
            c1, c2 = st.columns(2)
            c1.metric("문서 청크", f"{doc_cnt:,}개")
            c2.metric("질문 벡터", f"{query_cnt:,}개")
            st.caption(f"총 {total:,}개 임베딩 저장됨 · 재호출 $0")

            if st.button("🗑️ 임베딩 캐시 비우기", key="btn_clear_embed_cache", width="stretch"):
                deleted = rag.clear_embedding_cache()
                st.success(f"{deleted:,}개 캐시 삭제 완료")
                st.rerun()
        else:
            st.caption("⚪ **캐시 비활성화됨** (`config.EMBEDDING_CACHE=False`)")


def render_left_menu() -> None:
    """왼쪽 사이드바(st.sidebar): 네비게이션 메뉴, 실시간 과금 카운터 및 임베딩 캐시 상태 위젯."""
    with st.sidebar:
        # 프로젝트 공통 .streamlit/config.toml이 자동 페이지 목록을 숨기므로(web_chat.py용 showSidebarNavigation = false)
        # 페이지 이동 링크를 직접 배치
        st.page_link("app.py", label="PDF RAG 실습", icon="📄")
        st.page_link("pages/ocr_eval.py", label="OCR · RAG 평가", icon="🔎")
        st.page_link("pages/arxiv_qa.py", label="arXiv 논문 질의", icon="📚")
        st.page_link("pages/benchmark.py", label="LLM 성능 벤치마크", icon="⚡")
        st.page_link("pages/billing_dashboard.py", label="과금 대시보드", icon="💰")
        st.divider()

        # 1. 실시간 과금 카운터 위젯 (모든 페이지 공통)
        billing.render_billing_counter_widget()

        # 2. 임베딩 캐시 상태 위젯
        render_embedding_cache_widget()


def render_header_toolbar(page: str = "main") -> Settings:
    """상단 가로 툴바: 자주 쓰는 핵심 설정(모델, top-k, 컬렉션)과 팝오버 상세 설정(청크, 임베딩, 프롬프트)."""
    for key, default in DEFAULTS.items():
        st.session_state[key] = st.session_state.get(key, default)

    with st.container(border=True):
        if page == "eval":
            col_doc, col_model, col_topk, col_adv = st.columns([1.8, 2.2, 1.0, 1.0], gap="medium")
            with col_doc:
                st.selectbox(
                    "📄 평가 문서", list(config.EVAL_DOCS),
                    format_func=lambda key: config.EVAL_DOCS[key].label,
                    key="cfg_eval_doc", on_change=_follow_eval_doc,
                    help="고르면 PDF · 질문 세트 · 기본 컬렉션이 함께 전환됩니다.",
                )
            with col_model:
                st.selectbox(
                    "🤖 답변 모델", list(config.LLM_OPTIONS),
                    format_func=config.LLM_OPTIONS.get,
                    key="cfg_llm_choice",
                )
            with col_topk:
                st.number_input(
                    "🔎 top-k", config.TOP_K_MIN, config.TOP_K_MAX,
                    step=1, key="cfg_top_k", help="검색할 청크 개수",
                )
            with col_adv:
                st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
                with st.popover("⚙️ 상세 설정", width="stretch"):
                    st.markdown("##### ⚙️ 평가 및 모델 상세 설정")
                    tab_ind, tab_mod = st.tabs(["📁 인덱싱 & OCR", "💬 프롬프트 & 채점"])
                    with tab_ind:
                        st.text_input("컬렉션 이름 (pgvector)", key="cfg_eval_collection")
                        st.text_area("OCR 프롬프트", key="cfg_ocr_prompt", height=150)
                        c1, c2 = st.columns(2)
                        c1.number_input("청크 크기 (글자)", config.CHUNK_SIZE_MIN, config.CHUNK_SIZE_MAX, step=100, key="cfg_chunk_size")
                        c2.number_input("청크 겹침 (글자)", config.CHUNK_OVERLAP_MIN, config.CHUNK_OVERLAP_MAX, step=50, key="cfg_chunk_overlap")
                        st.checkbox("페이지를 넘어 이어서 청크 나누기", key="cfg_across_pages")
                        e1, e2 = st.columns(2)
                        e1.selectbox("임베딩 모델", list(config.EMBEDDING_OPTIONS), format_func=config.EMBEDDING_OPTIONS.get, key="cfg_embedding_model")
                        e2.selectbox("임베딩 차원", config.EMBEDDING_DIMENSION_OPTIONS, key="cfg_embedding_dim")
                    with tab_mod:
                        st.text_input("채점 모델 (Gemini)", key="cfg_judge_model", help="정답 채점 및 근거 충실도 판정 모델")
                        st.number_input("출력 토큰 상한", 256, 65536, step=256, key="cfg_max_output_tokens")
                        st.text_area("지시문 (시스템 프롬프트)", key="cfg_instructions", height=150)
        else:
            col_model, col_topk, col_coll, col_adv = st.columns([2.2, 0.9, 2.2, 0.9], gap="medium")
            with col_model:
                st.selectbox(
                    "🤖 답변 모델", list(config.LLM_OPTIONS),
                    format_func=config.LLM_OPTIONS.get,
                    key="cfg_llm_choice",
                    help="질문에 답변을 생성할 LLM 모델",
                )
            with col_topk:
                st.number_input(
                    "🔎 top-k", config.TOP_K_MIN, config.TOP_K_MAX,
                    step=1, key="cfg_top_k", help="검색할 청크 개수",
                )
            with col_coll:
                # arXiv 논문 컬렉션은 여기서 저장하면 통째로 비워지므로 목록에서 뺌 (논문 화면 전용)
                stored = [c for c in rag.list_stored_collections() if not rag.is_paper_collection(c["metadata"])]
                stored_names = [c["name"] for c in stored]
                current = st.session_state.get("cfg_collection", config.COLLECTION_NAME)

                chunk_map = {c["name"]: c["stored_chunks"] for c in stored}
                file_map = {c["name"]: c["file_name"] for c in stored}

                if not stored_names:
                    st.text_input(
                        "📚 컬렉션 (pgvector)",
                        key="cfg_collection",
                        help="청크가 저장된 PostgreSQL pgvector 컬렉션명",
                    )
                else:
                    options = list(stored_names)
                    if current and current not in options and current != "➕ 직접 입력...":
                        options.insert(0, current)
                    options.append("➕ 직접 입력...")

                    def fmt_coll(name: str) -> str:
                        if name == "➕ 직접 입력...":
                            return "➕ 직접 입력..."
                        cnt = chunk_map.get(name)
                        cnt_str = f" ({cnt}개)" if cnt is not None else ""
                        fname = file_map.get(name)
                        fn_str = f" · {fname}" if fname else ""
                        return f"🗂️ {name}{cnt_str}{fn_str}"

                    default_idx = options.index(current) if current in options else 0
                    selected = st.selectbox(
                        "📚 컬렉션 (pgvector)",
                        options,
                        index=default_idx,
                        format_func=fmt_coll,
                        help="청크가 저장된 PostgreSQL pgvector 컬렉션을 선택합니다.",
                    )
                    if selected == "➕ 직접 입력...":
                        custom_name = st.text_input(
                            "신규 컬렉션 이름",
                            value="" if current in stored_names or current == "➕ 직접 입력..." else current,
                            placeholder="예: doc_report_c500_o50_d1536",
                        )
                        if custom_name.strip():
                            st.session_state["cfg_collection"] = custom_name.strip()
                    else:
                        st.session_state["cfg_collection"] = selected
            with col_adv:
                st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
                with st.popover("⚙️ 상세 설정", width="stretch"):
                    st.markdown("##### ⚙️ 파이프라인 상세 설정")
                    tab_ind, tab_mod = st.tabs(["📁 인덱싱 (청크 · 임베딩)", "💬 답변 & 프롬프트"])
                    with tab_ind:
                        st.radio(
                            "PDF 속 이미지 처리", list(config.IMAGE_OPTIONS),
                            format_func=config.IMAGE_OPTIONS.get, key="cfg_image_mode",
                        )
                        if st.session_state.cfg_image_mode == config.IMAGE_GEMINI:
                            st.text_area("이미지 읽기 프롬프트", key="cfg_image_prompt", height=120)
                        c1, c2 = st.columns(2)
                        c1.number_input("청크 크기 (글자)", config.CHUNK_SIZE_MIN, config.CHUNK_SIZE_MAX, step=100, key="cfg_chunk_size")
                        c2.number_input("청크 겹침 (글자)", config.CHUNK_OVERLAP_MIN, config.CHUNK_OVERLAP_MAX, step=50, key="cfg_chunk_overlap")
                        st.checkbox("페이지를 넘어 이어서 청크 나누기", key="cfg_across_pages")
                        e1, e2 = st.columns(2)
                        e1.selectbox("임베딩 모델", list(config.EMBEDDING_OPTIONS), format_func=config.EMBEDDING_OPTIONS.get, key="cfg_embedding_model")
                        e2.selectbox("임베딩 차원", config.EMBEDDING_DIMENSION_OPTIONS, key="cfg_embedding_dim")
                    with tab_mod:
                        selected_choice = st.session_state.get("cfg_llm_choice")
                        selected_id = {
                            config.LLM_DEEPSEEK_FLASH: config.DEEPSEEK_FLASH_MODEL,
                            config.LLM_GEMMA_26B: config.GEMMA_26B_MODEL,
                            config.LLM_GEMINI_FLASH_LITE: config.GEMINI_FLASH_LITE_MODEL,
                            config.LLM_CLAUDE_HAIKU: config.CLAUDE_HAIKU_MODEL,
                            config.LLM_GEMINI: config.LLM_MODEL,
                            config.LLM_OLLAMA_GEMMA: config.OLLAMA_MODEL,
                        }.get(selected_choice, "")
                        if selected_id:
                            st.caption(f"호출 모델 식별자: `{selected_id}`")
                        st.number_input("출력 토큰 상한", 256, 65536, step=256, key="cfg_max_output_tokens")
                        st.text_area("지시문 (시스템 프롬프트)", key="cfg_instructions", height=150)

    s = st.session_state
    collection_key = "cfg_eval_collection" if page == "eval" else "cfg_collection"
    settings = Settings(
        image_mode=s.cfg_image_mode,
        # 비우면 기본 프롬프트 사용 (빈 지시로 호출하면 결과가 들쭉날쭉해짐)
        image_prompt=(s.cfg_image_prompt.strip() or config.IMAGE_PROMPT) if s.cfg_image_mode == config.IMAGE_GEMINI else None,
        ocr_prompt=s.cfg_ocr_prompt.strip() or config.OCR_PROMPT,
        chunk_size=s.cfg_chunk_size,
        chunk_overlap=s.cfg_chunk_overlap,
        across_pages=s.cfg_across_pages,
        embedding_model=s.cfg_embedding_model,
        embedding_dim=s.cfg_embedding_dim,
        collection=s[collection_key],
        top_k=s.cfg_top_k,
        llm_choice=s.cfg_llm_choice,
        llm_model=s.cfg_llm_model,
        max_output_tokens=s.cfg_max_output_tokens,
        instructions=s.cfg_instructions,
        judge_model=s.cfg_judge_model,
        eval_doc=s.cfg_eval_doc,
    )
    if not settings.chunk_ok:
        st.error("청크 겹침은 청크 크기보다 작아야 합니다.")
    return settings


def render_settings_panel(container=None, page: str = "main") -> Settings:
    """하위 호환성을 위해 유지: render_header_toolbar로 위임."""
    return render_header_toolbar(page=page)


def render_sidebar(page: str = "main", settings_container=None) -> Settings:
    """왼쪽 사이드바 메뉴 렌더링 및 설정 반환."""
    render_left_menu()
    for key, default in DEFAULTS.items():
        st.session_state[key] = st.session_state.get(key, default)
    s = st.session_state
    collection_key = "cfg_eval_collection" if page == "eval" else "cfg_collection"
    return Settings(
        image_mode=s.cfg_image_mode,
        image_prompt=(s.cfg_image_prompt.strip() or config.IMAGE_PROMPT) if s.cfg_image_mode == config.IMAGE_GEMINI else None,
        ocr_prompt=s.cfg_ocr_prompt.strip() or config.OCR_PROMPT,
        chunk_size=s.cfg_chunk_size,
        chunk_overlap=s.cfg_chunk_overlap,
        across_pages=s.cfg_across_pages,
        embedding_model=s.cfg_embedding_model,
        embedding_dim=s.cfg_embedding_dim,
        collection=s[collection_key],
        top_k=s.cfg_top_k,
        llm_choice=s.cfg_llm_choice,
        llm_model=s.cfg_llm_model,
        max_output_tokens=s.cfg_max_output_tokens,
        instructions=s.cfg_instructions,
        judge_model=s.cfg_judge_model,
        eval_doc=s.cfg_eval_doc,
    )


@st.cache_resource(show_spinner=False)
def get_embeddings(model: str, dimension: int):
    return rag.create_embeddings(model, dimension)


@st.cache_resource(show_spinner=False)
def get_store(model: str, dimension: int, collection: str):
    return rag.get_vector_store(get_embeddings(model, dimension), collection)


@st.cache_resource(show_spinner=False)
def get_llm(model: str, max_tokens: int):
    return rag.create_llm(model, max_tokens)


@st.cache_resource(show_spinner=False)
def get_answer_llm(choice: str, gemini_model: str, max_tokens: int):
    return rag.create_answer_llm(choice, max_tokens, gemini_model)


def step_header(label: str, title: str, caption: str) -> None:
    st.markdown(f"#### {label}. {title}")
    st.caption(caption)


def show_error(step: str, error: Exception) -> None:
    hint = ""
    message = str(error)
    if "Reauthentication" in message or "DefaultCredentialsError" in type(error).__name__:
        hint = "\n\n터미널에서 `gcloud auth application-default login`으로 Vertex AI 인증을 갱신하세요."
    elif "connection" in message.lower() and "5433" in message:
        hint = "\n\n`docker compose up -d`로 PostgreSQL(pgvector)을 먼저 실행하세요."
    elif 'type "vector" does not exist' in message or "extension" in message and "vector" in message:
        hint = "\n\n도커 이미지가 pgvector를 포함하는지 확인하세요 (`pgvector/pgvector:pg17`)."
    elif "langchain_ollama" in message:
        hint = "\n\n`uv add langchain-ollama`로 패키지를 설치하세요."
    elif "11434" in message or "ollama" in message.lower():
        hint = f"\n\n`ollama serve`로 Ollama를 실행하고 `ollama pull {config.OLLAMA_MODEL}`로 모델을 받아 두세요."
    elif "RESOURCE_EXHAUSTED" in message or "429" in message:
        hint = "\n\nVertex AI 할당량을 초과했습니다. 잠시 후 다시 시도하거나 동시 요청 수(RAG_*_CONCURRENCY)를 낮추세요."
    st.error(f"{step} 실패: {error}{hint}")


def preview(text: str, length: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[:length] + " …"
