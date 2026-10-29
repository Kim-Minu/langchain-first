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


def render_sidebar(page: str = "main") -> Settings:
    """사이드바 설정. page='eval'이면 이미지 읽기 대신 OCR 프롬프트를, 컬렉션은 평가용 이름을 보여 줌."""
    for key, default in DEFAULTS.items():
        st.session_state[key] = st.session_state.get(key, default)

    with st.sidebar:
        # 프로젝트 공통 .streamlit/config.toml이 자동 페이지 목록을 숨기므로(web_chat.py용 showSidebarNavigation = false)
        # 두 화면 사이 이동 링크를 직접 둠
        st.page_link("app.py", label="PDF RAG 실습", icon="📄")
        st.page_link("pages/ocr_eval.py", label="OCR · RAG 평가", icon="🔎")
        st.page_link("pages/benchmark.py", label="LLM 성능 벤치마크", icon="⚡")
        st.page_link("pages/billing_dashboard.py", label="과금 대시보드", icon="💰")
        st.divider()

        # 실시간 과금 카운터 위젯 (모든 페이지 공통)
        billing.render_billing_counter_widget()
        st.divider()

        st.header("설정")
        st.caption("두 화면(PDF RAG 실습 · OCR · RAG 평가)이 같은 설정을 공유합니다.")

        st.subheader("A. 인덱싱")
        if page == "eval":
            st.selectbox(
                "평가 문서", list(config.EVAL_DOCS), format_func=lambda key: config.EVAL_DOCS[key].label,
                key="cfg_eval_doc", on_change=_follow_eval_doc,
                help="고르면 PDF · 질문 세트 · 컬렉션이 함께 바뀜. 컬렉션만 다른 문서 것으로 바꾸면 "
                     "같은 질문을 더 큰 인덱스에서 평가할 수 있음 (예: 1~20쪽 질문을 1~100쪽 컬렉션에서)",
            )
            st.text_area(
                "OCR 프롬프트", key="cfg_ocr_prompt", height=220,
                help="스캔한 페이지마다 Gemini에 이 지시와 함께 보냄. 바꾸면 OCR을 다시 실행 (결과는 프롬프트별로 캐시)",
            )
        else:
            st.radio(
                "PDF 속 이미지", list(config.IMAGE_OPTIONS), format_func=config.IMAGE_OPTIONS.get, key="cfg_image_mode",
                help="스캔 PDF는 페이지 전체가 이미지라 끄면 텍스트가 0자. 켜면 이미지마다 Gemini(아래 B의 Gemini 모델 ID)를 한 번씩 호출",
            )
            if st.session_state.cfg_image_mode == config.IMAGE_GEMINI:
                st.text_area(
                    "이미지 읽기 프롬프트", key="cfg_image_prompt", height=200,
                    help="이미지마다 Gemini에 이 지시와 함께 보냄. 바꾸면 다시 인덱싱해야 반영됨",
                )
        st.number_input(
            "청크 크기 (글자)", config.CHUNK_SIZE_MIN, config.CHUNK_SIZE_MAX, step=100, key="cfg_chunk_size",
            help="청크 하나의 최대 글자 수. 작을수록 검색이 정밀하지만 문맥이 끊기기 쉬움",
        )
        st.number_input(
            "청크 겹침 (글자)", config.CHUNK_OVERLAP_MIN, config.CHUNK_OVERLAP_MAX, step=50, key="cfg_chunk_overlap",
            help="이웃한 청크가 공유하는 글자 수. 경계에서 문장이 잘려도 앞뒤 문맥을 보존",
        )
        st.checkbox(
            "페이지를 넘어 이어서 청크 나누기", key="cfg_across_pages",
            help="페이지를 이어 붙인 뒤 나눠, 페이지 경계에서 끊긴 문장도 한 청크에 들어가게 함 (청크에 page ~ page_end 기록)",
        )
        st.selectbox(
            "임베딩 모델",
            list(config.EMBEDDING_OPTIONS),
            format_func=config.EMBEDDING_OPTIONS.get,
            key="cfg_embedding_model",
            help="문서 청크와 질문을 벡터로 변환하는 모델 (Vertex AI global)",
        )
        st.selectbox(
            "임베딩 차원", config.EMBEDDING_DIMENSION_OPTIONS, key="cfg_embedding_dim",
            help="바꾸면 다시 인덱싱해야 함 (같은 컬렉션 안의 벡터는 차원이 같아야 비교 가능)",
        )
        collection_key = "cfg_eval_collection" if page == "eval" else "cfg_collection"
        st.text_input("컬렉션 이름 (pgvector)", key=collection_key)

        st.subheader("B. 질문 · 답변")
        st.slider("검색할 청크 수 (top-k)", config.TOP_K_MIN, config.TOP_K_MAX, key="cfg_top_k")
        st.radio(
            "답변 모델", list(config.LLM_OPTIONS), format_func=config.LLM_OPTIONS.get, captions=config.LLM_CAPTIONS,
            key="cfg_llm_choice",
        )
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
        st.text_area("지시문 (시스템 프롬프트)", key="cfg_instructions", height=200)
        if page == "eval":
            st.subheader("C. 채점")
            st.text_input(
                "채점 모델 (Gemini)", key="cfg_judge_model",
                help="정답 채점 · 근거 충실도 판정에 사용. 답변 모델과 같으면 자기 답에 후한 편향이 생길 수 있어 다른 모델 권장",
            )

    s = st.session_state
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
        st.sidebar.error("청크 겹침은 청크 크기보다 작아야 합니다.")
    return settings


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
