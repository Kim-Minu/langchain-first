"""Ragas를 이용한 RAG 파이프라인 종합 평가 스크립트.

기존 rag_app 파이프라인(Gemini LLM, Google 임베딩, pgvector)과
eval/han_gang_qa.json 골든 데이터셋을 연동하여 Ragas 표준 지표를 측정합니다.
"""

import argparse
import json
import sys
import types
from pathlib import Path

import pandas as pd
from datasets import Dataset

# ragas 0.4.3은 langchain-community 0.4에서 제거된 chat_models.vertexai를 import한다.
# ragas 내부에서 isinstance 체크에만 쓰이므로 빈 클래스로 대체 모듈을 등록한다.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

# Ragas 메트릭 및 평가자 래퍼
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from rag_app import config
from rag_app import pipeline as rag
from rag_app.evaluation import _with_rate_limit_retry


# 질문 세트와 컬렉션은 짝이 맞아야 하므로 config의 평가 문서 설정을 기본값으로 쓴다
DEFAULT_EVAL_DOC = config.EVAL_DOCS[config.EVAL_DOC]


def run_ragas_evaluation(
    qa_path: Path = DEFAULT_EVAL_DOC.qa_file,
    collection_name: str = DEFAULT_EVAL_DOC.collection,
    top_k: int = config.TOP_K,
    limit: int | None = 5,
    output_dir: Path = Path("rag_app/eval/results"),
) -> pd.DataFrame | None:
    """Ragas 평가를 실행하고 결과를 CSV로 저장합니다."""
    # 1. 컬렉션 확인
    collections = rag.list_stored_collections()
    if not collections:
        print("[경고] pgvector DB에 인덱싱된 컬렉션이 없습니다. 먼저 문서를 업로드해 인덱싱해 주세요.")
        return None

    if collection_name not in {c["name"] for c in collections}:
        print(f"[오류] 컬렉션 '{collection_name}'을 찾을 수 없습니다. (사용 가능: {[c['name'] for c in collections]})")
        return None
    print(f"[*] 평가 컬렉션: '{collection_name}' / 질문 세트: {qa_path}")

    # 2. 모델 및 벡터 스토어 초기화
    print(f"[*] 답변 LLM: {config.LLM_MODEL} / 판정 LLM: {config.JUDGE_MODEL}")
    llm = rag.create_llm(config.LLM_MODEL)
    embeddings = rag.create_embeddings(config.EMBEDDING_MODEL, config.EMBEDDING_DIMENSION)
    store = rag.get_vector_store(embeddings, collection_name)

    # Ragas 평가자 래핑 (기본 OpenAI 대신 Vertex Gemini 사용)
    # 답변 모델이 자기 답을 채점하지 않도록 평가 화면과 같은 별도 판정 모델을 쓴다
    ragas_llm = LangchainLLMWrapper(rag.create_llm(config.JUDGE_MODEL))
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # 3. QA 데이터셋 로드
    if not qa_path.exists():
        print(f"[오류] QA 데이터셋 파일을 찾을 수 없습니다: {qa_path}")
        return None

    qa_items = json.loads(qa_path.read_text(encoding="utf-8"))
    # Ragas 평가를 위해 답이 있는 질문(answerable == True)만 필터링
    eval_items = [item for item in qa_items if item.get("answerable", True)]
    if limit:
        eval_items = eval_items[:limit]

    print(f"[*] 총 {len(eval_items)}개 문항에 대해 RAG 답변 생성 및 문맥 수집을 시작합니다.")

    questions = []
    responses = []
    retrieved_contexts = []
    references = []

    for idx, item in enumerate(eval_items, start=1):
        q = item["question"]
        gold_answer = item["answer"]
        print(f"  [{idx}/{len(eval_items)}] 질문: {q}")

        # (1) 검색 수행 (pgvector)
        # evaluate() 밖이라 RunConfig 재시도가 닿지 않으므로 429는 여기서 기다렸다 다시 시도
        query_vec = _with_rate_limit_retry(lambda: rag.embed_question(embeddings, q))
        hits = rag.search_chunks(store, query_vec, k=top_k)
        docs = [doc for doc, _ in hits]

        # (2) 답변 생성
        prompt = rag.build_prompt(config.SYSTEM_PROMPT, docs, q)
        reply = _with_rate_limit_retry(lambda: rag.generate_answer(llm, prompt))

        # (3) Ragas 포맷 데이터 수집
        questions.append(q)
        responses.append(reply.text)
        retrieved_contexts.append([doc.page_content for doc in docs])
        references.append(gold_answer)

    # 4. Dataset 변환
    eval_dataset = Dataset.from_dict({
        "user_input": questions,
        "response": responses,
        "retrieved_contexts": retrieved_contexts,
        "reference": references,
    })

    # 5. Ragas 지표 평가
    print("\n[*] Ragas 평가 실행 중 (Faithfulness, Answer Relevancy, Context Precision, Context Recall)...")
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
        # 기본값(동시 16, 재시도 10)은 Vertex 할당량(429)에 걸려 백오프 대기로 오래 걸림
        run_config=RunConfig(
            max_workers=config.EVAL_CONCURRENCY,
            max_retries=3,
            max_wait=30,
            timeout=120,
        ),
    )

    print("\n" + "=" * 50)
    print("           [Ragas 평가 결과 요약]           ")
    print("=" * 50)
    print(results)
    print("=" * 50)

    # 6. 결과 저장
    df = results.to_pandas()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"ragas_{collection_name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[완료] 상세 평가 결과가 저장되었습니다: {csv_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 파이프라인 Ragas 평가 실행 스크립트")
    parser.add_argument(
        "--qa-file",
        type=Path,
        default=DEFAULT_EVAL_DOC.qa_file,
        help=f"평가용 QA JSON 파일 경로 (기본값: {config.EVAL_DOC} 질문 세트)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_EVAL_DOC.collection,
        help=f"평가할 pgvector 컬렉션 이름 (기본값: {DEFAULT_EVAL_DOC.collection})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K,
        help="검색할 청크 개수 (기본값: config.TOP_K)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="평가할 문항 수 (전체 평가는 0 또는 음수 지정, 기본값: 5)",
    )

    args = parser.parse_args()
    limit = None if args.limit <= 0 else args.limit

    run_ragas_evaluation(
        qa_path=args.qa_file,
        collection_name=args.collection,
        top_k=args.top_k,
        limit=limit,
    )

