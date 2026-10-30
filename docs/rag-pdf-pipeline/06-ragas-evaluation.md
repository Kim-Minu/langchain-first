# Ragas 표준 지표 평가 결과

기획 [01-plan.md](01-plan.md) · 설계 [02-design.md](02-design.md) · 태스크 [03-tasks.md](03-tasks.md) · 자체 채점 평가 [04-evaluation.md](04-evaluation.md) · 100쪽 평가 [05-evaluation-100p.md](05-evaluation-100p.md)

| 항목 | 내용 |
| --- | --- |
| 스크립트 | [`rag_app/eval_ragas.py`](../../rag_app/eval_ragas.py) |
| 평가 문서 · 컬렉션 | `han-gang_1-20.pdf` → `rag_eval_han_gang` (26청크, 1000/200, 페이지별) |
| 평가 세트 | [`rag_app/eval/han_gang_qa.json`](../../rag_app/eval/han_gang_qa.json) 중 답이 있는 질문 앞 **5문항** (`--limit` 기본값) |
| 모델 | 답변 · Ragas 판정 모두 `gemini-3.8-flash`, 임베딩 `gemini-embedding-2` 768차원, 검색 k=4 |
| 평가일 | 2026-10-30 |
| 결론 | **faithfulness 1.00 · answer_relevancy 0.98 · context_recall 1.00 · context_precision 0.44**. 정답 근거는 매번 top-4 안에 들어오지만 순위가 낮음. 5문항 표본이라 수치 신뢰도는 낮고, precision 평균에는 NaN 1건 · 판정 오판 의심 1건이 섞여 있음 |

> 04 · 05 문서와 같이 책 본문(검색된 청크 원문)은 기록하지 않고 질문 · 정답 · 점수만 남깁니다. 상세 CSV는 `rag_app/eval/results/ragas_rag_eval_han_gang.csv`.

## 1. 실행 전에 고친 것

| 증상 | 원인 | 조치 |
| --- | --- | --- |
| `ModuleNotFoundError: langchain_community.chat_models.vertexai` | ragas 0.4.3(최신)이 langchain-community 0.4에서 제거된 모듈을 import. community 0.3으로 내리면 langchain 1.x와 충돌 | ragas import 전에 빈 `ChatVertexAI` 대체 모듈 등록. ragas 안에서는 n-completion 지원 여부 `isinstance` 체크에만 쓰임 |
| `AttributeError: config.INSTRUCTIONS_DEFAULT` | 존재하지 않는 상수 참조 | 다른 경로와 같은 `config.SYSTEM_PROMPT` 사용 |
| 모든 점수 0 (faithfulness만 1.0) | `--collection` 미지정 시 "최신"이 아니라 **이름순 첫 컬렉션**(`home_pdf`, 다른 문서)을 선택 | 기본값을 `config.EVAL_DOCS[config.EVAL_DOC]`의 질문 세트 · 컬렉션 쌍으로 고정, 없는 컬렉션이면 중단 |

잘못된 컬렉션 결과(`ragas_home_pdf.csv`)는 5문항 모두 "문서에서 찾을 수 없습니다"였습니다. 거절 답변은 지어낸 내용이 없어 faithfulness가 자동으로 1.0이 되므로, **faithfulness는 answer_relevancy와 함께 봐야** 합니다.

## 2. 결과

| 지표 | 값 | 의미 |
| --- | --- | --- |
| faithfulness | 1.00 | 답변이 검색된 문서 안의 내용만 사용 |
| answer_relevancy | 0.98 | 답변이 질문에 정확히 대응 |
| context_recall | 1.00 | 정답에 필요한 정보가 매번 top-4 안에 있음 |
| context_precision | 0.44 | 정답 근거 청크가 상위 순위에 있는 정도 (순위 가중) |

### 문항별

| # | 질문 (요약) | 정답 | faith | rel | prec | rec | 비고 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 센강 · 템스강 강폭 | 200~250m | 1.00 | 0.98 | **NaN** | 1.00 | 판정 실패 → 평균에서 제외 |
| 2 | 서울이 한강을 품고 확장한 두 요인 | 6·25 피난 기억, 인구 유입 | 1.00 | 0.95 | 0.50 | 1.00 | 근거 2위 |
| 3 | 고대 중국 역사서의 한강 명칭 | 대수帶水 | 1.00 | 0.96 | 0.25 | 1.00 | 근거 **4위** |
| 4 | 고구려 · 백제의 한강 명칭 | 아리수, 욱리하 | 1.00 | 0.99 | **0.00** | 1.00 | 답은 맞고 `[1][3]` 인용 — 판정 오판 의심 |
| 5 | 서빙고~동작동 구간 이름 | 동작강 · 동재기강 | 1.00 | 0.99 | 1.00 | 1.00 | 근거 1위 |

precision 평균은 (0.50 + 0.25 + 0.00 + 1.00) / 4 = 0.4375 (NaN 제외 4문항).

### 해석

- **recall 1.0 · precision 0.44**: 검색이 정답을 놓치지는 않지만 순위가 나쁨. k=4라 답변 모델이 뒤쪽 청크까지 읽어 정답을 맞히므로 정답률에는 드러나지 않음. k를 줄이거나 문서가 커지면(100쪽) 놓치는 원인이 됨.
- **같은 개요 청크가 반복 상위**: "한강은 어디에서 와서 어디로 가는가?", "0-5. 남한강의 발원…" 청크가 거의 모든 질문의 top-4에 등장. 넓은 주제를 다루는 청크가 임베딩 유사도를 독식하는 전형 — 04 문서 6장의 "한 청크에 여러 주제가 섞여 임베딩이 흐려짐"과 같은 계열.
- **Q4 0점**: 1위 청크가 단어 중간(`리하郁利河’…`)에서 시작해 "백제 = 욱리하" 연결이 청크 안에 온전하지 않음. 판정 모델이 같은 `gemini-3.8-flash`인 점도 영향 가능.
- **NaN과 0 구분**: `raise_exceptions=False`라 판정 실패는 NaN으로 조용히 평균에서 빠지고, 0은 실제 판정. 문항이 적으면 둘 다 평균을 크게 흔듦.

## 3. 실행 시간 분석

### 문항당 LLM 호출 (ragas 0.4.3 코드 기준)

| 단계 | 호출 | 실행 방식 |
| --- | --- | --- |
| RAG 답변 생성 | 1 | 스크립트 for 루프, 순차 |
| faithfulness | 2 | 진술 추출 → 검증, 순차 |
| answer_relevancy | 1 (n=3) + 임베딩 | 질문 3개 역생성 |
| context_precision | 4 (= k) | 청크마다 1회, `for` 안에서 `await` — 순차 |
| context_recall | 1 | |
| **합계** | **약 9회** | 5문항 ≈ 45회, 20문항 ≈ 180회 |

### 원인

1. **동시 실행 16 → 429 → 이중 재시도 대기 (주원인)**: `evaluate()`에 `RunConfig`를 넘기지 않아 ragas 기본값(`max_workers=16`, `max_retries=10`, `max_wait=60`, `timeout=180`)이 적용됨. langchain 단 재시도(`max_retries=3`, `timeout=120`)와 겹쳐 실제 계산보다 백오프 대기가 시간을 지배. 앱의 평가 화면은 이미 `config.EVAL_CONCURRENCY = 2`("너무 높이면 할당량(429)에 걸림")를 쓰고 있었음. Q1의 NaN도 재시도 소진 가능성이 큼.
2. **context_precision이 k에 비례하는 직렬 경로**: 문항당 가장 긴 순차 구간.
3. **추론 모델 지연**: `reasoning_effort="low"`여도 추론 토큰을 먼저 생성하고, 판정은 JSON 구조화 출력이라 호출당 무거움.

### 측정 시도

2문항 프로파일(단계별 시간 · 호출 수 · 지연 기록)을 두 번 실행했으나, 두 번 모두 **동시 호출 없는 첫 단일 답변 호출에서 `429 RESOURCE_EXHAUSTED`**로 약 8초 만에 중단. 직전 평가 실행이 할당량을 소진한 것으로 보이며, 단일 호출도 계속 거절되는 점은 분당 한도 초과 이상(일일 한도 · 다른 작업과 공유)일 가능성도 있음. **전후 실행 시간은 아직 측정하지 못함.**

### 조치

```python
results = evaluate(
    ...,
    run_config=RunConfig(
        max_workers=config.EVAL_CONCURRENCY,  # 16 → 2
        max_retries=3,                        # 10 → 3
        max_wait=30,                          # 60 → 30
        timeout=120,                          # 180 → 120
    ),
)
```

동시성을 올려도 할당량(RPM/TPM)이 상한이라 처리량은 늘지 않고 429 · 백오프만 늘며, 할당량을 소진해 다음 실행까지 느려짐.

## 4. 남은 과제

| 과제 | 상태 |
| --- | --- |
| `RunConfig` 적용 후 실행 시간 전후 비교 | 할당량 회복 대기 — 2026-10-30 재확인 시 `gemini-3.8-flash`는 단일 호출도 429, `gemini-3.1-pro-preview`는 정상 (할당량이 모델별로 분리) |
| 답변 생성 루프 429 대응 — `evaluate()` 밖이라 `RunConfig` 미적용, 429 시 스크립트 전체 중단 | **적용** — 임베딩 · 답변 호출을 `evaluation._with_rate_limit_retry`(10 · 30 · 60초 대기, 429 외 오류는 즉시 전파)로 감쌈 |
| 전체 세트 평가 (`--limit 0`) — 5문항은 표본 부족 | 미실행 (답변 모델 할당량 대기) |
| Ragas 판정 모델을 `config.JUDGE_MODEL`(`gemini-3.1-pro-preview`)로 분리 | **적용** — 다음 실행부터 판정은 pro, 답변은 flash. 2장 결과(같은 모델 판정)와는 직접 비교 불가 |
| context_precision 개선 실험 (청크 500/100, k 조정, reranker) | 미실행 |
| 공백 포함 이름의 찌꺼기 컬렉션 정리 (`rag_eval_across 1000 200 4 /private/tmp/...` 등) | 미정리 |

## 5. 재현 방법

```bash
# 기본: rag_eval_han_gang 컬렉션 + han_gang_qa.json 앞 5문항
uv run python -m rag_app.eval_ragas

# 전체 문항
uv run python -m rag_app.eval_ragas --limit 0

# 다른 평가 문서 (질문 세트와 컬렉션을 함께 지정)
uv run python -m rag_app.eval_ragas --qa-file rag_app/eval/han_gang_100_qa.json --collection rag_eval_han_gang_100
```

`--qa-file`만 바꾸면 컬렉션은 기본값 그대로이므로 짝을 맞춰 지정해야 합니다.

## 6. 한계와 주의

- 5문항 결과라 지표 하나가 문항 1개로 20~25%p 움직임.
- 답변 생성과 판정이 같은 모델이라 자기 채점 편향 가능.
- ragas 0.4.3은 `ragas.metrics` 직접 import, `LangchainLLMWrapper` · `LangchainEmbeddingsWrapper`를 deprecated로 표시 (v1.0에서 제거 예정). 현재 동작에는 영향 없음.
- `ChatVertexAI` 대체 모듈은 ragas · langchain-community 버전이 바뀌면 다시 확인 필요 (`try` 가드로 모듈이 생기면 자동으로 건너뜀).
