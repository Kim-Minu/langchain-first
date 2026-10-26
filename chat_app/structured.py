"""구조화된 출력: 원샷 예시 → 출력 형식(OutputFormat) (Streamlit에 의존하지 않는 순수 로직).

원샷 예시는 두 가지 형태를 받습니다.
- JSON 예시   : {"summary": "...", "key_points": ["..."]}  → 같은 구조의 JSON으로 답변
- 텍스트 템플릿: "뉴스 제목 : \n기자 이름 : \n뉴스 내용 요약 :" → 같은 "라벨 : 값" 텍스트로 답변

어느 쪽이든 예시에서 JSON Schema를 만들어 create_agent(response_format=ProviderStrategy(schema))로
형식을 강제하고, 예시 자체는 시스템 프롬프트에 원샷으로 넣어 각 항목을 무엇으로 채울지 알려줍니다.
텍스트 템플릿은 라벨마다 field_1, field_2 … 키를 만들어 JSON으로 받은 뒤, 원래 라벨 순서대로 텍스트로 되돌립니다.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

DEFAULT_STRUCTURED_EXAMPLE = """뉴스 제목 :
기자 이름 :
뉴스 내용 요약 :"""

SCHEMA_TITLE = "StructuredAnswer"
_LABEL_SEPARATOR = re.compile(r"\s*[:：]\s*")


class StructuredExampleError(ValueError):
    """원샷 예시가 구조화된 출력에 쓸 수 없는 형태일 때."""


# ------------------------------------------------------------------------------
# JSON 예시 → JSON Schema
# ------------------------------------------------------------------------------
def _infer(value: Any, path: str) -> dict:
    if isinstance(value, bool):  # bool은 int의 하위 클래스이므로 먼저 검사
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if value is None:
        return {"type": ["string", "null"]}
    if isinstance(value, list):
        if not value:
            raise StructuredExampleError(f"'{path}' 배열이 비어 있어 항목 형식을 알 수 없습니다. 예시 항목을 하나 넣어주세요")
        return {"type": "array", "items": _infer(value[0], f"{path}[0]")}
    if isinstance(value, dict):
        if not value:
            raise StructuredExampleError(f"'{path}' 객체가 비어 있습니다. 필드를 하나 이상 넣어주세요")
        return {
            "type": "object",
            "properties": {k: _infer(v, f"{path}.{k}") for k, v in value.items()},
            "required": list(value),
            "additionalProperties": False,
        }
    raise StructuredExampleError(f"'{path}'의 값 형식({type(value).__name__})은 지원하지 않습니다")


def infer_json_schema(example: dict) -> dict:
    """원샷 JSON 예시와 같은 구조(필드 이름 · 타입 · 중첩)를 강제하는 JSON Schema. 모든 필드는 필수."""
    schema = _infer(example, "$")
    return {"title": SCHEMA_TITLE, "description": "사용자가 제시한 원샷 예시와 같은 구조의 답변", **schema}


# ------------------------------------------------------------------------------
# 출력 형식
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class OutputFormat:
    kind: Literal["json", "text"]
    json_example: dict | None = None  # kind="json"
    labels: tuple[str, ...] = ()  # kind="text": 항목 라벨 (순서 유지)
    hints: tuple[str, ...] = ()  # kind="text": 라벨 옆에 적은 예시 값 (비어 있을 수 있음)

    @staticmethod
    def _field_key(index: int) -> str:
        return f"field_{index + 1}"

    def json_schema(self) -> dict:
        if self.kind == "json":
            return infer_json_schema(self.json_example)
        properties = {}
        for i, (label, hint) in enumerate(zip(self.labels, self.hints)):
            description = label if not hint else f"{label} (예: {hint})"
            properties[self._field_key(i)] = {"type": "string", "description": description}
        return {
            "title": SCHEMA_TITLE,
            "description": "사용자가 제시한 텍스트 템플릿의 각 항목 값",
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    def system_prompt(self) -> str:
        """시스템 프롬프트에 덧붙일 원샷 지시문."""
        if self.kind == "json":
            example_json = json.dumps(self.json_example, ensure_ascii=False, indent=2)
            return (
                "\n\n[출력 형식]\n"
                "모든 답변은 아래 예시와 똑같은 구조의 JSON 객체 하나로만 출력하세요. "
                "필드 이름과 중첩 구조, 값의 타입을 예시와 동일하게 유지하고, 값은 질문에 맞는 실제 내용으로 채우세요. "
                "JSON 바깥에 설명이나 마크다운 코드 블록을 덧붙이지 마세요.\n"
                f"예시:\n{example_json}"
            )
        template = "\n".join(f"{label} : {hint}".rstrip() for label, hint in zip(self.labels, self.hints))
        mapping = "\n".join(f"- {self._field_key(i)}: {label}" for i, label in enumerate(self.labels))
        return (
            "\n\n[출력 형식]\n"
            "모든 답변은 아래 템플릿의 각 항목을 채우는 형태로만 작성하세요. "
            "출력은 JSON 객체이며, 각 키가 뜻하는 템플릿 항목은 다음과 같습니다.\n"
            f"{mapping}\n"
            "값은 마크다운 없이 자연스러운 한국어 일반 텍스트로 쓰고, 템플릿에 예시 값이 있으면 그 길이와 어투를 따르세요. "
            "내용을 확인할 수 없는 항목은 지어내지 말고 '확인 불가'라고 쓰세요.\n"
            f"템플릿:\n{template}"
        )

    def canonical(self) -> str:
        """에이전트 재생성 · 캐시 키 비교용 정규화 문자열."""
        if self.kind == "json":
            return "json:" + json.dumps(self.json_example, ensure_ascii=False, sort_keys=True)
        return "text:" + json.dumps(list(zip(self.labels, self.hints)), ensure_ascii=False)

    def describe(self) -> str:
        """설정 패널에 보여줄 한 줄 설명."""
        if self.kind == "json":
            return f"JSON 형식 · 최상위 필드 {len(self.json_example)}개: {', '.join(self.json_example)}"
        return f"텍스트 형식 · 항목 {len(self.labels)}개: {', '.join(self.labels)}"

    def render(self, response_text: str) -> tuple[str, bool]:
        """모델 응답(JSON 문자열)을 화면 · 저장용 마크다운으로 변환합니다. (마크다운, 형식대로 해석 성공 여부)"""
        text = response_text.strip()
        if text.startswith("```"):  # 모델이 코드 블록으로 감싼 경우 벗겨냄
            text = text.strip("`").removeprefix("json").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return response_text, False

        if self.kind == "json":
            return f"```json\n{json.dumps(parsed, ensure_ascii=False, indent=2)}\n```", True

        if not isinstance(parsed, dict):
            return response_text, False
        lines = [
            f"**{label}** : {str(parsed.get(self._field_key(i), '')).strip()}" for i, label in enumerate(self.labels)
        ]
        return "\n\n".join(lines), all(self._field_key(i) in parsed for i in range(len(self.labels)))


# ------------------------------------------------------------------------------
# 원샷 예시 파싱 (JSON / 텍스트 템플릿 자동 판별)
# ------------------------------------------------------------------------------
def _parse_text_template(text: str) -> OutputFormat:
    labels, hints = [], []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = _LABEL_SEPARATOR.split(line, maxsplit=1)
        if len(parts) != 2 or not parts[0]:
            raise StructuredExampleError(f"{line_no}행이 '라벨 : 값' 형식이 아닙니다 → {line!r}")
        label, hint = parts[0].strip(), parts[1].strip()
        if label in labels:
            raise StructuredExampleError(f"'{label}' 항목이 중복되었습니다")
        labels.append(label)
        hints.append(hint)
    if not labels:
        raise StructuredExampleError("항목이 없습니다. 한 줄에 하나씩 '라벨 : 값' 형식으로 적어주세요")
    return OutputFormat(kind="text", labels=tuple(labels), hints=tuple(hints))


def parse_output_format(text: str) -> OutputFormat:
    """원샷 예시를 해석합니다. '{'로 시작하면 JSON 예시, 아니면 '라벨 : 값' 텍스트 템플릿으로 봅니다."""
    stripped = text.strip()
    if not stripped:
        raise StructuredExampleError("원샷 예시를 입력해 주세요")
    if not stripped.startswith(("{", "[")):
        return _parse_text_template(stripped)

    try:
        example = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise StructuredExampleError(f"JSON 형식이 아닙니다 ({e.msg}, {e.lineno}행 {e.colno}열)") from e
    if not isinstance(example, dict) or not example:
        raise StructuredExampleError("JSON 예시의 최상위는 필드가 하나 이상 있는 객체({...})여야 합니다")
    infer_json_schema(example)  # 빈 배열 등 추론 불가능한 값 검사
    return OutputFormat(kind="json", json_example=example)
