from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from term_ai.contracts import APPROVED_AUG_STATUS, SYSTEM_PROMPT, answer_label, status_reaches


PARSER_VERSION = "mcq_answer_parser_v2"
PROMPT_CONTRACT_VERSION = "mcq_answer_json_v2"


@dataclass(frozen=True)
class ParseResult:
    answer: str | None
    confidence: float | None = None
    parse_method: str = "unparsed"
    parser_version: str = PARSER_VERSION
    strict_parse_error: bool = True
    confidence_normalized_from_percent: bool = False

    @property
    def parse_error(self) -> bool:
        return self.answer is None


@dataclass(frozen=True)
class MCQItem:
    item_id: str
    split: str
    task_type: str
    word: str
    context: str
    meaning_ko: str
    options: list[str]
    answer_idx: int
    teacher_scores: list[float] | None = None
    status: str = ""
    source: str = ""
    dataset_view: str = ""
    stress_tags: tuple[str, ...] = ()
    embedding_top2_similarity: float | None = None
    embedding_top2_gap: float | None = None

    @property
    def label(self) -> str:
        return answer_label(self.answer_idx)

    def query_text(self) -> str:
        parts = [self.task_type, self.word, self.meaning_ko, self.context]
        return " ".join(part for part in parts if part)

    def prompt(self) -> str:
        return "\n\n".join(message["content"] for message in self.prompt_messages())

    def prompt_messages(self) -> list[dict[str, str]]:
        options = "\n".join(f"{answer_label(idx)}) {option}" for idx, option in enumerate(self.options))
        system = (
            f"{SYSTEM_PROMPT} Return one valid JSON object first. "
            'Schema: {"answer": "A|B|C|D", "confidence": 0.0, '
            '"distribution": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}, '
            '"rationale": "optional short reason"}. '
            "Do not put prose before the JSON object."
        )
        user = (
            f"Task: {self.task_type}\n"
            f"Word: {self.word}\n"
            f"Meaning: {self.meaning_ko}\n"
            f"Context: {self.context}\n\n"
            f"{options}\n\n"
            'Return JSON first, for example: {"answer": "A", "confidence": 0.0}'
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_mcq_items(metadata_path: str | Path, min_status: str = APPROVED_AUG_STATUS) -> list[MCQItem]:
    items: list[MCQItem] = []
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not status_reaches(row.get("status", ""), min_status):
                continue
            payload: dict[str, Any] = row.get("payload") or {}
            options = [str(option) for option in payload.get("options") or []]
            answer_idx = payload.get("answer_idx")
            if len(options) != 4 or not isinstance(answer_idx, int):
                raise ValueError(f"metadata line {line_no} is not a 4-option MCQ")
            task_type = str(payload.get("source_task_type") or payload.get("task_type") or "")
            teacher_scores = payload.get("teacher_scores") or row.get("teacher_scores")
            stress_tags = tuple(str(tag) for tag in row.get("stress_tags") or payload.get("stress_tags") or [])
            top2_similarity = payload.get("embedding_top2_similarity")
            top2_gap = payload.get("embedding_top2_gap")
            items.append(
                MCQItem(
                    item_id=str(row.get("item_id") or f"line-{line_no}"),
                    split=str(row.get("split") or "unknown"),
                    task_type=task_type,
                    word=str(payload.get("word") or ""),
                    context=str(payload.get("context") or ""),
                    meaning_ko=str(payload.get("meaning_ko") or ""),
                    options=options,
                    answer_idx=answer_idx,
                    teacher_scores=teacher_scores,
                    status=str(row.get("status") or ""),
                    source=str(row.get("source") or ""),
                    dataset_view=str(row.get("dataset_view") or ""),
                    stress_tags=stress_tags,
                    embedding_top2_similarity=(
                        float(top2_similarity) if isinstance(top2_similarity, (int, float)) else None
                    ),
                    embedding_top2_gap=float(top2_gap) if isinstance(top2_gap, (int, float)) else None,
                )
            )
    return items


def _normalize_confidence(value: Any) -> tuple[float | None, bool]:
    if isinstance(value, str):
        stripped = value.strip().rstrip("%")
        try:
            value = float(stripped)
        except ValueError:
            return None, False
    if not isinstance(value, (int, float)):
        return None, False
    confidence = float(value)
    if 0.0 <= confidence <= 1.0:
        return confidence, False
    if 1.0 < confidence <= 100.0:
        return confidence / 100.0, True
    return None, False


def _strict_json_parse(text: str) -> tuple[str | None, float | None, bool, bool]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, None, True, False
    if not isinstance(data, dict):
        return None, None, True, False
    answer = str(data.get("answer") or data.get("letter") or "").strip().upper()
    confidence, normalized = _normalize_confidence(data.get("confidence"))
    if answer in {"A", "B", "C", "D"}:
        return answer, confidence, False, normalized
    return None, None, True, False


def _answer_from_json(data: Any) -> tuple[str | None, float | None, bool]:
    if not isinstance(data, dict):
        return None, None, False
    answer = str(data.get("answer") or data.get("letter") or "").strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        return None, None, False
    confidence, normalized = _normalize_confidence(data.get("confidence"))
    return answer, confidence, normalized


def _iter_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def _parse_json_candidates(text: str) -> tuple[str | None, float | None, bool, str | None]:
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        for candidate in _iter_json_objects(match.group(1)) or [match.group(1).strip()]:
            try:
                answer, confidence, normalized = _answer_from_json(json.loads(candidate))
            except json.JSONDecodeError:
                continue
            if answer:
                return answer, confidence, normalized, "fenced_json"
    for candidate in _iter_json_objects(text):
        try:
            answer, confidence, normalized = _answer_from_json(json.loads(candidate))
        except json.JSONDecodeError:
            continue
        if answer:
            return answer, confidence, normalized, "embedded_json"
    return None, None, False, None


def _confidence_from_text(text: str) -> tuple[float | None, bool]:
    patterns = [
        r"confidence(?:\s+score)?\"?\s*(?:[:=]|is|of)\s*([0-9]+(?:\.[0-9]+)?%?)",
        r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:confidence|confident|확신|신뢰)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _normalize_confidence(match.group(1))
    return None, False


def _parse_answer_key(text: str) -> tuple[str | None, float | None, bool]:
    match = re.search(r'"(?:answer|letter)"\s*:\s*["\']?([ABCD])(?=$|["\'},\]\s])', text, flags=re.IGNORECASE)
    if not match:
        return None, None, False
    confidence, normalized = _confidence_from_text(text)
    return match.group(1).upper(), confidence, normalized


def _parse_marked_text_answer(text: str) -> tuple[str | None, float | None, bool, str | None]:
    patterns = [
        (
            r"\b(?:final\s+answer|correct\s+answer|answer)\s*"
            r"(?:[:=\-]|\bis\b|\bwould\s+be\b|\bshould\s+be\b)\s*"
            r"(?:option\s+)?[*_`'\"]*([ABCD])(?=$|[^A-Z])",
            "english_answer",
        ),
        (
            r"\boption\s+[*_`'\"]*([ABCD])[*_`'\"]*\s+"
            r"(?:is|would\s+be|should\s+be|seems)\s+"
            r"(?:clearly\s+)?(?:the\s+)?(?:correct|best|most\s+appropriate)\b",
            "english_answer",
        ),
        (
            r"(?<![A-Z])([ABCD])\s+(?:is|would\s+be|should\s+be)\s+"
            r"(?:the\s+)?(?:correct\s+)?(?:answer|choice|option)\b",
            "english_answer",
        ),
        (r"정답\s*(?:은|:|=)?\s*['\"]?([ABCD])(?=$|[^A-Z])", "korean_answer"),
        (r"(?<![A-Z])([ABCD])\s*(?:가|이)?\s*정답", "korean_answer"),
        (r"(?<![A-Z])['\"]?([ABCD])['\"]?\s*(?:가|이)?\s*(?:정확합니다|정확하다|맞습니다|맞다)", "korean_answer"),
        (r"['\"]([ABCD])['\"]\s*에\s*해당합니다", "korean_answer"),
    ]
    for pattern, method in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            confidence, normalized = _confidence_from_text(text)
            return match.group(1).upper(), confidence, normalized, method
    return None, None, False, None


def _parse_option_line_answer(text: str) -> tuple[str | None, float | None, bool]:
    positive_marker = re.compile(
        r"\b("
        r"correct\s+answer|"
        r"is\s+(?:the\s+)?correct\b|"
        r"most\s+accurate|"
        r"most\s+appropriate|"
        r"best\s+(?:answer|choice|option)"
        r")\b",
        flags=re.IGNORECASE,
    )
    for line in (line.strip() for line in text.splitlines() if line.strip()):
        if not positive_marker.search(line):
            continue
        option_markers = re.findall(r"(?<![A-Z])[*_`'\"]*([ABCD])\s*\)", line, flags=re.IGNORECASE)
        unique = {marker.upper() for marker in option_markers}
        if len(unique) == 1:
            confidence, normalized = _confidence_from_text(text)
            return unique.pop(), confidence, normalized
    return None, None, False


def _parse_first_line_answer(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    option_like_lines = [line for line in lines[:6] if re.match(r"^[ABCD](?:\)|\.)\s+\S", line, re.IGNORECASE)]
    if len(option_like_lines) >= 2:
        return None
    first = lines[0]
    if len(first) > 80:
        return None
    match = re.match(r"^([ABCD])(?:\)|\.|$)(?:\s|$)", first, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def parse_answer_response(text: str) -> ParseResult:
    strict_answer, strict_confidence, strict_error, strict_normalized = _strict_json_parse(text)
    if strict_answer:
        return ParseResult(
            strict_answer,
            strict_confidence,
            parse_method="strict_json",
            strict_parse_error=False,
            confidence_normalized_from_percent=strict_normalized,
        )

    answer, confidence, normalized, method = _parse_json_candidates(text)
    if answer:
        return ParseResult(
            answer,
            confidence,
            parse_method=method or "embedded_json",
            strict_parse_error=strict_error,
            confidence_normalized_from_percent=normalized,
        )

    answer, confidence, normalized = _parse_answer_key(text)
    if answer:
        return ParseResult(
            answer,
            confidence,
            parse_method="answer_key",
            strict_parse_error=strict_error,
            confidence_normalized_from_percent=normalized,
        )

    answer, confidence, normalized, method = _parse_marked_text_answer(text)
    if answer:
        return ParseResult(
            answer,
            confidence,
            parse_method=method or "marked_answer",
            strict_parse_error=strict_error,
            confidence_normalized_from_percent=normalized,
        )

    answer, confidence, normalized = _parse_option_line_answer(text)
    if answer:
        return ParseResult(
            answer,
            confidence,
            parse_method="option_line_answer",
            strict_parse_error=strict_error,
            confidence_normalized_from_percent=normalized,
        )

    first_line_answer = _parse_first_line_answer(text)
    if first_line_answer:
        return ParseResult(
            first_line_answer,
            None,
            parse_method="first_line_answer",
            strict_parse_error=strict_error,
        )

    return ParseResult(None, None, strict_parse_error=strict_error)


def parse_answer_letter(text: str) -> tuple[str | None, float | None]:
    result = parse_answer_response(text)
    return result.answer, result.confidence


def prediction_row(
    item: MCQItem,
    prediction: str,
    confidence: float,
    latency_ms: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "item_id": item.item_id,
        "split": item.split,
        "task_type": item.task_type,
        "label": item.label,
        "prediction": prediction,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "status": item.status,
        "source": item.source,
        "dataset_view": item.dataset_view,
        "stress_tags": list(item.stress_tags),
    }
    if item.embedding_top2_similarity is not None:
        row["embedding_top2_similarity"] = item.embedding_top2_similarity
    if item.embedding_top2_gap is not None:
        row["embedding_top2_gap"] = item.embedding_top2_gap
    if extra:
        row.update(extra)
    return row
