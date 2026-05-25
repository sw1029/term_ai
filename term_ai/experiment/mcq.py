from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from term_ai.contracts import answer_label, status_reaches


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

    @property
    def label(self) -> str:
        return answer_label(self.answer_idx)

    def query_text(self) -> str:
        parts = [self.task_type, self.word, self.meaning_ko, self.context]
        return " ".join(part for part in parts if part)

    def prompt(self) -> str:
        options = "\n".join(f"{answer_label(idx)}) {option}" for idx, option in enumerate(self.options))
        return (
            "You are a TOEIC business vocabulary expert. "
            "Return the answer letter and confidence as JSON.\n\n"
            f"Task: {self.task_type}\n"
            f"Word: {self.word}\n"
            f"Meaning: {self.meaning_ko}\n"
            f"Context: {self.context}\n\n"
            f"{options}\n\n"
            'Return JSON: {"answer": "A", "confidence": 0.0}'
        )


def load_mcq_items(metadata_path: str | Path, min_status: str = "aug_auto_pass") -> list[MCQItem]:
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
            items.append(
                MCQItem(
                    item_id=str(row.get("item_id") or f"line-{line_no}"),
                    split=str(row.get("split") or "unknown"),
                    task_type=str(payload.get("task_type") or ""),
                    word=str(payload.get("word") or ""),
                    context=str(payload.get("context") or ""),
                    meaning_ko=str(payload.get("meaning_ko") or ""),
                    options=options,
                    answer_idx=answer_idx,
                    teacher_scores=payload.get("teacher_scores"),
                )
            )
    return items


def parse_answer_letter(text: str) -> tuple[str | None, float | None]:
    try:
        data = json.loads(text)
        answer = str(data.get("answer") or data.get("letter") or "").strip().upper()
        confidence = data.get("confidence")
        if answer in {"A", "B", "C", "D"}:
            return answer, float(confidence) if isinstance(confidence, (int, float)) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"(?<![A-Z])([ABCD])(?:\)|\.|번|입니다|$|\s)", text.upper())
    if match:
        return match.group(1), None
    return None, None


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
    }
    if extra:
        row.update(extra)
    return row
