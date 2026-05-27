from __future__ import annotations

import json
from typing import Any

from term_ai.contracts import (
    SYSTEM_PROMPT,
    TASK_ANTONYM,
    TASK_CONTEXT_CLOZE,
    TASK_SENSE_DISAMBIGUATION,
    TASK_SYNONYM,
    answer_label,
    ensure_task_type,
    make_sft_record,
)


def _format_options(options: list[str]) -> str:
    return "\n".join(f"{answer_label(idx)}) {option}" for idx, option in enumerate(options))


def _answer_text(options: list[str], answer_idx: int, rationale: str) -> str:
    label = answer_label(answer_idx)
    return f"{label}) {options[answer_idx]}\n\n{rationale.strip()}"


def _json_answer_text(answer_idx: int, rationale: str, confidence: float = 1.0) -> str:
    return json.dumps(
        {
            "answer": answer_label(answer_idx),
            "confidence": confidence,
            "rationale": rationale.strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def candidate_payload_to_sft_record(payload: dict[str, Any], response_format: str = "json_answer") -> dict[str, Any]:
    if response_format not in {"json_answer", "letter_reason"}:
        raise ValueError("response_format must be json_answer or letter_reason")
    task_type = ensure_task_type(str(payload.get("task_type", "")))
    word = str(payload["word"]).strip()
    options = [str(option).strip() for option in payload["options"]]
    answer_idx = int(payload["answer_idx"])
    rationale = str(payload.get("rationale") or payload.get("teacher_rationale") or "").strip()
    if not rationale:
        raise ValueError("candidate payload requires rationale before SFT conversion")

    if task_type == TASK_SYNONYM:
        context = str(payload.get("context") or "").strip()
        context_block = f"\nContext: {context}" if context else ""
        user = (
            "Task: Synonym Selection\n\n"
            "다음 단어와 가장 의미가 가까운 것을 고르시오.\n\n"
            f"Word: {word}{context_block}\n\n"
            f"{_format_options(options)}"
        )
    elif task_type == TASK_ANTONYM:
        meaning = str(payload.get("meaning_ko") or "").strip()
        meaning_part = f" ({meaning})" if meaning else ""
        user = (
            "Task: Antonym Selection\n\n"
            "다음 단어의 반대 의미에 가장 가까운 것을 고르시오.\n\n"
            f"Word: {word}{meaning_part}\n\n"
            f"{_format_options(options)}"
        )
    elif task_type == TASK_CONTEXT_CLOZE:
        context = str(payload.get("context") or "").strip()
        user = (
            "Task: Context Cloze\n\n"
            "빈칸에 가장 알맞은 단어를 고르시오.\n\n"
            f"{context}\n\n"
            f"{_format_options(options)}"
        )
    elif task_type == TASK_SENSE_DISAMBIGUATION:
        context = str(payload.get("context") or "").strip()
        user = (
            "Task: Sense Disambiguation\n\n"
            "아래 문장에서 밑줄 친 단어의 의미는?\n\n"
            f"{context}\n\n"
            f"{_format_options(options)}"
        )
    else:
        raise ValueError(f"unsupported task type: {task_type}")

    assistant = (
        _json_answer_text(answer_idx, rationale)
        if response_format == "json_answer"
        else _answer_text(options, answer_idx, rationale)
    )
    return make_sft_record(SYSTEM_PROMPT, user, assistant)
