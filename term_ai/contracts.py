from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are a TOEIC business vocabulary expert. "
    "Answer with the letter and a brief reason in Korean."
)

TASK_SYNONYM = "Synonym Selection"
TASK_ANTONYM = "Antonym Selection"
TASK_CONTEXT_CLOZE = "Context Cloze"
TASK_SENSE_DISAMBIGUATION = "Sense Disambiguation"
TASK_RAW_MEANING_SELECTION = "Raw Meaning Selection"

TASK_TYPES = {
    TASK_SYNONYM,
    TASK_ANTONYM,
    TASK_CONTEXT_CLOZE,
    TASK_SENSE_DISAMBIGUATION,
}

TASK_RATIOS = {
    TASK_SYNONYM: 0.40,
    TASK_ANTONYM: 0.20,
    TASK_CONTEXT_CLOZE: 0.25,
    TASK_SENSE_DISAMBIGUATION: 0.15,
}

SPLIT_RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}

RAW_GT_STATUS = "raw_gt"
APPROVED_AUG_STATUS = "aug_human_pass"

STATUS_ORDER = [
    "aug_candidate",
    "aug_auto_pass",
    "aug_judge_pass",
    APPROVED_AUG_STATUS,
]
VALID_STATUSES = set(STATUS_ORDER + [RAW_GT_STATUS, "rejected"])


class ContractError(ValueError):
    """Raised when a record violates a documented interface contract."""


def _stable_part(part: object) -> str:
    if isinstance(part, (dict, list)):
        return json.dumps(part, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(part)


def stable_id(*parts: object, length: int = 16) -> str:
    payload = "\x1f".join(_stable_part(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_key(value: str) -> str:
    return normalize_space(value).casefold()


def normalize_openai_model_id(model: str) -> str:
    """Normalize the project document spelling to the API model id.

    The planning docs use the human-readable spelling "gpt 5.4 mini"; the
    OpenAI API expects the dash-delimited model id.
    """

    normalized = normalize_key(model).replace("_", " ")
    if normalized == "gpt 5.4 mini":
        return "gpt-5.4-mini"
    return model.strip()


@dataclass(frozen=True)
class RawAnchor:
    anchor_id: str
    word_id: str
    word: str
    pos: str
    meaning: str
    source_line: int
    definition_index: int
    duplicate_of: str | None = None
    split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidatePayload:
    task_type: str
    word: str
    options: list[str]
    answer_idx: int
    context: str | None = None
    meaning_ko: str | None = None
    rationale: str | None = None
    teacher_scores: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AugmentationMetadata:
    item_id: str
    anchor_id: str
    word_id: str
    split: str
    status: str
    prompt_version: str
    generator_model: str
    payload: dict[str, Any]
    teacher_rationale: str | None = None
    teacher_scores: list[float] | None = None
    validation_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AutoFilterResult:
    item_id: str
    status: str
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JudgeValidation:
    item_id: str
    semantic_correctness: int
    distractor_validity: int
    context_naturalness: int
    leakage_check: str
    final_decision: str
    judge_model: str | None = None
    notes: str | None = None

    def accepted(self) -> bool:
        return (
            self.semantic_correctness == 2
            and self.distractor_validity >= 1
            and self.context_naturalness >= 1
            and self.leakage_check == "pass"
            and self.final_decision == "accept"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def answer_label(answer_idx: int) -> str:
    if answer_idx < 0 or answer_idx > 3:
        raise ContractError(f"answer_idx out of range: {answer_idx}")
    return chr(ord("A") + answer_idx)


def ensure_task_type(task_type: str) -> str:
    if task_type not in TASK_TYPES:
        raise ContractError(f"unsupported task_type: {task_type!r}")
    return task_type


def status_reaches(status: str, minimum: str) -> bool:
    if minimum == "any":
        return status != "rejected"
    if status == "rejected":
        return False
    if status not in VALID_STATUSES:
        raise ContractError(f"unknown status: {status!r}")
    if minimum == RAW_GT_STATUS:
        return status == RAW_GT_STATUS
    if status == RAW_GT_STATUS:
        return False
    if minimum not in STATUS_ORDER:
        raise ContractError(f"unknown minimum status: {minimum!r}")
    return STATUS_ORDER.index(status) >= STATUS_ORDER.index(minimum)


def make_sft_record(system: str, user: str, assistant: str) -> dict[str, Any]:
    record = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }
    validate_sft_record(record)
    return record


def validate_sft_record(record: dict[str, Any]) -> None:
    if set(record.keys()) != {"messages"}:
        extra = sorted(set(record.keys()) - {"messages"})
        missing = sorted({"messages"} - set(record.keys()))
        raise ContractError(f"SFT record must contain only messages; extra={extra}, missing={missing}")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ContractError("messages must be a 3-item list: system, user, assistant")

    expected_roles = ["system", "user", "assistant"]
    for idx, (message, role) in enumerate(zip(messages, expected_roles)):
        if not isinstance(message, dict):
            raise ContractError(f"message {idx} must be an object")
        if set(message.keys()) != {"role", "content"}:
            raise ContractError(f"message {idx} must contain only role and content")
        if message["role"] != role:
            raise ContractError(f"message {idx} role must be {role!r}")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ContractError(f"message {idx} content must be a non-empty string")


def iter_jsonl(path: str | Any) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSONL at line {line_no}: {exc}") from exc


def dumps_jsonl(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_jsonl(path: str | Any, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(dumps_jsonl(record))
            count += 1
    return count
