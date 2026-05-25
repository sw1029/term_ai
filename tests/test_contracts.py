import pytest

from term_ai.contracts import (
    APPROVED_AUG_STATUS,
    ContractError,
    RAW_GT_STATUS,
    SYSTEM_PROMPT,
    make_sft_record,
    normalize_openai_model_id,
    status_reaches,
    validate_sft_record,
)
from term_ai.augmentation.sft_builder import candidate_payload_to_sft_record


def test_sft_record_allows_only_messages():
    record = make_sft_record(SYSTEM_PROMPT, "Task: Context Cloze\n\nA) x", "A) x\n\nreason")
    validate_sft_record(record)


def test_sft_record_rejects_extra_fields():
    record = make_sft_record(SYSTEM_PROMPT, "user", "assistant")
    record["metadata"] = {"split": "train"}
    with pytest.raises(ContractError):
        validate_sft_record(record)


def test_candidate_sft_conversion_keeps_metadata_out():
    record = candidate_payload_to_sft_record(
        {
            "task_type": "Context Cloze",
            "word": "outstanding",
            "meaning_ko": "미결제",
            "context": "The finance team reported three ___ payments before the annual audit review.",
            "options": ["outstanding", "additional", "optional", "preliminary"],
            "answer_idx": 0,
            "rationale": "감사 전에 해결해야 할 payments에는 outstanding이 가장 자연스럽습니다.",
            "teacher_scores": [0.9, 0.05, 0.03, 0.02],
        }
    )
    assert set(record.keys()) == {"messages"}
    validate_sft_record(record)


def test_raw_gt_status_is_not_approved_augmentation():
    assert status_reaches(RAW_GT_STATUS, RAW_GT_STATUS)
    assert not status_reaches(RAW_GT_STATUS, APPROVED_AUG_STATUS)


def test_openai_model_id_normalization_keeps_doc_alias_usable():
    assert normalize_openai_model_id("gpt 5.4 mini") == "gpt-5.4-mini"
