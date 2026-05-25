from term_ai.augmentation.filters import AutoFilter
from term_ai.contracts import TASK_CONTEXT_CLOZE, TASK_SYNONYM


def test_context_cloze_rejects_answer_leakage():
    payload = {
        "task_type": TASK_CONTEXT_CLOZE,
        "word": "outstanding",
        "meaning_ko": "미결제",
        "context": "The finance team reported outstanding payments before the audit.",
        "options": ["outstanding", "additional", "optional", "preliminary"],
        "answer_idx": 0,
        "rationale": "정답 단어가 문맥상 맞습니다.",
        "teacher_scores": [0.9, 0.05, 0.03, 0.02],
    }
    result = AutoFilter().validate_payload(payload)
    assert result.status == "rejected"
    assert "answer_leakage_in_context" in result.errors


def test_context_cloze_accepts_basic_candidate():
    payload = {
        "task_type": TASK_CONTEXT_CLOZE,
        "word": "outstanding",
        "meaning_ko": "미결제",
        "context": "The finance team reported three ___ payments before the annual audit review.",
        "options": ["outstanding", "additional", "optional", "preliminary"],
        "answer_idx": 0,
        "rationale": "감사 전에 해결해야 할 payments에는 outstanding이 가장 자연스럽습니다.",
        "teacher_scores": [0.9, 0.05, 0.03, 0.02],
    }
    result = AutoFilter().validate_payload(payload)
    assert result.status == "aug_auto_pass"


def test_synonym_rejects_target_word_and_blank_context():
    payload = {
        "task_type": TASK_SYNONYM,
        "word": "executive",
        "meaning_ko": "임원",
        "context": "The company appointed a new ___ to oversee sales.",
        "options": ["executive", "manager", "director", "leader"],
        "answer_idx": 0,
        "rationale": "target word 자체를 정답으로 쓰면 유의어 문항이 아닙니다.",
        "teacher_scores": [0.9, 0.04, 0.04, 0.02],
    }
    result = AutoFilter().validate_payload(payload)
    assert result.status == "rejected"
    assert "target_word_in_options" in result.errors
    assert "blank_context_for_non_cloze" in result.errors
