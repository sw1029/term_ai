import json
from pathlib import Path

import pytest

from term_ai.experiment.hybrid import run_hybrid_policy
from term_ai.experiment.lora_kd import metadata_to_kd_rows
from term_ai.experiment.mcq import parse_answer_letter
from term_ai.experiment.test_lock import enforce_final_test_once


def test_parse_answer_letter_from_json_and_text():
    assert parse_answer_letter('{"answer": "B", "confidence": 0.7}') == ("B", 0.7)
    assert parse_answer_letter("정답은 C입니다.")[0] == "C"


def test_hybrid_policy_uses_fallback_when_confidence_low(tmp_path: Path):
    primary = tmp_path / "primary.jsonl"
    fallback = tmp_path / "fallback.jsonl"
    primary.write_text(
        json.dumps(
            {"item_id": "i1", "label": "A", "prediction": "B", "confidence": 0.2, "task_type": "t"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    fallback.write_text(
        json.dumps(
            {"item_id": "i1", "label": "A", "prediction": "A", "confidence": 0.8, "task_type": "t"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = run_hybrid_policy(primary, fallback, tmp_path / "out", confidence_threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["fallback_rate"] == 1.0


def test_final_test_lock_blocks_second_run(tmp_path: Path):
    output = tmp_path / "runs" / "B0"
    output.mkdir(parents=True)
    first = enforce_final_test_once(output, "B0", "test")
    assert first is not None and first.exists()
    with pytest.raises(RuntimeError):
        enforce_final_test_once(output, "B0", "test")
    assert enforce_final_test_once(output, "B0", "dev") is None


def test_lora_kd_view_uses_teacher_scores_and_can_drop_rationale(tmp_path: Path):
    metadata = tmp_path / "metadata.jsonl"
    row = {
        "item_id": "i1",
        "status": "aug_human_pass",
        "split": "train",
        "payload": {
            "task_type": "Context Cloze",
            "word": "outstanding",
            "context": "The team reported ___ invoices.",
            "options": ["outstanding", "optional", "new", "late"],
            "answer_idx": 0,
            "rationale": "문맥상 outstanding이 맞습니다.",
            "teacher_scores": [0.7, 0.1, 0.1, 0.1],
        },
    }
    metadata.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = metadata_to_kd_rows(metadata, include_rationale=False)
    assert rows[0]["teacher_scores"] == [0.7, 0.1, 0.1, 0.1]
    assert "정답은 A입니다." in rows[0]["messages"][2]["content"]
