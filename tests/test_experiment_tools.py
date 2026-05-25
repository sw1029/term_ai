import json
from pathlib import Path

from term_ai.experiment.hybrid import run_hybrid_policy
from term_ai.experiment.mcq import parse_answer_letter


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
