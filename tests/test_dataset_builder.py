import json
from pathlib import Path

from term_ai.augmentation.dataset_builder import build_raw_mcq_metadata_from_anchors, build_raw_sft_from_anchors
from term_ai.contracts import RAW_GT_STATUS, TASK_RAW_MEANING_SELECTION, validate_sft_record


def test_build_raw_sft_from_anchors(tmp_path: Path):
    anchors = tmp_path / "anchors.jsonl"
    rows = [
        {"anchor_id": "a1", "word_id": "w1", "word": "contract", "pos": "명사", "meaning": "계약", "split": "train"},
        {"anchor_id": "a2", "word_id": "w2", "word": "invoice", "pos": "명사", "meaning": "청구서", "split": "train"},
        {"anchor_id": "a3", "word_id": "w3", "word": "audit", "pos": "명사", "meaning": "감사", "split": "train"},
        {"anchor_id": "a4", "word_id": "w4", "word": "budget", "pos": "명사", "meaning": "예산", "split": "train"},
    ]
    anchors.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    output = tmp_path / "sft"
    counts = build_raw_sft_from_anchors(anchors, output)
    assert counts["train"] == 4
    first = json.loads((output / "raw_train_sft_v1.jsonl").read_text(encoding="utf-8").splitlines()[0])
    validate_sft_record(first)


def test_build_raw_mcq_metadata_marks_raw_gt_and_source_task(tmp_path: Path):
    anchors = tmp_path / "anchors.jsonl"
    rows = [
        {"anchor_id": "a1", "word_id": "w1", "word": "contract", "pos": "명사", "meaning": "계약", "split": "test"},
        {"anchor_id": "a2", "word_id": "w2", "word": "invoice", "pos": "명사", "meaning": "청구서", "split": "test"},
        {"anchor_id": "a3", "word_id": "w3", "word": "audit", "pos": "명사", "meaning": "감사", "split": "test"},
        {"anchor_id": "a4", "word_id": "w4", "word": "budget", "pos": "명사", "meaning": "예산", "split": "test"},
    ]
    anchors.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    output = tmp_path / "metadata"
    counts = build_raw_mcq_metadata_from_anchors(anchors, output)
    assert counts["test"] == 4
    first = json.loads((output / "raw_test_mcq_v1.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["status"] == RAW_GT_STATUS
    assert first["source"] == "raw_gt"
    assert first["dataset_view"] == "test_raw"
    assert first["payload"]["source_task_type"] == TASK_RAW_MEANING_SELECTION
