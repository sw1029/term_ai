import json
from pathlib import Path

import pytest

from term_ai.augmentation.dataset_builder import (
    build_approved_sft_by_split,
    build_kd_metadata_view,
    build_raw_aug_sft_by_split,
    build_raw_mcq_metadata_from_anchors,
    build_raw_sft_from_anchors,
    build_validated_aug_sft_by_split,
)
from term_ai.augmentation.orchestrator import generate_split_batch
from term_ai.augmentation.prompt_variation import write_sft_prompt_variants
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


def test_judge_validated_sft_is_not_named_human_approved(tmp_path: Path):
    metadata = tmp_path / "aug.jsonl"
    row = {
        "item_id": "i1",
        "status": "aug_judge_pass",
        "split": "train",
        "payload": {
            "task_type": "Context Cloze",
            "word": "renew",
            "context": "The procurement team decided to ___ the supplier contract after reviewing service quality.",
            "options": ["renew", "reject", "delay", "audit"],
            "answer_idx": 0,
            "rationale": "renew fits the contract extension context.",
            "teacher_scores": [0.8, 0.1, 0.05, 0.05],
        },
    }
    metadata.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = build_validated_aug_sft_by_split(metadata, tmp_path / "sft")
    assert counts["train"] == 1
    assert (tmp_path / "sft" / "judge_validated_train_sft_v1.jsonl").exists()
    assert not (tmp_path / "sft" / "approved_train_sft_v1.jsonl").exists()
    with pytest.raises(ValueError, match="human-approved"):
        build_approved_sft_by_split(metadata, tmp_path / "approved", min_status="aug_judge_pass")


def test_raw_aug_sft_merge_uses_explicit_judge_validated_prefix(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_record = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}
    for split in ("train", "dev", "test"):
        (raw_dir / f"raw_{split}_sft_v1.jsonl").write_text(
            json.dumps(raw_record, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    metadata = tmp_path / "aug.jsonl"
    row = {
        "item_id": "i1",
        "status": "aug_judge_pass",
        "split": "train",
        "payload": {
            "task_type": "Antonym Selection",
            "word": "expand",
            "meaning_ko": "grow",
            "options": ["shrink", "increase", "extend", "raise"],
            "answer_idx": 0,
            "rationale": "shrink is the opposite of expand.",
        },
    }
    metadata.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = build_raw_aug_sft_by_split(raw_dir, metadata, tmp_path / "merged")
    assert counts["train"] == 2
    assert counts["train_raw"] == 1
    assert counts["train_aug"] == 1
    assert (tmp_path / "merged" / "raw_judge_aug_train_sft_v1.jsonl").exists()


def test_kd_metadata_view_requires_explicit_raw_teacher_scores(tmp_path: Path):
    raw = tmp_path / "raw.jsonl"
    raw_row = {
        "item_id": "raw1",
        "anchor_id": "a1",
        "word_id": "w1",
        "status": "raw_gt",
        "split": "train",
        "source": "raw_gt",
        "payload": {"options": ["A", "B", "C", "D"], "answer_idx": 0},
    }
    raw.write_text(json.dumps(raw_row, ensure_ascii=False) + "\n", encoding="utf-8")
    generated = tmp_path / "generated.jsonl"
    gen_row = {
        "item_id": "gen1",
        "status": "aug_judge_pass",
        "split": "train",
        "payload": {
            "options": ["A", "B", "C", "D"],
            "answer_idx": 1,
            "teacher_scores": [0.1, 0.7, 0.1, 0.1],
        },
    }
    generated.write_text(json.dumps(gen_row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw metadata rows are missing teacher_scores"):
        build_kd_metadata_view(raw, generated, tmp_path / "kd.jsonl")

    scores = tmp_path / "scores.jsonl"
    scores.write_text(json.dumps({"item_id": "raw1", "teacher_scores": [0.7, 0.1, 0.1, 0.1]}) + "\n", encoding="utf-8")
    counts = build_kd_metadata_view(raw, generated, tmp_path / "kd.jsonl", raw_teacher_scores_path=scores)
    assert counts["raw_written"] == 1
    assert counts["generated_written"] == 1


def test_sft_prompt_variants_keep_messages_only_contract(tmp_path: Path):
    sft = tmp_path / "train.jsonl"
    record = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Task: Context Cloze"},
            {"role": "assistant", "content": "A) answer"},
        ]
    }
    sft.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = write_sft_prompt_variants(sft, tmp_path / "variants", variants=["default", "concise"])
    assert counts == {"default": 1, "concise": 1}
    concise = json.loads((tmp_path / "variants" / "train_concise.jsonl").read_text(encoding="utf-8").splitlines()[0])
    validate_sft_record(concise)
    assert "Choose the best option" in concise["messages"][1]["content"]


def test_generate_split_batch_uses_requested_dev_or_test_split(tmp_path: Path):
    anchors = tmp_path / "anchors.jsonl"
    rows = [
        {"anchor_id": "a1", "word_id": "w1", "word": "renew", "pos": "verb", "meaning": "extend", "split": "dev"},
        {"anchor_id": "a2", "word_id": "w2", "word": "audit", "pos": "noun", "meaning": "review", "split": "test"},
    ]
    anchors.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    class FakeTeacher:
        def generate_json(self, _prompt: str) -> dict:
            return {
                "task_type": "Context Cloze",
                "word": "renew",
                "context": "The team will ___ the contract.",
                "options": ["renew", "audit", "delay", "reject"],
                "answer_idx": 0,
                "rationale": "renew fits.",
                "teacher_scores": [0.8, 0.1, 0.05, 0.05],
            }

    manifest = generate_split_batch(
        anchors,
        tmp_path / "dev_aug.jsonl",
        total=1,
        split="dev",
        teacher_client=FakeTeacher(),
    )
    row = json.loads((tmp_path / "dev_aug.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert manifest["source_split"] == "dev"
    assert row["split"] == "dev"
