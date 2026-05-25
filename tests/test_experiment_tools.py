import json
from pathlib import Path

import pytest

from term_ai.experiment.hybrid import run_hybrid_policy, tune_hybrid_policy
from term_ai.experiment.kd_sweep import KDAblationSweepConfig, run_kd_ablation_sweep
from term_ai.experiment.lora_kd import metadata_to_kd_rows
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import parse_answer_letter
from term_ai.experiment.prompt_variation_sweep import PromptVariationSweepConfig, run_prompt_variation_sweep
from term_ai.experiment.quantization import validate_g3_adapter_checkpoint
from term_ai.experiment.reporting import write_final_report_inputs
from term_ai.experiment.test_lock import enforce_final_test_once
from term_ai.experiment.workflow import _augmentation_split_totals, _default_phase_jobs


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


def test_hybrid_policy_uses_cross_encoder_for_middle_confidence(tmp_path: Path):
    primary = tmp_path / "primary.jsonl"
    cross = tmp_path / "cross.jsonl"
    fallback = tmp_path / "fallback.jsonl"
    primary.write_text(
        json.dumps({"item_id": "i1", "label": "A", "prediction": "B", "confidence": 0.55}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    cross.write_text(
        json.dumps({"item_id": "i1", "label": "A", "prediction": "A", "confidence": 0.7}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fallback.write_text(
        json.dumps({"item_id": "i1", "label": "A", "prediction": "B", "confidence": 0.8}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics = run_hybrid_policy(
        primary,
        fallback,
        tmp_path / "out_cross",
        cross_encoder_predictions=cross,
        low_confidence_threshold=0.4,
        high_confidence_threshold=0.7,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["cross_encoder_rate"] == 1.0


def test_final_test_lock_blocks_second_run(tmp_path: Path):
    output = tmp_path / "runs" / "B0"
    output.mkdir(parents=True)
    lock_dir = tmp_path / "locks"
    first = enforce_final_test_once(output, "B0", "test", lock_dir=lock_dir)
    assert first is not None and first.exists()
    with pytest.raises(RuntimeError):
        enforce_final_test_once(tmp_path / "another" / "B0", "B0", "test", lock_dir=lock_dir)
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
    rows = metadata_to_kd_rows(metadata, include_rationale=False, response_format="letter_reason")
    assert rows[0]["teacher_scores"] == [0.7, 0.1, 0.1, 0.1]
    assert "정답은 A입니다." in rows[0]["messages"][2]["content"]


def test_lora_kd_view_requires_teacher_scores_by_default(tmp_path: Path):
    metadata = tmp_path / "metadata_missing_scores.jsonl"
    row = {
        "item_id": "i1",
        "status": "aug_judge_pass",
        "split": "train",
        "payload": {
            "task_type": "Context Cloze",
            "word": "outstanding",
            "context": "The team reported ___ invoices before the quarterly audit review meeting ended.",
            "options": ["outstanding", "optional", "new", "late"],
            "answer_idx": 0,
            "rationale": "문맥상 outstanding이 맞습니다.",
        },
    }
    metadata.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="teacher_scores"):
        metadata_to_kd_rows(metadata, min_status="aug_judge_pass")


def test_lora_kd_view_can_emit_json_distribution(tmp_path: Path):
    metadata = tmp_path / "metadata_json.jsonl"
    row = {
        "item_id": "i1",
        "status": "aug_judge_pass",
        "split": "train",
        "payload": {
            "task_type": "Context Cloze",
            "word": "outstanding",
            "context": "The team reported ___ invoices before the quarterly audit review meeting ended.",
            "options": ["outstanding", "optional", "new", "late"],
            "answer_idx": 0,
            "rationale": "outstanding matches the invoice context.",
            "teacher_scores": [0.7, 0.1, 0.1, 0.1],
        },
    }
    metadata.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = metadata_to_kd_rows(metadata, min_status="aug_judge_pass")
    assistant = json.loads(rows[0]["messages"][2]["content"])
    assert assistant["answer"] == "A"
    assert assistant["distribution"]["A"] == 0.7


def test_hybrid_policy_tuning_writes_selected_policy(tmp_path: Path):
    primary = tmp_path / "primary.jsonl"
    fallback = tmp_path / "fallback.jsonl"
    primary.write_text(
        "\n".join(
            [
                json.dumps({"item_id": "i1", "label": "A", "prediction": "B", "confidence": 0.2}),
                json.dumps({"item_id": "i2", "label": "B", "prediction": "B", "confidence": 0.9}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fallback.write_text(
        "\n".join(
            [
                json.dumps({"item_id": "i1", "label": "A", "prediction": "A", "confidence": 0.8}),
                json.dumps({"item_id": "i2", "label": "B", "prediction": "A", "confidence": 0.8}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = tune_hybrid_policy(primary, fallback, tmp_path / "hybrid", threshold_grid=[0.3, 0.7])
    assert metrics["accuracy"] == 1.0
    assert (tmp_path / "hybrid" / "hybrid_policy_tuning.json").exists()


def test_prompt_variation_sweep_writes_variant_matrix_without_training(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    record = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Task: Context Cloze"},
            {"role": "assistant", "content": "A) answer"},
        ]
    }
    train.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    dev.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = run_prompt_variation_sweep(
        PromptVariationSweepConfig(
            train_jsonl=str(train),
            dev_jsonl=str(dev),
            output_dir=str(tmp_path / "sweep"),
            variants=["default", "concise"],
        )
    )

    assert [row["status"] for row in manifest["runs"]] == ["planned", "planned"]
    assert (tmp_path / "sweep" / "prompt_variation_sweep.json").exists()


def test_kd_ablation_sweep_writes_hard_soft_rationale_matrix(tmp_path: Path):
    manifest = run_kd_ablation_sweep(
        KDAblationSweepConfig(
            model_name_or_path="local-model",
            metadata_jsonl="train.jsonl",
            dev_metadata_jsonl="dev.jsonl",
            output_dir=str(tmp_path / "kd"),
        )
    )

    ablations = {row["ablation"]: row for row in manifest["runs"]}
    assert set(ablations) == {
        "hard_only_with_rationale",
        "soft_kd_with_rationale",
        "soft_kd_no_rationale",
        "classification_head_kd",
    }
    assert ablations["hard_only_with_rationale"]["hard_label_only"] is True
    assert ablations["soft_kd_no_rationale"]["include_rationale"] is False
    assert ablations["classification_head_kd"]["response_format"] == "option_classification_logits"


def test_g4_adapter_validation_requires_g3_manifest(tmp_path: Path):
    adapter = tmp_path / "final_adapter"
    adapter.mkdir()
    with pytest.raises(ValueError, match="G3 KD manifest"):
        validate_g3_adapter_checkpoint(adapter)

    manifest = {
        "experiment_family": "G3",
        "checkpoint_type": "lora_sft_kd",
        "final_adapter": str(adapter),
    }
    (tmp_path / "g3_checkpoint_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_g3_adapter_checkpoint(adapter)
    assert result["verified"] is True
    assert result["manifest_type"] == "g3_checkpoint_manifest"


def test_ops_summary_includes_batch_cold_start_and_local_cost():
    metrics = summarize_predictions(
        [
            {
                "label": "A",
                "prediction": "A",
                "confidence": 0.9,
                "latency_ms": 100.0,
                "tokens_per_sec": 20.0,
                "ram_mb": 512.0,
                "batch_size": 1,
                "cold_start_ms": 250.0,
                "local_cost_per_hour_usd": 1.8,
            }
        ]
    )

    assert metrics["batch_size_1_latency_p95"] == 100.0
    assert metrics["cold_start_ms"] == 250.0
    assert metrics["cost_per_1000_questions"] > 0


def test_final_report_collects_explanation_judge_summary(tmp_path: Path):
    runs = tmp_path / "runs"
    summary_dir = runs / "G0"
    summary_dir.mkdir(parents=True)
    (summary_dir / "G0_explanation_judgment_summary.json").write_text(
        json.dumps(
            {
                "n": 2,
                "semantic_correctness_avg": 1.5,
                "reasoning_faithfulness_avg": 1.0,
                "hallucination_fail_rate": 0.5,
                "final_score_avg": 1.25,
            }
        ),
        encoding="utf-8",
    )

    outputs = write_final_report_inputs(runs, tmp_path / "reports")
    report = Path(outputs["final_report"]).read_text(encoding="utf-8")
    assert "Explanation Judge" in report
    assert "hallucination_fail_rate=0.5000" in report


def test_master_workflow_default_jobs_cover_g0_g4_and_h1():
    jobs = _default_phase_jobs(
        {
            "runs_dir": "runs",
            "auto_phase_jobs": {
                "enabled": True,
                "output_dir": "runs/master_matrix",
                "eval_split": "dev",
                "model_ids": {"gemma": "google/gemma-2-2b-it", "qwen": "Qwen/Qwen2.5-3B-Instruct"},
            },
        }
    )
    names = {job["name"] for job in jobs}
    assert {"G0-Gemma", "G0-Qwen", "G4", "H1", "prompt-template-variation", "G3-KD-ablation"} <= names
    assert any("execution.adapter_path=" in part for job in jobs for part in job["command"])


def test_augmentation_split_totals_enable_dev_test_generation():
    totals = _augmentation_split_totals({"split_totals": {"train": 4, "dev": 2, "test": 2}}, total=0)
    assert totals == {"train": 4, "dev": 2, "test": 2}
