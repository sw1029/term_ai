import json
from pathlib import Path

import pytest

from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.progress import InterruptGuard, ProgressLogger, resolve_latest_checkpoint


def _prediction(item_id: str, prediction: str = "A") -> dict:
    return {
        "item_id": item_id,
        "label": "A",
        "prediction": prediction,
        "confidence": 0.9,
        "latency_ms": 1.0,
        "task_type": "unit",
    }


def test_progress_logger_resumes_partial_predictions(tmp_path: Path):
    first = ProgressLogger(tmp_path, resume=False, total_count=2)
    first.append_prediction(_prediction("i1"))

    resumed = ProgressLogger(tmp_path, resume=True, total_count=2)
    assert resumed.has_prediction("i1")
    resumed.append_prediction(_prediction("i1", prediction="B"))
    resumed.append_prediction(_prediction("i2"))

    rows = resumed.predictions_for_items(["i1", "i2"])
    assert [row["item_id"] for row in rows] == ["i1", "i2"]
    assert [row["prediction"] for row in rows] == ["A", "A"]


def test_progress_partial_and_final_metrics_match(tmp_path: Path):
    logger = ProgressLogger(tmp_path, resume=False, total_count=2)
    logger.append_prediction(_prediction("i1"))
    logger.append_prediction(_prediction("i2", prediction="B"))
    rows = logger.predictions_for_items(["i1", "i2"])
    expected = summarize_predictions(rows)
    logger.finalize_predictions(expected, rows)

    final_metrics = json.loads((tmp_path / "metric_log.json").read_text(encoding="utf-8"))
    partial_metrics = json.loads((tmp_path / "metric_log.partial.json").read_text(encoding="utf-8"))
    assert final_metrics["accuracy"] == expected["accuracy"]
    assert partial_metrics["accuracy"] == expected["accuracy"]
    assert partial_metrics["partial"] is False


def test_interrupt_guard_records_interrupted_state(tmp_path: Path):
    logger = ProgressLogger(tmp_path, resume=False, total_count=1)
    logger.append_prediction(_prediction("i1"))
    checkpoint = tmp_path / "checkpoints" / "checkpoint-2"

    with pytest.raises(KeyboardInterrupt):
        with InterruptGuard(logger, stage="unit", checkpoint_callback=lambda: checkpoint):
            raise KeyboardInterrupt("ctrl-c")

    state = json.loads((tmp_path / "resume_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"
    assert state["completed_count"] == 1
    assert state["latest_checkpoint"] == str(checkpoint)
    assert "ctrl-c" in state["reason"]


def test_resolve_latest_checkpoint_uses_largest_step(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "checkpoint-1").mkdir(parents=True)
    (checkpoints / "checkpoint-12").mkdir()
    (checkpoints / "checkpoint-3").mkdir()

    assert resolve_latest_checkpoint(tmp_path) == checkpoints / "checkpoint-12"
