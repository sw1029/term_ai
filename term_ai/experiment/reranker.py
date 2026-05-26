from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from term_ai.contracts import RAW_GT_STATUS, answer_label
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import MCQItem, load_mcq_items, prediction_row
from term_ai.experiment.ops import memory_snapshot, timed
from term_ai.experiment.progress import InterruptGuard, ProgressLogger, backup_artifact
from term_ai.experiment.test_lock import enforce_final_test_once


def _reranker_training_examples(items: list[MCQItem]) -> list[Any]:
    from sentence_transformers import InputExample

    examples: list[Any] = []
    for item in items:
        query = item.query_text()
        for idx, option in enumerate(item.options):
            examples.append(InputExample(texts=[query, option], label=float(idx == item.answer_idx)))
    return examples


def _normalize_scores(scores: np.ndarray, method: str) -> np.ndarray:
    if method == "raw":
        return scores.astype(float)
    if method == "sigmoid":
        return 1.0 / (1.0 + np.exp(-scores.astype(float)))
    if method == "minmax":
        minimum = float(scores.min())
        maximum = float(scores.max())
        if maximum == minimum:
            return np.ones_like(scores, dtype=float)
        return (scores.astype(float) - minimum) / (maximum - minimum)
    raise ValueError("score_normalization must be raw, sigmoid, or minmax")


def _score_items(model: Any, items: list[MCQItem], score_normalization: str) -> list[tuple[MCQItem, np.ndarray, np.ndarray]]:
    scored: list[tuple[MCQItem, np.ndarray, np.ndarray]] = []
    for item in items:
        pairs = [(item.query_text(), option) for option in item.options]
        raw_scores = np.asarray(model.predict(pairs), dtype=float)
        normalized_scores = _normalize_scores(raw_scores, score_normalization)
        scored.append((item, raw_scores, normalized_scores))
    return scored


def tune_reranker_threshold(
    model: Any,
    items: list[MCQItem],
    score_normalization: str = "sigmoid",
) -> dict[str, Any]:
    if not items:
        raise ValueError("no dev items available for reranker threshold tuning")
    scored = _score_items(model, items, score_normalization)
    candidates = sorted({float(scores.max()) for _, _, scores in scored} | {0.0, 1.0})
    best = {"threshold": candidates[0], "accuracy": -1.0, "coverage": 0.0}
    for threshold in candidates:
        correct = 0
        covered = 0
        for item, _, scores in scored:
            idx = int(np.argmax(scores))
            if float(scores[idx]) >= threshold:
                covered += 1
                correct += int(answer_label(idx) == item.label)
        accuracy = correct / len(scored)
        coverage = covered / len(scored)
        if (accuracy, coverage, -threshold) > (best["accuracy"], best["coverage"], -best["threshold"]):
            best = {"threshold": threshold, "accuracy": accuracy, "coverage": coverage}
    best["score_normalization"] = score_normalization
    return best


def run_reranker(
    metadata_path: str | Path,
    output_dir: str | Path,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    eval_split: str = "dev",
    min_status: str = RAW_GT_STATUS,
    train_split: str = "train",
    fine_tune: bool = False,
    epochs: int = 1,
    batch_size: int = 8,
    score_normalization: str = "sigmoid",
    threshold: float | None = None,
    threshold_split: str = "dev",
    final_test_once: bool = True,
    test_lock_dir: str | Path | None = None,
    resume: bool = True,
    progress_interval_items: int = 1,
    save_steps: int | None = None,
    save_total_limit: int = 3,
    backup_weights: bool = True,
    backup_checkpoints: bool = True,
) -> dict[str, Any]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Install baseline dependencies first: pip install -e .[baseline]") from exc

    all_items = load_mcq_items(metadata_path, min_status=min_status)
    items = [item for item in all_items if item.split == eval_split]
    if not items:
        raise ValueError(f"no reranker eval items: split={eval_split}, min_status={min_status}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enforce_final_test_once(output, "B3", eval_split, enabled=final_test_once, lock_dir=test_lock_dir)
    progress = ProgressLogger(
        output,
        resume=resume,
        progress_interval_items=progress_interval_items,
        stage="B3:reranker",
        total_count=len(items),
    )
    completed_metrics = progress.completed_metrics_if_available(item.item_id for item in items)
    if completed_metrics is not None:
        return completed_metrics
    model = CrossEncoder(model_name)
    final_model_path = output / "reranker_finetuned"
    with InterruptGuard(progress, stage="B3:reranker-training"):
        if fine_tune:
            if resume and final_model_path.exists():
                model = CrossEncoder(str(final_model_path))
            else:
                try:
                    from torch.utils.data import DataLoader
                except ImportError as exc:
                    raise RuntimeError("Install train dependencies first: pip install -e .[train]") from exc
                train_items = [item for item in all_items if item.split == train_split]
                if not train_items:
                    raise ValueError("no train items available for reranker fine-tuning")
                train_loader = DataLoader(_reranker_training_examples(train_items), shuffle=True, batch_size=batch_size)
                fit_kwargs: dict[str, Any] = {
                    "train_dataloader": train_loader,
                    "epochs": epochs,
                    "output_path": str(final_model_path),
                }
                fit_params = inspect.signature(model.fit).parameters
                checkpoint_dir: Path | None = None
                if backup_checkpoints and save_steps is not None and "checkpoint_path" in fit_params:
                    checkpoint_dir = output / "checkpoints" / "reranker"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    fit_kwargs["checkpoint_path"] = str(checkpoint_dir)
                    if "checkpoint_save_steps" in fit_params:
                        fit_kwargs["checkpoint_save_steps"] = int(save_steps)
                    if "checkpoint_save_total_limit" in fit_params:
                        fit_kwargs["checkpoint_save_total_limit"] = int(save_total_limit)
                model.fit(**fit_kwargs)
                if backup_checkpoints and checkpoint_dir is not None and checkpoint_dir.exists():
                    backup_artifact(checkpoint_dir, output, name="reranker_checkpoints")
                if backup_weights:
                    backup_artifact(final_model_path, output, name="reranker_finetuned")
                model = CrossEncoder(str(final_model_path))
    tuning: dict[str, Any] | None = None
    if threshold is None:
        threshold_items = [item for item in all_items if item.split == threshold_split]
        tuning = tune_reranker_threshold(model, threshold_items, score_normalization=score_normalization)
        threshold = float(tuning["threshold"])
    with InterruptGuard(progress, stage="B3:reranker-eval"):
        for item in items:
            if progress.has_prediction(item.item_id):
                continue
            pairs = [(item.query_text(), option) for option in item.options]
            with timed() as state:
                raw_scores = np.asarray(model.predict(pairs), dtype=float)
                scores = _normalize_scores(raw_scores, score_normalization)
            idx = int(np.argmax(scores))
            confidence = float(scores[idx])
            prediction = answer_label(idx) if threshold is None or confidence >= threshold else "ABSTAIN"
            progress.append_prediction(
                prediction_row(
                    item,
                    prediction,
                    confidence,
                    latency_ms=state["latency_ms"],
                    extra={
                        "reranker_score": float(raw_scores[idx]),
                        "reranker_normalized_score": confidence,
                        "score_normalization": score_normalization,
                        "threshold": threshold,
                        "abstained": prediction == "ABSTAIN",
                        **memory_snapshot(),
                    },
                )
            )

    predictions = progress.predictions_for_items(item.item_id for item in items)
    metrics = summarize_predictions(predictions)
    metrics["fine_tuned"] = fine_tune
    metrics["score_normalization"] = score_normalization
    metrics["threshold"] = threshold
    if tuning is not None:
        metrics["threshold_tuning"] = tuning
        (output / "reranker_threshold.json").write_text(json.dumps(tuning, ensure_ascii=False, indent=2), encoding="utf-8")
    progress.finalize_predictions(
        metrics,
        predictions,
        final_artifact=final_model_path if fine_tune and final_model_path.exists() else None,
        details={"model_name": model_name, "fine_tuned": fine_tune},
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B3 cross-encoder/reranker evaluation.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--min-status", default=RAW_GT_STATUS)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-normalization", choices=["raw", "sigmoid", "minmax"], default="sigmoid")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-split", default="dev")
    parser.add_argument("--test-lock-dir")
    parser.add_argument("--allow-repeat-test", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress-interval-items", type=int, default=1)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--no-weight-backup", action="store_true")
    parser.add_argument("--no-checkpoint-backup", action="store_true")
    args = parser.parse_args()
    metrics = run_reranker(
        args.metadata,
        args.output_dir,
        model_name=args.model_name,
        eval_split=args.eval_split,
        min_status=args.min_status,
        train_split=args.train_split,
        fine_tune=args.fine_tune,
        epochs=args.epochs,
        batch_size=args.batch_size,
        score_normalization=args.score_normalization,
        threshold=args.threshold,
        threshold_split=args.threshold_split,
        final_test_once=not args.allow_repeat_test,
        test_lock_dir=args.test_lock_dir,
        resume=not args.no_resume,
        progress_interval_items=args.progress_interval_items,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        backup_weights=not args.no_weight_backup,
        backup_checkpoints=not args.no_checkpoint_backup,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
