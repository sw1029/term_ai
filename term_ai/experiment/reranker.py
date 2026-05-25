from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from term_ai.contracts import RAW_GT_STATUS, answer_label, write_jsonl
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import MCQItem, load_mcq_items, prediction_row
from term_ai.experiment.ops import memory_snapshot, timed
from term_ai.experiment.test_lock import enforce_final_test_once


def _reranker_training_examples(items: list[MCQItem]) -> list[Any]:
    from sentence_transformers import InputExample

    examples: list[Any] = []
    for item in items:
        query = item.query_text()
        for idx, option in enumerate(item.options):
            examples.append(InputExample(texts=[query, option], label=float(idx == item.answer_idx)))
    return examples


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
    final_test_once: bool = True,
    test_lock_dir: str | Path | None = None,
) -> dict[str, Any]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Install baseline dependencies first: pip install -e .[baseline]") from exc

    all_items = load_mcq_items(metadata_path, min_status=min_status)
    items = [item for item in all_items if item.split == eval_split]
    if not items:
        raise ValueError(f"no reranker eval items: split={eval_split}, min_status={min_status}")
    model = CrossEncoder(model_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enforce_final_test_once(output, "B3", eval_split, enabled=final_test_once, lock_dir=test_lock_dir)
    if fine_tune:
        try:
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise RuntimeError("Install train dependencies first: pip install -e .[train]") from exc
        train_items = [item for item in all_items if item.split == train_split]
        if not train_items:
            raise ValueError("no train items available for reranker fine-tuning")
        train_loader = DataLoader(_reranker_training_examples(train_items), shuffle=True, batch_size=batch_size)
        model.fit(train_dataloader=train_loader, epochs=epochs, output_path=str(output / "reranker_finetuned"))
        model = CrossEncoder(str(output / "reranker_finetuned"))
    predictions = []
    for item in items:
        pairs = [(item.query_text(), option) for option in item.options]
        with timed() as state:
            scores = np.asarray(model.predict(pairs))
        idx = int(np.argmax(scores))
        predictions.append(
            prediction_row(
                item,
                answer_label(idx),
                float(scores[idx]),
                latency_ms=state["latency_ms"],
                extra={"reranker_score": float(scores[idx]), **memory_snapshot()},
            )
        )

    metrics = summarize_predictions(predictions)
    metrics["fine_tuned"] = fine_tune
    write_jsonl(output / "prediction_log.jsonl", predictions)
    (output / "metric_log.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
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
    parser.add_argument("--test-lock-dir")
    parser.add_argument("--allow-repeat-test", action="store_true")
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
        final_test_once=not args.allow_repeat_test,
        test_lock_dir=args.test_lock_dir,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
