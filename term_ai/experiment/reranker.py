from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from term_ai.contracts import answer_label, write_jsonl
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import load_mcq_items, prediction_row
from term_ai.experiment.ops import timed


def run_reranker(
    metadata_path: str | Path,
    output_dir: str | Path,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    eval_split: str = "dev",
    min_status: str = "aug_auto_pass",
) -> dict[str, Any]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Install baseline dependencies first: pip install -e .[baseline]") from exc

    items = [item for item in load_mcq_items(metadata_path, min_status=min_status) if item.split == eval_split]
    model = CrossEncoder(model_name)
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
                extra={"reranker_score": float(scores[idx])},
            )
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = summarize_predictions(predictions)
    write_jsonl(output / "prediction_log.jsonl", predictions)
    (output / "metric_log.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B3 cross-encoder/reranker evaluation.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--min-status", default="aug_auto_pass")
    args = parser.parse_args()
    metrics = run_reranker(
        args.metadata,
        args.output_dir,
        model_name=args.model_name,
        eval_split=args.eval_split,
        min_status=args.min_status,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
