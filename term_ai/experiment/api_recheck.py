from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from term_ai.augmentation.teacher import OpenAITeacherClient
from term_ai.contracts import write_jsonl
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import load_mcq_items, parse_answer_letter, prediction_row
from term_ai.experiment.ops import timed


def run_api_recheck(
    metadata_path: str | Path,
    output_dir: str | Path,
    model: str = "gpt-5.4-mini",
    env_path: str | Path = ".env",
    eval_split: str = "dev",
    min_status: str = "aug_auto_pass",
    requests_per_second: float = 1.0,
    limit: int | None = None,
    cost_per_1000_questions: float = 0.0,
) -> dict[str, Any]:
    if requests_per_second <= 0:
        raise ValueError("requests_per_second must be positive")
    items = [item for item in load_mcq_items(metadata_path, min_status=min_status) if item.split == eval_split]
    if limit is not None:
        items = items[:limit]
    client = OpenAITeacherClient(model=model, env_path=str(env_path))
    min_interval = 1.0 / requests_per_second
    last_request_at: float | None = None
    predictions: list[dict[str, Any]] = []

    for item in items:
        if last_request_at is not None:
            elapsed = time.monotonic() - last_request_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        with timed() as state:
            last_request_at = time.monotonic()
            result = client.generate_json(item.prompt())
        answer, confidence = parse_answer_letter(json.dumps(result, ensure_ascii=False))
        predictions.append(
            prediction_row(
                item,
                answer or "PARSE_ERROR",
                confidence if confidence is not None else 0.0,
                latency_ms=state["latency_ms"],
                extra={"parse_error": answer is None, "raw_response": result},
            )
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = summarize_predictions(predictions)
    metrics["cost_per_1000_questions"] = cost_per_1000_questions
    write_jsonl(output / "prediction_log.jsonl", predictions)
    (output / "metric_log.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B4 API recheck evaluation.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--min-status", default="aug_auto_pass")
    parser.add_argument("--requests-per-second", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cost-per-1000-questions", type=float, default=0.0)
    args = parser.parse_args()
    metrics = run_api_recheck(
        args.metadata,
        args.output_dir,
        model=args.model,
        env_path=args.env,
        eval_split=args.eval_split,
        min_status=args.min_status,
        requests_per_second=args.requests_per_second,
        limit=args.limit,
        cost_per_1000_questions=args.cost_per_1000_questions,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
