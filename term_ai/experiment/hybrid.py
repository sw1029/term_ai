from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from term_ai.contracts import write_jsonl
from term_ai.experiment.metrics import summarize_predictions


def _load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["item_id"])] = row
    return rows


def run_hybrid_policy(
    primary_predictions: str | Path,
    fallback_predictions: str | Path,
    output_dir: str | Path,
    confidence_threshold: float = 0.7,
    primary_cost_per_1000: float = 0.0,
    fallback_cost_per_1000: float = 0.0,
) -> dict[str, Any]:
    primary = _load_predictions(primary_predictions)
    fallback = _load_predictions(fallback_predictions)
    rows: list[dict[str, Any]] = []
    fallback_count = 0
    for item_id, row in primary.items():
        use_fallback = float(row.get("confidence", 0.0)) < confidence_threshold and item_id in fallback
        chosen = dict(fallback[item_id] if use_fallback else row)
        chosen["used_fallback"] = use_fallback
        chosen["primary_confidence"] = row.get("confidence")
        rows.append(chosen)
        fallback_count += int(use_fallback)

    metrics = summarize_predictions(rows)
    total = len(rows)
    fallback_rate = fallback_count / total if total else 0.0
    metrics["fallback_rate"] = fallback_rate
    metrics["cost_per_1000_questions"] = (
        primary_cost_per_1000 + fallback_rate * fallback_cost_per_1000
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "prediction_log.jsonl", rows)
    (output / "metric_log.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1 confidence based hybrid fallback policy.")
    parser.add_argument("--primary-predictions", required=True)
    parser.add_argument("--fallback-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--primary-cost-per-1000", type=float, default=0.0)
    parser.add_argument("--fallback-cost-per-1000", type=float, default=0.0)
    args = parser.parse_args()
    metrics = run_hybrid_policy(
        args.primary_predictions,
        args.fallback_predictions,
        args.output_dir,
        confidence_threshold=args.confidence_threshold,
        primary_cost_per_1000=args.primary_cost_per_1000,
        fallback_cost_per_1000=args.fallback_cost_per_1000,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
