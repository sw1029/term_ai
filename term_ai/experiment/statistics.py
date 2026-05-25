from __future__ import annotations

import argparse
import json
from pathlib import Path

from term_ai.contracts import iter_jsonl
from term_ai.experiment.metrics import compare_prediction_sets


def compare_prediction_files(
    predictions_a: str | Path,
    predictions_b: str | Path,
    output_path: str | Path,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    rows_a = list(iter_jsonl(predictions_a))
    rows_b = list(iter_jsonl(predictions_b))
    result = compare_prediction_sets(rows_a, rows_b, samples=samples, seed=seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two prediction logs with paired statistics.")
    parser.add_argument("--predictions-a", required=True)
    parser.add_argument("--predictions-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = compare_prediction_files(
        args.predictions_a,
        args.predictions_b,
        args.output,
        samples=args.samples,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
