from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from term_ai.experiment.lm_eval import run_hf_zero_shot


def compare_quantization(
    metadata_path: str | Path,
    output_dir: str | Path,
    model_name_or_path: str,
    eval_split: str = "dev",
    min_status: str = "aug_auto_pass",
    limit: int | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    results: dict[str, Any] = {}
    for mode in ("fp16", "8bit", "4bit"):
        results[mode] = run_hf_zero_shot(
            metadata_path=metadata_path,
            output_dir=output / mode,
            model_name_or_path=model_name_or_path,
            eval_split=eval_split,
            min_status=min_status,
            quantization=mode,
            limit=limit,
        )
    (output / "quantization_compare.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G4 FP16/8bit/4bit quantization comparison.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--min-status", default="aug_auto_pass")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    results = compare_quantization(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        eval_split=args.eval_split,
        min_status=args.min_status,
        limit=args.limit,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
