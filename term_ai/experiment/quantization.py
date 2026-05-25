from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from term_ai.contracts import APPROVED_AUG_STATUS
from term_ai.experiment.lm_eval import run_hf_zero_shot
from term_ai.experiment.test_lock import enforce_final_test_once


def compare_quantization(
    metadata_path: str | Path,
    output_dir: str | Path,
    model_name_or_path: str,
    adapter_path: str | Path,
    eval_split: str = "dev",
    min_status: str = APPROVED_AUG_STATUS,
    limit: int | None = None,
    g3_checkpoint_id: str | None = None,
    final_test_once: bool = True,
) -> dict[str, Any]:
    adapter = Path(adapter_path)
    if not adapter.exists():
        raise FileNotFoundError(f"G4 requires an existing G3 LoRA adapter checkpoint: {adapter}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enforce_final_test_once(output, "G4", eval_split, enabled=final_test_once)
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
            adapter_path=adapter,
            final_test_once=False,
        )
        results[mode]["g3_checkpoint_id"] = g3_checkpoint_id or str(adapter)
    (output / "quantization_compare.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G4 FP16/8bit/4bit quantization comparison.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--min-status", default=APPROVED_AUG_STATUS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--g3-checkpoint-id")
    parser.add_argument("--allow-repeat-test", action="store_true")
    args = parser.parse_args()
    results = compare_quantization(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        eval_split=args.eval_split,
        min_status=args.min_status,
        limit=args.limit,
        g3_checkpoint_id=args.g3_checkpoint_id,
        final_test_once=not args.allow_repeat_test,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
