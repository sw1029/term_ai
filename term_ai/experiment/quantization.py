from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from term_ai.contracts import RAW_GT_STATUS
from term_ai.experiment.lm_eval import run_hf_zero_shot
from term_ai.experiment.progress import atomic_write_json, load_json
from term_ai.experiment.test_lock import enforce_final_test_once


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_g3_adapter_checkpoint(
    adapter_path: str | Path,
    g3_checkpoint_id: str | None = None,
    require_manifest: bool = True,
) -> dict[str, Any]:
    adapter = Path(adapter_path)
    if not adapter.exists():
        raise FileNotFoundError(f"G4 requires an existing G3 LoRA adapter checkpoint: {adapter}")

    candidates = [
        adapter / "g3_checkpoint_manifest.json",
        adapter.parent / "g3_checkpoint_manifest.json",
        adapter / "kd_training_config.json",
        adapter.parent / "kd_training_config.json",
        adapter / "run_manifest.json",
        adapter.parent / "run_manifest.json",
        adapter.parent.parent / "run_manifest.json",
    ]
    manifest_path: Path | None = None
    manifest: dict[str, Any] | None = None
    for candidate in candidates:
        loaded = _load_json(candidate)
        if loaded is not None:
            manifest_path = candidate
            manifest = loaded
            break

    if manifest is None:
        if require_manifest:
            raise ValueError(
                "G4 quantization requires a G3 KD manifest next to the adapter "
                "(kd_training_config.json or run_manifest.json)."
            )
        return {"adapter_path": str(adapter), "verified": False, "reason": "manifest_not_found"}

    is_kd_config = {"metadata_jsonl", "dev_metadata_jsonl", "lambda_soft"} <= set(manifest)
    is_g3_checkpoint_manifest = manifest.get("experiment_family") == "G3" and manifest.get("checkpoint_type") == "lora_sft_kd"
    is_g3_run = str(manifest.get("experiment_id", "")).startswith("G3") and bool(
        ((manifest.get("model_spec") or {}).get("uses_kd"))
    )
    if require_manifest and not (is_kd_config or is_g3_checkpoint_manifest or is_g3_run):
        raise ValueError(f"adapter manifest does not identify a G3 KD checkpoint: {manifest_path}")

    if g3_checkpoint_id:
        expected = str(g3_checkpoint_id)
        resolved_expected = str(Path(expected).resolve()) if Path(expected).exists() else expected
        resolved_adapter = str(adapter.resolve())
        if expected not in {str(adapter), resolved_adapter} and resolved_expected != resolved_adapter:
            manifest_checkpoint = str(manifest.get("checkpoint_id") or manifest.get("final_adapter") or "")
            if manifest_checkpoint != expected:
                raise ValueError(
                    f"G4 expected G3 checkpoint {expected}, but adapter path is {adapter} "
                    "and manifest does not match the expected checkpoint id."
                )

    return {
        "adapter_path": str(adapter),
        "verified": True,
        "manifest_path": str(manifest_path),
        "manifest_type": "g3_checkpoint_manifest"
        if is_g3_checkpoint_manifest
        else "kd_training_config"
        if is_kd_config
        else "run_manifest",
        "g3_checkpoint_id": g3_checkpoint_id or str(adapter),
    }


def compare_quantization(
    metadata_path: str | Path,
    output_dir: str | Path,
    model_name_or_path: str,
    adapter_path: str | Path,
    eval_split: str = "dev",
    min_status: str = RAW_GT_STATUS,
    limit: int | None = None,
    g3_checkpoint_id: str | None = None,
    require_g3_manifest: bool = True,
    final_test_once: bool = True,
    test_lock_dir: str | Path | None = None,
    local_cost_per_hour_usd: float = 0.0,
    resume: bool = True,
    progress_interval_items: int = 1,
) -> dict[str, Any]:
    checkpoint_validation = validate_g3_adapter_checkpoint(
        adapter_path,
        g3_checkpoint_id=g3_checkpoint_id,
        require_manifest=require_g3_manifest,
    )
    adapter = Path(adapter_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enforce_final_test_once(output, "G4", eval_split, enabled=final_test_once, lock_dir=test_lock_dir)
    final_compare = output / "quantization_compare.json"
    if resume:
        completed = load_json(final_compare)
        if completed is not None:
            return completed
    partial_compare = output / "quantization_compare.partial.json"
    results: dict[str, Any] = load_json(partial_compare) if resume else {}
    for mode in ("fp16", "8bit", "4bit"):
        if mode in results:
            continue
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
            experiment_id=f"G4-{mode}",
            local_cost_per_hour_usd=local_cost_per_hour_usd,
            resume=resume,
            progress_interval_items=progress_interval_items,
        )
        results[mode]["g3_checkpoint_id"] = g3_checkpoint_id or str(adapter)
        results[mode]["g3_checkpoint_validation"] = checkpoint_validation
        atomic_write_json(partial_compare, results)
    atomic_write_json(final_compare, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G4 FP16/8bit/4bit quantization comparison.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--min-status", default=RAW_GT_STATUS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--g3-checkpoint-id")
    parser.add_argument("--allow-unverified-g3-adapter", action="store_true")
    parser.add_argument("--local-cost-per-hour-usd", type=float, default=0.0)
    parser.add_argument("--test-lock-dir")
    parser.add_argument("--allow-repeat-test", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress-interval-items", type=int, default=1)
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
        require_g3_manifest=not args.allow_unverified_g3_adapter,
        final_test_once=not args.allow_repeat_test,
        test_lock_dir=args.test_lock_dir,
        local_cost_per_hour_usd=args.local_cost_per_hour_usd,
        resume=not args.no_resume,
        progress_interval_items=args.progress_interval_items,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
