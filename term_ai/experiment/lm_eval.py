from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from term_ai.contracts import RAW_GT_STATUS
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import load_mcq_items, parse_answer_letter, prediction_row
from term_ai.experiment.ops import memory_snapshot, timed, tokens_per_second
from term_ai.experiment.progress import InterruptGuard, ProgressLogger
from term_ai.experiment.test_lock import enforce_final_test_once


def run_hf_zero_shot(
    metadata_path: str | Path,
    output_dir: str | Path,
    model_name_or_path: str,
    eval_split: str = "dev",
    min_status: str = RAW_GT_STATUS,
    max_new_tokens: int = 64,
    quantization: str | None = None,
    limit: int | None = None,
    adapter_path: str | Path | None = None,
    final_test_once: bool = True,
    experiment_id: str | None = None,
    test_lock_dir: str | Path | None = None,
    local_cost_per_hour_usd: float = 0.0,
    resume: bool = True,
    progress_interval_items: int = 1,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_experiment_id = experiment_id or ("G4" if adapter_path and quantization else "G0")
    enforce_final_test_once(
        output_dir,
        lock_experiment_id,
        eval_split,
        enabled=final_test_once,
        lock_dir=test_lock_dir,
    )

    items = [item for item in load_mcq_items(metadata_path, min_status=min_status) if item.split == eval_split]
    if not items:
        raise ValueError(f"no LM eval items: split={eval_split}, min_status={min_status}")
    if limit is not None:
        items = items[:limit]
    progress = ProgressLogger(
        output_dir,
        resume=resume,
        progress_interval_items=progress_interval_items,
        stage=f"{lock_experiment_id}:eval",
        total_count=len(items),
    )
    completed_metrics = progress.completed_metrics_if_available(item.item_id for item in items)
    if completed_metrics is not None:
        return completed_metrics

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install train dependencies first: pip install -e .[train]") from exc

    cold_start_begin = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    quant_config = None
    if quantization == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization not in {None, "fp16"}:
        raise ValueError("quantization must be fp16, 8bit, 4bit, or omitted")

    model_kwargs: dict[str, Any] = {"device_map": "auto"}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Install train dependencies first: pip install -e .[train]") from exc
        model = PeftModel.from_pretrained(model, str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cold_start_ms = (time.perf_counter() - cold_start_begin) * 1000

    with InterruptGuard(progress, stage=f"{lock_experiment_id}:eval"):
        for index, item in enumerate(items):
            if progress.has_prediction(item.item_id):
                continue
            prompt = item.prompt()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with timed() as state:
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            new_tokens = int(output.shape[-1] - inputs["input_ids"].shape[-1])
            text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            answer, confidence = parse_answer_letter(text)
            progress.append_prediction(
                prediction_row(
                    item,
                    answer or "PARSE_ERROR",
                    confidence if confidence is not None else 0.0,
                    latency_ms=state["latency_ms"],
                    extra={
                        "parse_error": answer is None,
                        "raw_response": text,
                        "tokens_per_sec": tokens_per_second(new_tokens, state["latency_ms"]),
                        "quantization": quantization or "fp16",
                        "adapter_path": str(adapter_path) if adapter_path is not None else None,
                        "batch_size": 1,
                        "cold_start_ms": cold_start_ms if index == 0 else 0.0,
                        "local_cost_per_hour_usd": local_cost_per_hour_usd,
                        **memory_snapshot(),
                    },
                )
            )

    predictions = progress.predictions_for_items(item.item_id for item in items)
    metrics = summarize_predictions(predictions)
    metrics["adapter_path"] = str(adapter_path) if adapter_path is not None else None
    metrics["quantization"] = quantization or "fp16"
    metrics["cold_start_ms"] = cold_start_ms
    metrics["local_cost_per_hour_usd"] = local_cost_per_hour_usd
    progress.finalize_predictions(
        metrics,
        predictions,
        details={
            "model_name_or_path": model_name_or_path,
            "adapter_path": str(adapter_path) if adapter_path is not None else None,
            "quantization": quantization or "fp16",
        },
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G0/G4 Hugging Face LM MCQ evaluation.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--min-status", default=RAW_GT_STATUS)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--quantization", choices=["fp16", "8bit", "4bit"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapter-path")
    parser.add_argument("--experiment-id")
    parser.add_argument("--test-lock-dir")
    parser.add_argument("--local-cost-per-hour-usd", type=float, default=0.0)
    parser.add_argument("--allow-repeat-test", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress-interval-items", type=int, default=1)
    args = parser.parse_args()
    metrics = run_hf_zero_shot(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        eval_split=args.eval_split,
        min_status=args.min_status,
        max_new_tokens=args.max_new_tokens,
        quantization=args.quantization,
        limit=args.limit,
        adapter_path=args.adapter_path,
        final_test_once=not args.allow_repeat_test,
        experiment_id=args.experiment_id,
        test_lock_dir=args.test_lock_dir,
        local_cost_per_hour_usd=args.local_cost_per_hour_usd,
        resume=not args.no_resume,
        progress_interval_items=args.progress_interval_items,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
