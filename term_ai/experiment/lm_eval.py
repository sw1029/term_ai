from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from term_ai.contracts import RAW_GT_STATUS
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import (
    PARSER_VERSION,
    PROMPT_CONTRACT_VERSION,
    MCQItem,
    load_mcq_items,
    parse_answer_response,
    prediction_row,
)
from term_ai.experiment.hf_loading import (
    bitnet_loading_config,
    clear_bitnet_quantization_training_guard,
    from_pretrained_with_trust,
    is_bitnet_config,
    repair_bitnet_autobitlinear_weights,
)
from term_ai.experiment.ops import memory_snapshot, timed, tokens_per_second
from term_ai.experiment.progress import InterruptGuard, ProgressLogger
from term_ai.experiment.test_lock import enforce_final_test_once
from term_ai.experiment.training import _format_chat


def _format_eval_prompt(tokenizer: Any, item: MCQItem, prompt_mode: str) -> tuple[str, str]:
    if prompt_mode == "plain":
        return item.prompt(), "plain"
    if prompt_mode != "chat":
        raise ValueError("prompt_mode must be chat or plain")
    if not hasattr(tokenizer, "apply_chat_template"):
        return item.prompt(), "plain_fallback_no_chat_template"
    try:
        return _format_chat(tokenizer, {"messages": item.prompt_messages()}, add_generation_prompt=True), "chat"
    except Exception:
        return item.prompt(), "plain_fallback_chat_template_error"


def _generation_pad_token_id(tokenizer: Any) -> int | None:
    for attr in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            return value
    return None


def _clear_generation_max_length(model: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "max_length"):
        generation_config.max_length = None


def _decode_generated_text(tokenizer: Any, token_ids: Any) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


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
    prompt_mode: str = "chat",
    trust_remote_code: bool = False,
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
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install train dependencies first: pip install -e .[train]") from exc

    cold_start_begin = time.perf_counter()
    model_config = from_pretrained_with_trust(AutoConfig, model_name_or_path, trust_remote_code)
    is_bitnet = is_bitnet_config(model_config)
    if is_bitnet and quantization in {"8bit", "4bit"}:
        raise ValueError(
            "BitNet checkpoints are already BitNet-quantized and cannot be evaluated with "
            "bitsandbytes 8bit/4bit G4 modes. Set RUN_BITNET_G4=0 for the batch runner."
        )
    model_config = bitnet_loading_config(model_config, for_lora=adapter_path is not None)
    tokenizer = from_pretrained_with_trust(
        AutoTokenizer,
        model_name_or_path,
        trust_remote_code,
        **({"fix_mistral_regex": True} if is_bitnet else {}),
    )
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
    elif torch.cuda.is_available() and not is_bitnet:
        model_kwargs["dtype"] = torch.float16
    model_kwargs["config"] = model_config
    model = from_pretrained_with_trust(
        AutoModelForCausalLM,
        model_name_or_path,
        trust_remote_code,
        **model_kwargs,
    )
    repair_bitnet_autobitlinear_weights(model)
    clear_bitnet_quantization_training_guard(model)
    _clear_generation_max_length(model)
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Install train dependencies first: pip install -e .[train]") from exc
        model = PeftModel.from_pretrained(model, str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = _generation_pad_token_id(tokenizer)
    cold_start_ms = (time.perf_counter() - cold_start_begin) * 1000

    with InterruptGuard(progress, stage=f"{lock_experiment_id}:eval"):
        for index, item in enumerate(items):
            if progress.has_prediction(item.item_id):
                continue
            prompt, prompt_mode_effective = _format_eval_prompt(tokenizer, item, prompt_mode)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            with timed() as state:
                with torch.no_grad():
                    output = model.generate(**inputs, **generation_kwargs)
            new_tokens = int(output.shape[-1] - inputs["input_ids"].shape[-1])
            text = _decode_generated_text(tokenizer, output[0][inputs["input_ids"].shape[-1] :])
            parsed = parse_answer_response(text)
            progress.append_prediction(
                prediction_row(
                    item,
                    parsed.answer or "PARSE_ERROR",
                    parsed.confidence if parsed.confidence is not None else 0.0,
                    latency_ms=state["latency_ms"],
                    extra={
                        "parse_error": parsed.parse_error,
                        "raw_response": text,
                        "parser_version": parsed.parser_version,
                        "parse_method": parsed.parse_method,
                        "strict_parse_error": parsed.strict_parse_error,
                        "confidence_normalized_from_percent": parsed.confidence_normalized_from_percent,
                        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
                        "prompt_mode": prompt_mode,
                        "prompt_mode_effective": prompt_mode_effective,
                        "model_name_or_path": model_name_or_path,
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
    metrics["model_name_or_path"] = model_name_or_path
    metrics["cold_start_ms"] = cold_start_ms
    metrics["local_cost_per_hour_usd"] = local_cost_per_hour_usd
    metrics["parser_version"] = PARSER_VERSION
    metrics["prompt_contract_version"] = PROMPT_CONTRACT_VERSION
    metrics["prompt_mode"] = prompt_mode
    metrics["trust_remote_code"] = trust_remote_code
    prompt_modes = sorted({str(row.get("prompt_mode_effective")) for row in predictions})
    metrics["prompt_mode_effective"] = prompt_modes[0] if len(prompt_modes) == 1 else prompt_modes
    progress.finalize_predictions(
        metrics,
        predictions,
        details={
            "model_name_or_path": model_name_or_path,
            "adapter_path": str(adapter_path) if adapter_path is not None else None,
            "quantization": quantization or "fp16",
            "parser_version": PARSER_VERSION,
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "prompt_mode": prompt_mode,
            "prompt_mode_effective": metrics["prompt_mode_effective"],
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
    parser.add_argument("--prompt-mode", choices=["chat", "plain"], default="chat")
    parser.add_argument("--trust-remote-code", action="store_true")
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
        prompt_mode=args.prompt_mode,
        trust_remote_code=args.trust_remote_code,
        resume=not args.no_resume,
        progress_interval_items=args.progress_interval_items,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
