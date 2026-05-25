from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from term_ai.contracts import APPROVED_AUG_STATUS, write_jsonl
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import load_mcq_items, parse_answer_letter, prediction_row
from term_ai.experiment.ops import memory_snapshot, timed, tokens_per_second
from term_ai.experiment.test_lock import enforce_final_test_once


def run_hf_zero_shot(
    metadata_path: str | Path,
    output_dir: str | Path,
    model_name_or_path: str,
    eval_split: str = "dev",
    min_status: str = APPROVED_AUG_STATUS,
    max_new_tokens: int = 64,
    quantization: str | None = None,
    limit: int | None = None,
    adapter_path: str | Path | None = None,
    final_test_once: bool = True,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install train dependencies first: pip install -e .[train]") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enforce_final_test_once(
        output_dir,
        "G4" if adapter_path and quantization else "G0",
        eval_split,
        enabled=final_test_once,
    )

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

    items = [item for item in load_mcq_items(metadata_path, min_status=min_status) if item.split == eval_split]
    if limit is not None:
        items = items[:limit]
    predictions: list[dict[str, Any]] = []

    for item in items:
        prompt = item.prompt()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with timed() as state:
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = int(output.shape[-1] - inputs["input_ids"].shape[-1])
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        answer, confidence = parse_answer_letter(text)
        predictions.append(
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
                    **memory_snapshot(),
                },
            )
        )

    metrics = summarize_predictions(predictions)
    metrics["adapter_path"] = str(adapter_path) if adapter_path is not None else None
    metrics["quantization"] = quantization or "fp16"
    write_jsonl(output_dir / "prediction_log.jsonl", predictions)
    (output_dir / "metric_log.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G0/G4 Hugging Face LM MCQ evaluation.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--min-status", default=APPROVED_AUG_STATUS)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--quantization", choices=["fp16", "8bit", "4bit"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapter-path")
    parser.add_argument("--allow-repeat-test", action="store_true")
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
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
