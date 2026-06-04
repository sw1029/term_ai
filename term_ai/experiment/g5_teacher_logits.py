from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from term_ai.contracts import answer_label, dumps_jsonl, status_reaches
from term_ai.experiment.hf_loading import (
    bitnet_loading_config,
    clear_bitnet_quantization_training_guard,
    from_pretrained_with_trust,
    is_bitnet_config,
    repair_bitnet_autobitlinear_weights,
)
from term_ai.experiment.mcq import MCQItem
from term_ai.experiment.training import _format_chat


@dataclass
class G5TeacherLogitConfig:
    metadata_jsonl: str
    output: str
    model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"
    adapter_path: str = "runs/G3_Qwen_dev/final_adapter"
    min_status: str = "any"
    temperature: float = 2.0
    max_length: int = 1024
    limit: int | None = None
    prompt_mode: str = "chat"
    resume: bool = True
    trust_remote_code: bool = False


def _completed_item_ids(output_path: str | Path) -> set[str]:
    output = Path(output_path)
    if not output.exists():
        return set()
    completed: set[str] = set()
    with output.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid existing G5 teacher output at line {line_no}: {output}") from exc
            if row.get("item_id") is not None:
                completed.add(str(row["item_id"]))
    return completed


def _item_from_metadata_row(row: dict[str, Any], line_no: int) -> MCQItem:
    payload = row.get("payload") or {}
    options = [str(option) for option in payload.get("options") or []]
    answer_idx = payload.get("answer_idx")
    if len(options) != 4 or not isinstance(answer_idx, int):
        raise ValueError(f"metadata line {line_no} is not a 4-option MCQ")
    return MCQItem(
        item_id=str(row.get("item_id") or f"line-{line_no}"),
        split=str(row.get("split") or "unknown"),
        task_type=str(payload.get("source_task_type") or payload.get("task_type") or ""),
        word=str(payload.get("word") or ""),
        context=str(payload.get("context") or ""),
        meaning_ko=str(payload.get("meaning_ko") or ""),
        options=options,
        answer_idx=answer_idx,
        teacher_scores=payload.get("teacher_scores") or row.get("teacher_scores"),
        status=str(row.get("status") or ""),
        source=str(row.get("source") or ""),
        dataset_view=str(row.get("dataset_view") or ""),
        stress_tags=tuple(str(tag) for tag in row.get("stress_tags") or payload.get("stress_tags") or []),
    )


def _teacher_prompt(tokenizer: Any, item: MCQItem, prompt_mode: str) -> str:
    prefix = '{"answer": "'
    if prompt_mode == "plain":
        return f"{item.prompt()}\n\n{prefix}"
    if prompt_mode != "chat":
        raise ValueError("prompt_mode must be chat or plain")
    return _format_chat(tokenizer, {"messages": item.prompt_messages()}, add_generation_prompt=True) + prefix


def _option_token_ids(tokenizer: Any) -> list[int]:
    ids: list[int] = []
    for idx in range(4):
        encoded = tokenizer.encode(answer_label(idx), add_special_tokens=False)
        if not encoded:
            raise ValueError(f"tokenizer produced no token id for option {answer_label(idx)}")
        ids.append(int(encoded[0]))
    return ids


def _softmax(values: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    import math

    scaled = [float(value) / temperature for value in values]
    offset = max(scaled)
    exp_values = [math.exp(value - offset) for value in scaled]
    total = sum(exp_values)
    if total <= 0:
        raise ValueError("invalid softmax total")
    return [value / total for value in exp_values]


def _score_item(model: Any, tokenizer: Any, item: MCQItem, config: G5TeacherLogitConfig, option_token_ids: list[int]) -> tuple[list[float], list[float]]:
    import torch

    prompt = _teacher_prompt(tokenizer, item, config.prompt_mode)
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, option_token_ids].detach().float().cpu().tolist()
    scores = _softmax([float(value) for value in logits], config.temperature)
    return scores, [float(value) for value in logits]


def _load_teacher_model(config: G5TeacherLogitConfig) -> tuple[Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install training dependencies first: pip install -e .[train]") from exc

    model_config = from_pretrained_with_trust(AutoConfig, config.model_name_or_path, config.trust_remote_code)
    is_bitnet = is_bitnet_config(model_config)
    model_config = bitnet_loading_config(model_config, for_lora=True)
    tokenizer = from_pretrained_with_trust(
        AutoTokenizer,
        config.model_name_or_path,
        config.trust_remote_code,
        use_fast=True,
        **({"fix_mistral_regex": True} if is_bitnet else {}),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"device_map": "auto", "config": model_config}
    if torch.cuda.is_available() and not is_bitnet:
        model_kwargs["dtype"] = torch.float16
    base_model = from_pretrained_with_trust(
        AutoModelForCausalLM,
        config.model_name_or_path,
        config.trust_remote_code,
        **model_kwargs,
    )
    repair_bitnet_autobitlinear_weights(base_model)
    clear_bitnet_quantization_training_guard(base_model)
    model = PeftModel.from_pretrained(base_model, config.adapter_path)
    model.eval()
    return model, tokenizer


def write_g5_teacher_logits(config: G5TeacherLogitConfig) -> dict[str, Any]:
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_item_ids(output) if config.resume else set()
    output_mode = "a" if config.resume and output.exists() else "w"
    model, tokenizer = _load_teacher_model(config)
    option_token_ids = _option_token_ids(tokenizer)
    counts = {
        "written": 0,
        "skipped_existing": 0,
        "skipped_status": 0,
        "skipped_limit": 0,
    }
    started = time.perf_counter()
    with open(config.metadata_jsonl, "r", encoding="utf-8") as input_handle, output.open(
        output_mode, encoding="utf-8", newline="\n"
    ) as output_handle:
        for line_no, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue
            if config.limit is not None and counts["written"] >= config.limit:
                counts["skipped_limit"] += 1
                continue
            row = json.loads(line)
            item_id = str(row.get("item_id") or f"line-{line_no}")
            if item_id in completed:
                counts["skipped_existing"] += 1
                continue
            status = str(row.get("status") or "")
            if not status_reaches(status, config.min_status):
                counts["skipped_status"] += 1
                continue
            item = _item_from_metadata_row(row, line_no)
            scores, logits = _score_item(model, tokenizer, item, config, option_token_ids)
            payload = dict(row.get("payload") or {})
            previous_scores = row.get("teacher_scores") or payload.get("teacher_scores")
            if previous_scores is not None:
                row["previous_teacher_scores"] = previous_scores
            payload["teacher_scores"] = scores
            payload["teacher_score_source"] = "g3_qwen_answer_logits"
            payload["teacher_score_model"] = config.model_name_or_path
            payload["teacher_adapter_path"] = config.adapter_path
            payload["teacher_temperature"] = config.temperature
            row["teacher_scores"] = scores
            row["teacher_score_logits"] = logits
            row["teacher_score_source"] = "g3_qwen_answer_logits"
            row["teacher_score_model"] = config.model_name_or_path
            row["teacher_adapter_path"] = config.adapter_path
            row["teacher_temperature"] = config.temperature
            row["payload"] = payload
            output_handle.write(dumps_jsonl(row))
            counts["written"] += 1

    manifest = {
        "task": "g5_teacher_answer_logit_scores",
        "metadata_jsonl": config.metadata_jsonl,
        "output": str(output),
        "model_name_or_path": config.model_name_or_path,
        "adapter_path": config.adapter_path,
        "min_status": config.min_status,
        "temperature": config.temperature,
        "option_token_ids": option_token_ids,
        "elapsed_seconds": time.perf_counter() - started,
        "counts": counts,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate G5 KD teacher_scores from a G3 Qwen teacher adapter's A-D logits.")
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-path", default="runs/G3_Qwen_dev/final_adapter")
    parser.add_argument("--min-status", default="any")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-mode", choices=["chat", "plain"], default="chat")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    manifest = write_g5_teacher_logits(
        G5TeacherLogitConfig(
            metadata_jsonl=args.metadata_jsonl,
            output=args.output,
            model_name_or_path=args.model_name_or_path,
            adapter_path=args.adapter_path,
            min_status=args.min_status,
            temperature=args.temperature,
            max_length=args.max_length,
            limit=args.limit,
            prompt_mode=args.prompt_mode,
            resume=not args.no_resume,
            trust_remote_code=args.trust_remote_code,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
