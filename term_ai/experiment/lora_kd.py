from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import re
from typing import Any

from term_ai.augmentation.sft_builder import candidate_payload_to_sft_record
from term_ai.contracts import RAW_GT_STATUS, answer_label, iter_jsonl, status_reaches
from term_ai.experiment.progress import (
    InterruptGuard,
    ProgressLogger,
    backup_artifact,
    resolve_latest_checkpoint,
    utc_timestamp,
)
from term_ai.experiment.training import (
    _format_chat,
    _make_trainer_progress_callback,
    _pad_completion_labels,
    _trainer_tokenizer_kwargs,
    _tokenize_chat_completion,
)


@dataclass
class LoRAKDConfig:
    model_name_or_path: str
    metadata_jsonl: str
    dev_metadata_jsonl: str
    output_dir: str
    min_status: str = "aug_judge_pass"
    dev_min_status: str = "aug_judge_pass"
    max_length: int = 1024
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lambda_soft: float = 0.5
    hard_label_only: bool = False
    include_rationale: bool = True
    require_teacher_scores: bool = True
    response_format: str = "json_distribution"
    resume_from_checkpoint: str | None = None
    resume: bool = True
    backup_weights: bool = True
    backup_checkpoints: bool = True
    save_steps: int | None = None
    save_total_limit: int = 3
    progress_interval_items: int = 1


def _soft_scores(row: dict[str, Any], answer_idx: int, require_teacher_scores: bool) -> list[float]:
    payload = row.get("payload") or {}
    scores = row.get("teacher_scores") or payload.get("teacher_scores")
    if isinstance(scores, list) and len(scores) == 4 and all(isinstance(score, (int, float)) for score in scores):
        total = float(sum(scores))
        if total > 0:
            return [float(score) / total for score in scores]
    if require_teacher_scores:
        raise ValueError(f"metadata item {row.get('item_id')} missing valid 4-way teacher_scores")
    return [1.0 if idx == answer_idx else 0.0 for idx in range(4)]


def metadata_to_kd_rows(
    metadata_path: str | Path,
    min_status: str = "aug_judge_pass",
    include_rationale: bool = True,
    require_teacher_scores: bool = True,
    response_format: str = "json_distribution",
) -> list[dict[str, Any]]:
    if response_format not in {"json_distribution", "letter_reason"}:
        raise ValueError("response_format must be json_distribution or letter_reason")
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(metadata_path):
        status = str(row.get("status") or "")
        if min_status == RAW_GT_STATUS:
            if status != RAW_GT_STATUS:
                continue
        elif not status_reaches(status, min_status):
            continue
        payload = dict(row.get("payload") or {})
        answer_idx = payload.get("answer_idx")
        if not isinstance(answer_idx, int):
            continue
        scores = _soft_scores(row, answer_idx, require_teacher_scores=require_teacher_scores)
        if not include_rationale:
            payload["rationale"] = f"정답은 {answer_label(answer_idx)}입니다."
        sft = candidate_payload_to_sft_record(
            payload,
            response_format="letter_reason" if response_format == "letter_reason" else "json_answer",
        )
        if response_format == "json_distribution":
            label = answer_label(answer_idx)
            assistant: dict[str, Any] = {
                "answer": label,
                "confidence": scores[answer_idx],
                "distribution": {answer_label(idx): scores[idx] for idx in range(4)},
            }
            if include_rationale:
                assistant["rationale"] = str(payload.get("rationale") or payload.get("teacher_rationale") or "").strip()
            sft["messages"][2]["content"] = json.dumps(assistant, ensure_ascii=False, sort_keys=True)
        rows.append(
            {
                "item_id": row.get("item_id"),
                "messages": sft["messages"],
                "answer_idx": answer_idx,
                "teacher_scores": scores,
            }
        )
    return rows


def _lora_kd_training_kwargs(config: LoRAKDConfig, output_dir: Path) -> dict[str, Any]:
    training_kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.epochs,
        "logging_dir": str(output_dir / "logs"),
        "logging_steps": 10,
        "save_strategy": "steps" if config.save_steps is not None else "epoch",
        "save_total_limit": config.save_total_limit,
        "report_to": [],
        "remove_unused_columns": False,
    }
    if config.save_steps is not None:
        training_kwargs["save_steps"] = int(config.save_steps)
    return training_kwargs


def _answer_value_char_start(assistant_content: str, expected_label: str) -> int | None:
    try:
        parsed = json.loads(assistant_content)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        parsed_answer = str(parsed.get("answer") or "").strip().upper()
        if parsed_answer != expected_label:
            return None
    for match in re.finditer(r'"answer"\s*:\s*"([ABCD])"', assistant_content, flags=re.IGNORECASE):
        if match.group(1).upper() == expected_label:
            return match.start(1)
    return None


def _find_answer_token_position(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    input_ids: list[int],
    assistant_start: int,
    answer_idx: int,
    letter_token_ids: list[int],
    max_length: int,
) -> int | None:
    expected_token_id = int(letter_token_ids[answer_idx])
    expected_label = answer_label(answer_idx)
    assistant_content = str(messages[-1].get("content") or "") if messages else ""
    value_start = _answer_value_char_start(assistant_content, expected_label)
    if value_start is not None:
        prefix_messages = [dict(message) for message in messages]
        prefix_messages[-1]["content"] = assistant_content[:value_start]
        prefix_text = _format_chat(tokenizer, {"messages": prefix_messages}, add_generation_prompt=False)
        prefix_ids = tokenizer(prefix_text, truncation=True, max_length=max_length)["input_ids"]
        window_start = max(int(assistant_start), min(len(prefix_ids) - 2, len(input_ids)))
        window_stop = min(len(input_ids), max(window_start + 1, len(prefix_ids) + 4))
        for idx in range(window_start, window_stop):
            if int(input_ids[idx]) == expected_token_id:
                return idx

    for idx in range(max(int(assistant_start), 0), len(input_ids)):
        if int(input_ids[idx]) == expected_token_id:
            return idx
    return None


def train_lora_sft_kd(config: LoRAKDConfig) -> Path:
    try:
        import torch
        import torch.nn.functional as F
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install training dependencies first: pip install -e .[train]") from exc

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressLogger(
        output_dir,
        resume=config.resume,
        progress_interval_items=config.progress_interval_items,
        stage="lora_kd:training",
    )
    train_rows = metadata_to_kd_rows(
        config.metadata_jsonl,
        min_status=config.min_status,
        include_rationale=config.include_rationale,
        require_teacher_scores=config.require_teacher_scores and not config.hard_label_only,
        response_format=config.response_format,
    )
    dev_rows = metadata_to_kd_rows(
        config.dev_metadata_jsonl,
        min_status=config.dev_min_status,
        include_rationale=config.include_rationale,
        require_teacher_scores=config.require_teacher_scores and not config.hard_label_only,
        response_format=config.response_format,
    )
    if not train_rows:
        raise ValueError("no training rows for LoRA KD")
    if not dev_rows:
        raise ValueError("no dev rows for LoRA KD")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_token_ids = [
        tokenizer.encode(answer_label(idx), add_special_tokens=False)[0]
        for idx in range(4)
    ]

    def encode_row(row: dict[str, Any]) -> dict[str, Any]:
        full = _tokenize_chat_completion(
            tokenizer,
            row["messages"],
            config.max_length,
            normalize_assistant_json=False,
        )
        answer_pos = _find_answer_token_position(
            tokenizer,
            row["messages"],
            list(full["input_ids"]),
            int(full["assistant_start"]),
            int(row["answer_idx"]),
            letter_token_ids,
            config.max_length,
        )
        if answer_pos is None:
            raise ValueError(f"metadata item {row.get('item_id')} has no answer token in assistant JSON")
        full["answer_pos"] = answer_pos
        full["answer_idx"] = int(row["answer_idx"])
        full["teacher_scores"] = row["teacher_scores"]
        return full

    train_dataset = Dataset.from_list(train_rows).map(encode_row, remove_columns=list(train_rows[0].keys()))
    dev_dataset = Dataset.from_list(dev_rows).map(encode_row, remove_columns=list(dev_rows[0].keys()))

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = tokenizer.pad(
            [{"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]} for feature in features],
            return_tensors="pt",
        )
        max_length = int(batch["input_ids"].shape[1])
        batch["labels"] = torch.tensor(_pad_completion_labels(tokenizer, features, max_length), dtype=torch.long)
        padding_side = str(getattr(tokenizer, "padding_side", "right") or "right")
        answer_positions = []
        for feature in features:
            shift = max_length - len(feature["input_ids"]) if padding_side == "left" else 0
            answer_positions.append(int(feature["answer_pos"]) + shift)
        batch["answer_pos"] = torch.tensor(answer_positions, dtype=torch.long)
        batch["answer_idx"] = torch.tensor([feature["answer_idx"] for feature in features], dtype=torch.long)
        batch["teacher_scores"] = torch.tensor([feature["teacher_scores"] for feature in features], dtype=torch.float32)
        return batch

    class SoftAnswerTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **_: Any) -> Any:
            answer_pos = inputs.pop("answer_pos")
            teacher_scores = inputs.pop("teacher_scores")
            inputs.pop("answer_idx")
            outputs = model(**inputs)
            loss = outputs.loss
            if not config.hard_label_only and config.lambda_soft > 0:
                logits = outputs.logits
                batch_indices = torch.arange(logits.shape[0], device=logits.device)
                position_logits = logits[batch_indices, answer_pos.to(logits.device) - 1]
                option_logits = position_logits[:, letter_token_ids]
                soft_loss = F.kl_div(
                    F.log_softmax(option_logits, dim=-1),
                    teacher_scores.to(option_logits.device),
                    reduction="batchmean",
                )
                loss = loss + config.lambda_soft * soft_loss
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    training_kwargs = _lora_kd_training_kwargs(config, output_dir)
    strategy_name = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    training_kwargs[strategy_name] = "epoch"
    training_args = TrainingArguments(**training_kwargs)
    trainer = SoftAnswerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        **_trainer_tokenizer_kwargs(SoftAnswerTrainer, tokenizer),
        data_collator=collate,
        callbacks=[
            _make_trainer_progress_callback(
                TrainerCallback,
                progress,
                backup_checkpoints=config.backup_checkpoints,
            )
        ],
    )
    auto_resume_checkpoint = resolve_latest_checkpoint(output_dir) if config.resume else None
    resume_checkpoint = config.resume_from_checkpoint or (str(auto_resume_checkpoint) if auto_resume_checkpoint else None)

    def save_interrupt_checkpoint() -> Path:
        salvage = output_dir / "checkpoints" / f"interrupt-{utc_timestamp()}"
        trainer.save_model(str(salvage))
        tokenizer.save_pretrained(str(salvage))
        return salvage

    with InterruptGuard(progress, stage="lora_kd:training", checkpoint_callback=save_interrupt_checkpoint):
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))
    final_backup = backup_artifact(final_adapter, output_dir, name="final_adapter") if config.backup_weights else None
    (output_dir / "kd_training_config.json").write_text(
        json.dumps(config.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checkpoint_manifest = {
        "experiment_family": "G3",
        "checkpoint_type": "lora_sft_kd",
        "final_adapter": str(final_adapter),
        "model_name_or_path": config.model_name_or_path,
        "metadata_jsonl": config.metadata_jsonl,
        "dev_metadata_jsonl": config.dev_metadata_jsonl,
        "lambda_soft": config.lambda_soft,
        "hard_label_only": config.hard_label_only,
        "include_rationale": config.include_rationale,
        "response_format": config.response_format,
        "latest_checkpoint": str(resolve_latest_checkpoint(output_dir)) if resolve_latest_checkpoint(output_dir) else None,
        "final_adapter_backup": str(final_backup) if final_backup else None,
    }
    (output_dir / "g3_checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress.write_state(
        "completed",
        "trained",
        final_artifact=final_adapter,
        latest_checkpoint=resolve_latest_checkpoint(output_dir),
        details={"final_adapter_backup": str(final_backup) if final_backup else None},
    )
    return final_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G3 LoRA SFT with teacher soft-score KD.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--dev-metadata-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-status", default="aug_judge_pass")
    parser.add_argument("--dev-min-status", default="aug_judge_pass")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lambda-soft", type=float, default=0.5)
    parser.add_argument("--hard-label-only", action="store_true")
    parser.add_argument("--drop-rationale", dest="include_rationale", action="store_false")
    parser.set_defaults(include_rationale=True)
    parser.add_argument("--allow-missing-teacher-scores", dest="require_teacher_scores", action="store_false")
    parser.set_defaults(require_teacher_scores=True)
    parser.add_argument("--response-format", choices=["json_distribution", "letter_reason"], default="json_distribution")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-weight-backup", action="store_true")
    parser.add_argument("--no-checkpoint-backup", action="store_true")
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--progress-interval-items", type=int, default=1)
    args = parser.parse_args()
    args.resume = not args.no_resume
    args.backup_weights = not args.no_weight_backup
    args.backup_checkpoints = not args.no_checkpoint_backup
    del args.no_resume
    del args.no_weight_backup
    del args.no_checkpoint_backup
    adapter = train_lora_sft_kd(LoRAKDConfig(**vars(args)))
    print(json.dumps({"final_adapter": str(adapter)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
