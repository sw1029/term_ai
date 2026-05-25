from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from typing import Any

from term_ai.augmentation.sft_builder import candidate_payload_to_sft_record
from term_ai.contracts import RAW_GT_STATUS, answer_label, iter_jsonl, status_reaches


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
        sft = candidate_payload_to_sft_record(payload)
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


def train_lora_sft_kd(config: LoRAKDConfig) -> Path:
    try:
        import torch
        import torch.nn.functional as F
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install training dependencies first: pip install -e .[train]") from exc

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    def chat_text(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        suffix = "\nassistant:" if add_generation_prompt else ""
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + suffix

    def encode_row(row: dict[str, Any]) -> dict[str, Any]:
        full_text = chat_text(row["messages"], add_generation_prompt=False)
        prompt_text = chat_text(row["messages"][:2], add_generation_prompt=True)
        full = tokenizer(full_text, truncation=True, max_length=config.max_length)
        prompt = tokenizer(prompt_text, truncation=True, max_length=config.max_length)
        answer_pos = min(len(prompt["input_ids"]), len(full["input_ids"]) - 1)
        for idx in range(answer_pos, len(full["input_ids"])):
            if full["input_ids"][idx] in letter_token_ids:
                answer_pos = idx
                break
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
        batch["labels"] = batch["input_ids"].clone()
        batch["answer_pos"] = torch.tensor([feature["answer_pos"] for feature in features], dtype=torch.long)
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
    training_kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.epochs,
        "logging_dir": str(output_dir / "logs"),
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "report_to": [],
    }
    strategy_name = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    training_kwargs[strategy_name] = "epoch"
    training_args = TrainingArguments(**training_kwargs)
    trainer = SoftAnswerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=collate,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))
    (output_dir / "kd_training_config.json").write_text(
        json.dumps(config.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
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
    args = parser.parse_args()
    adapter = train_lora_sft_kd(LoRAKDConfig(**vars(args)))
    print(json.dumps({"final_adapter": str(adapter)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
