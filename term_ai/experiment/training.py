from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class LoRATrainingConfig:
    model_name_or_path: str
    train_jsonl: str
    dev_jsonl: str
    output_dir: str
    max_length: int = 1024
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    resume_from_checkpoint: str | None = None
    backup_weights: bool = True


def _read_messages_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if set(record.keys()) != {"messages"}:
                raise ValueError(f"SFT row {line_no} violates messages-only contract")
            rows.append(record)
    return rows


def _format_chat(tokenizer: Any, record: dict[str, Any]) -> str:
    messages = record["messages"]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def train_lora_sft(config: LoRATrainingConfig) -> Path:
    """Run LoRA SFT when the optional training stack is installed.

    This function intentionally consumes only messages-only SFT JSONL. Teacher
    scores and validation metadata must be loaded by a separate KD path.
    """

    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install training dependencies first: pip install -e .[train]") from exc

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(
        json.dumps(config.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_rows = _read_messages_jsonl(config.train_jsonl)
    dev_rows = _read_messages_jsonl(config.dev_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        texts = [_format_chat(tokenizer, {"messages": messages}) for messages in batch["messages"]]
        encoded = tokenizer(texts, truncation=True, max_length=config.max_length)
        encoded["labels"] = [ids.copy() for ids in encoded["input_ids"]]
        return encoded

    train_dataset = Dataset.from_list(train_rows).map(tokenize_batch, batched=True, remove_columns=["messages"])
    dev_dataset = Dataset.from_list(dev_rows).map(tokenize_batch, batched=True, remove_columns=["messages"])

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))

    resume_state = {
        "stage": "trained",
        "final_adapter": str(output_dir / "final_adapter"),
        "resume_supported": True,
        "weight_backup_required": config.backup_weights,
    }
    (output_dir / "resume_state.json").write_text(json.dumps(resume_state, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir / "final_adapter"
