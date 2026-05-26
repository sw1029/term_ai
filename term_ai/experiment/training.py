from __future__ import annotations

from dataclasses import dataclass
import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from term_ai.experiment.progress import (
    InterruptGuard,
    ProgressLogger,
    backup_artifact,
    resolve_latest_checkpoint,
    utc_timestamp,
)


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
    resume: bool = True
    backup_weights: bool = True
    backup_checkpoints: bool = True
    save_steps: int | None = None
    save_total_limit: int = 3
    early_stopping_patience: int | None = 2
    eval_metadata: str | None = None
    eval_split: str = "dev"
    progress_interval_items: int = 1


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


def _make_trainer_progress_callback(
    trainer_callback_cls: type,
    progress: ProgressLogger,
    *,
    backup_checkpoints: bool = True,
) -> Any:
    class ExperimentProgressCallback(trainer_callback_cls):  # type: ignore[misc]
        def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            progress.write_state(
                "running",
                "training",
                completed_count=int(getattr(state, "global_step", 0) or 0),
                total_count=int(getattr(state, "max_steps", 0) or 0) or None,
            )
            return control

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any) -> Any:
            if logs:
                progress.record_metrics(
                    dict(logs),
                    stage="training",
                    event="log",
                    step=int(getattr(state, "global_step", 0) or 0),
                    epoch=getattr(state, "epoch", None),
                )
            return control

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict[str, Any] | None = None,
            **_: Any,
        ) -> Any:
            if metrics:
                progress.record_metrics(
                    dict(metrics),
                    stage="evaluation",
                    event="evaluate",
                    step=int(getattr(state, "global_step", 0) or 0),
                    epoch=getattr(state, "epoch", None),
                )
            return control

        def on_save(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            checkpoint = Path(args.output_dir) / f"checkpoint-{int(getattr(state, 'global_step', 0) or 0)}"
            backup_path: Path | None = None
            if checkpoint.exists() and backup_checkpoints:
                backup_path = backup_artifact(checkpoint, progress.output_dir, name=checkpoint.name)
            progress.record_metrics(
                {"global_step": int(getattr(state, "global_step", 0) or 0), "checkpoint_saved": True},
                stage="checkpoint",
                event="checkpoint",
                step=int(getattr(state, "global_step", 0) or 0),
                epoch=getattr(state, "epoch", None),
                latest_checkpoint=checkpoint if checkpoint.exists() else None,
            )
            if backup_path is not None:
                progress.write_state(
                    "running",
                    "checkpoint",
                    latest_checkpoint=checkpoint,
                    details={"checkpoint_backup": str(backup_path)},
                )
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            latest = resolve_latest_checkpoint(progress.output_dir)
            progress.write_state(
                "running",
                "trained",
                completed_count=int(getattr(state, "global_step", 0) or 0),
                total_count=int(getattr(state, "max_steps", 0) or 0) or None,
                latest_checkpoint=latest,
            )
            return control

    return ExperimentProgressCallback()


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
            EarlyStoppingCallback,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install training dependencies first: pip install -e .[train]") from exc

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressLogger(
        output_dir,
        resume=config.resume,
        progress_interval_items=config.progress_interval_items,
        stage="lora_sft:training",
    )
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
    }
    if config.save_steps is not None:
        training_kwargs["save_steps"] = int(config.save_steps)
    strategy_name = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    training_kwargs[strategy_name] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    callbacks = []
    if config.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))
    callbacks.append(
        _make_trainer_progress_callback(
            TrainerCallback,
            progress,
            backup_checkpoints=config.backup_checkpoints,
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=callbacks,
    )
    auto_resume_checkpoint = resolve_latest_checkpoint(output_dir) if config.resume else None
    resume_checkpoint = config.resume_from_checkpoint or (str(auto_resume_checkpoint) if auto_resume_checkpoint else None)

    def save_interrupt_checkpoint() -> Path:
        salvage = output_dir / "checkpoints" / f"interrupt-{utc_timestamp()}"
        trainer.save_model(str(salvage))
        tokenizer.save_pretrained(str(salvage))
        return salvage

    with InterruptGuard(progress, stage="lora_sft:training", checkpoint_callback=save_interrupt_checkpoint):
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))
    final_backup = backup_artifact(final_adapter, output_dir, name="final_adapter") if config.backup_weights else None

    resume_state = {
        "stage": "trained",
        "final_adapter": str(final_adapter),
        "final_adapter_backup": str(final_backup) if final_backup else None,
        "latest_checkpoint": str(resolve_latest_checkpoint(output_dir)) if resolve_latest_checkpoint(output_dir) else None,
        "resume_supported": True,
        "weight_backup_required": config.backup_weights,
    }
    (output_dir / "resume_state.json").write_text(json.dumps(resume_state, ensure_ascii=False, indent=2), encoding="utf-8")
    if config.eval_metadata:
        if config.eval_split == "test":
            raise ValueError("post-train auto evaluation must not use final test; run the locked final evaluation separately")
        from term_ai.experiment.lm_eval import run_hf_zero_shot

        eval_metrics = run_hf_zero_shot(
            metadata_path=config.eval_metadata,
            output_dir=output_dir / "post_train_eval",
            model_name_or_path=config.model_name_or_path,
            eval_split=config.eval_split,
            adapter_path=output_dir / "final_adapter",
            final_test_once=False,
            resume=config.resume,
            progress_interval_items=config.progress_interval_items,
        )
        (output_dir / "post_train_eval_metrics.json").write_text(
            json.dumps(eval_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
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
    parser = argparse.ArgumentParser(description="Run LoRA SFT training.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--dev-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--no-weight-backup", action="store_true")
    parser.add_argument("--no-checkpoint-backup", action="store_true")
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--eval-metadata")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--progress-interval-items", type=int, default=1)
    args = parser.parse_args()
    adapter = train_lora_sft(
        LoRATrainingConfig(
            model_name_or_path=args.model_name_or_path,
            train_jsonl=args.train_jsonl,
            dev_jsonl=args.dev_jsonl,
            output_dir=args.output_dir,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            resume_from_checkpoint=args.resume_from_checkpoint,
            resume=not args.no_resume,
            backup_weights=not args.no_weight_backup,
            backup_checkpoints=not args.no_checkpoint_backup,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            early_stopping_patience=args.early_stopping_patience,
            eval_metadata=args.eval_metadata,
            eval_split=args.eval_split,
            progress_interval_items=args.progress_interval_items,
        )
    )
    print(json.dumps({"final_adapter": str(adapter)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
