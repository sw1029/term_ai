from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from term_ai.augmentation.prompt_variation import PROMPT_TEMPLATE_VARIANTS, write_sft_prompt_variants
from term_ai.experiment.training import LoRATrainingConfig, train_lora_sft


@dataclass
class PromptVariationSweepConfig:
    train_jsonl: str
    dev_jsonl: str
    output_dir: str
    model_name_or_path: str | None = None
    variants: list[str] | None = None
    execute_training: bool = False
    eval_metadata: str | None = None
    eval_split: str = "dev"
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    resume: bool = True
    save_steps: int | None = None
    save_total_limit: int = 3
    backup_weights: bool = True
    backup_checkpoints: bool = True
    progress_interval_items: int = 1


def _variant_path(source_jsonl: str | Path, variant_dir: Path, variant: str) -> Path:
    return variant_dir / f"{Path(source_jsonl).stem}_{variant}.jsonl"


def run_prompt_variation_sweep(config: PromptVariationSweepConfig) -> dict[str, Any]:
    selected = config.variants or sorted(PROMPT_TEMPLATE_VARIANTS)
    output = Path(config.output_dir)
    train_variant_dir = output / "train_variants"
    dev_variant_dir = output / "dev_variants"
    train_counts = write_sft_prompt_variants(config.train_jsonl, train_variant_dir, variants=selected)
    dev_counts = write_sft_prompt_variants(config.dev_jsonl, dev_variant_dir, variants=selected)

    if config.execute_training and not config.model_name_or_path:
        raise ValueError("model_name_or_path is required when execute_training=True")

    runs: list[dict[str, Any]] = []
    for variant in selected:
        train_variant = _variant_path(config.train_jsonl, train_variant_dir, variant)
        dev_variant = _variant_path(config.dev_jsonl, dev_variant_dir, variant)
        row: dict[str, Any] = {
            "variant": variant,
            "train_jsonl": str(train_variant),
            "dev_jsonl": str(dev_variant),
            "train_count": train_counts.get(variant, 0),
            "dev_count": dev_counts.get(variant, 0),
            "status": "planned",
        }
        if config.execute_training:
            variant_output = output / "runs" / variant
            adapter = train_lora_sft(
                LoRATrainingConfig(
                    model_name_or_path=str(config.model_name_or_path),
                    train_jsonl=str(train_variant),
                    dev_jsonl=str(dev_variant),
                    output_dir=str(variant_output),
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    lora_r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    resume=config.resume,
                    save_steps=config.save_steps,
                    save_total_limit=config.save_total_limit,
                    backup_weights=config.backup_weights,
                    backup_checkpoints=config.backup_checkpoints,
                    eval_metadata=config.eval_metadata,
                    eval_split=config.eval_split,
                    progress_interval_items=config.progress_interval_items,
                )
            )
            row["status"] = "trained"
            row["final_adapter"] = str(adapter)
            row["run_dir"] = str(variant_output)
        runs.append(row)

    manifest = {
        "task": "prompt_template_variation_sweep",
        "train_jsonl": config.train_jsonl,
        "dev_jsonl": config.dev_jsonl,
        "execute_training": config.execute_training,
        "model_name_or_path": config.model_name_or_path,
        "variants": selected,
        "runs": runs,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt_variation_sweep.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt-template variation SFT sweep.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--dev-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--variants", nargs="*", choices=sorted(PROMPT_TEMPLATE_VARIANTS))
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--eval-metadata")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--no-weight-backup", action="store_true")
    parser.add_argument("--no-checkpoint-backup", action="store_true")
    parser.add_argument("--progress-interval-items", type=int, default=1)
    args = parser.parse_args()
    args.resume = not args.no_resume
    args.backup_weights = not args.no_weight_backup
    args.backup_checkpoints = not args.no_checkpoint_backup
    del args.no_resume
    del args.no_weight_backup
    del args.no_checkpoint_backup
    result = run_prompt_variation_sweep(PromptVariationSweepConfig(**vars(args)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
