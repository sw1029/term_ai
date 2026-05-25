from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from term_ai.experiment.classification_head_kd import OptionClassificationHeadConfig, train_option_classification_head_kd
from term_ai.experiment.lora_kd import LoRAKDConfig, train_lora_sft_kd
from term_ai.experiment.lm_eval import run_hf_zero_shot


@dataclass
class KDAblationSweepConfig:
    model_name_or_path: str
    metadata_jsonl: str
    dev_metadata_jsonl: str
    output_dir: str
    min_status: str = "aug_judge_pass"
    dev_min_status: str = "aug_judge_pass"
    execute_training: bool = False
    lambda_soft: float = 0.5
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    eval_metadata: str | None = None
    eval_split: str = "dev"
    include_classification_head: bool = True


def _ablation_configs(config: KDAblationSweepConfig) -> dict[str, LoRAKDConfig]:
    base = LoRAKDConfig(
        model_name_or_path=config.model_name_or_path,
        metadata_jsonl=config.metadata_jsonl,
        dev_metadata_jsonl=config.dev_metadata_jsonl,
        output_dir=config.output_dir,
        min_status=config.min_status,
        dev_min_status=config.dev_min_status,
        lambda_soft=config.lambda_soft,
        epochs=config.epochs,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )
    return {
        "hard_only_with_rationale": replace(
            base,
            output_dir=str(Path(config.output_dir) / "hard_only_with_rationale"),
            hard_label_only=True,
            include_rationale=True,
            require_teacher_scores=False,
            lambda_soft=0.0,
            response_format="letter_reason",
        ),
        "soft_kd_with_rationale": replace(
            base,
            output_dir=str(Path(config.output_dir) / "soft_kd_with_rationale"),
            hard_label_only=False,
            include_rationale=True,
            require_teacher_scores=True,
            response_format="json_distribution",
        ),
        "soft_kd_no_rationale": replace(
            base,
            output_dir=str(Path(config.output_dir) / "soft_kd_no_rationale"),
            hard_label_only=False,
            include_rationale=False,
            require_teacher_scores=True,
            response_format="json_distribution",
        ),
    }


def run_kd_ablation_sweep(config: KDAblationSweepConfig) -> dict[str, Any]:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for name, kd_config in _ablation_configs(config).items():
        row: dict[str, Any] = {
            "ablation": name,
            "status": "planned",
            "output_dir": kd_config.output_dir,
            "hard_label_only": kd_config.hard_label_only,
            "include_rationale": kd_config.include_rationale,
            "lambda_soft": kd_config.lambda_soft,
            "response_format": kd_config.response_format,
        }
        if config.execute_training:
            adapter = train_lora_sft_kd(kd_config)
            row["status"] = "trained"
            row["final_adapter"] = str(adapter)
            if config.eval_metadata:
                metrics = run_hf_zero_shot(
                    metadata_path=config.eval_metadata,
                    output_dir=Path(kd_config.output_dir) / "post_train_eval",
                    model_name_or_path=config.model_name_or_path,
                    eval_split=config.eval_split,
                    adapter_path=adapter,
                    final_test_once=False,
                    experiment_id=f"G3-{name}",
                )
                row["post_train_eval"] = metrics
        runs.append(row)

    if config.include_classification_head:
        head_output = output / "classification_head_kd"
        row = {
            "ablation": "classification_head_kd",
            "status": "planned",
            "output_dir": str(head_output),
            "hard_label_only": False,
            "include_rationale": False,
            "lambda_soft": config.lambda_soft,
            "response_format": "option_classification_logits",
        }
        if config.execute_training:
            metrics = train_option_classification_head_kd(
                OptionClassificationHeadConfig(
                    model_name_or_path=config.model_name_or_path,
                    metadata_jsonl=config.metadata_jsonl,
                    dev_metadata_jsonl=config.dev_metadata_jsonl,
                    output_dir=str(head_output),
                    min_status=config.min_status,
                    dev_min_status=config.dev_min_status,
                    epochs=config.epochs,
                    lambda_soft=config.lambda_soft,
                )
            )
            row["status"] = "trained"
            row["post_train_eval"] = metrics
        runs.append(row)

    manifest = {
        "task": "g3_lora_kd_ablation_sweep",
        "metadata_jsonl": config.metadata_jsonl,
        "dev_metadata_jsonl": config.dev_metadata_jsonl,
        "execute_training": config.execute_training,
        "runs": runs,
    }
    (output / "kd_ablation_sweep.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G3 LoRA KD hard/soft/rationale ablation sweep.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--dev-metadata-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-status", default="aug_judge_pass")
    parser.add_argument("--dev-min-status", default="aug_judge_pass")
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--lambda-soft", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-metadata")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--skip-classification-head", dest="include_classification_head", action="store_false")
    parser.set_defaults(include_classification_head=True)
    args = parser.parse_args()
    result = run_kd_ablation_sweep(KDAblationSweepConfig(**vars(args)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
