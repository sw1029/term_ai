from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from term_ai.contracts import answer_label
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import MCQItem, load_mcq_items, prediction_row
from term_ai.experiment.ops import memory_snapshot, timed
from term_ai.experiment.progress import (
    InterruptGuard,
    ProgressLogger,
    backup_artifact,
    resolve_latest_epoch_checkpoint,
)


@dataclass
class OptionClassificationHeadConfig:
    model_name_or_path: str
    metadata_jsonl: str
    dev_metadata_jsonl: str
    output_dir: str
    min_status: str = "aug_judge_pass"
    dev_min_status: str = "aug_judge_pass"
    max_length: int = 512
    learning_rate: float = 2e-5
    epochs: int = 3
    lambda_soft: float = 0.5
    require_teacher_scores: bool = True
    resume: bool = True
    progress_interval_items: int = 1
    backup_weights: bool = True
    backup_checkpoints: bool = True


def _soft_targets(item: MCQItem, require_teacher_scores: bool = True) -> list[float]:
    scores = item.teacher_scores
    if isinstance(scores, list) and len(scores) == 4 and all(isinstance(score, (int, float)) for score in scores):
        total = float(sum(scores))
        if total > 0:
            return [float(score) / total for score in scores]
    if require_teacher_scores:
        raise ValueError(f"metadata item {item.item_id} missing valid 4-way teacher_scores")
    return [1.0 if idx == item.answer_idx else 0.0 for idx in range(4)]


def option_texts(item: MCQItem) -> list[str]:
    base = (
        f"Task: {item.task_type}\n"
        f"Word: {item.word}\n"
        f"Meaning: {item.meaning_ko}\n"
        f"Context: {item.context}\n"
    )
    return [f"{base}Candidate answer {answer_label(idx)}: {option}" for idx, option in enumerate(item.options)]


def option_classifier_rows(
    metadata_jsonl: str | Path,
    split: str,
    min_status: str = "aug_judge_pass",
    require_teacher_scores: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in load_mcq_items(metadata_jsonl, min_status=min_status):
        if item.split != split:
            continue
        rows.append(
            {
                "item_id": item.item_id,
                "texts": option_texts(item),
                "answer_idx": item.answer_idx,
                "teacher_scores": _soft_targets(item, require_teacher_scores=require_teacher_scores),
            }
        )
    return rows


def train_option_classification_head_kd(config: OptionClassificationHeadConfig) -> dict[str, Any]:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_items = [item for item in load_mcq_items(config.metadata_jsonl, min_status=config.min_status) if item.split == "train"]
    dev_items = [item for item in load_mcq_items(config.dev_metadata_jsonl, min_status=config.dev_min_status) if item.split == "dev"]
    if not train_items:
        raise ValueError("no train items for option classification head KD")
    if not dev_items:
        raise ValueError("no dev items for option classification head KD")
    progress = ProgressLogger(
        output,
        resume=config.resume,
        progress_interval_items=config.progress_interval_items,
        stage="classification_head_kd",
        total_count=len(dev_items),
    )
    completed_metrics = progress.completed_metrics_if_available(item.item_id for item in dev_items)
    if completed_metrics is not None:
        return completed_metrics
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install training dependencies first: pip install -e .[train]") from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(config.model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    classifier = torch.nn.Linear(backbone.config.hidden_size, 1).to(device)
    optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(classifier.parameters()), lr=config.learning_rate)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_model = output / "option_classification_head.pt"
    start_epoch = 0
    if config.resume and final_model.exists():
        saved = torch.load(final_model, map_location=device)
        if "backbone_state_dict" in saved:
            backbone.load_state_dict(saved["backbone_state_dict"])
        classifier.load_state_dict(saved["classifier_state_dict"])
        start_epoch = config.epochs
    elif config.resume:
        latest = resolve_latest_epoch_checkpoint(output, "option_classification_head")
        if latest is not None:
            saved = torch.load(latest, map_location=device)
            if "backbone_state_dict" in saved:
                backbone.load_state_dict(saved["backbone_state_dict"])
            classifier.load_state_dict(saved["classifier_state_dict"])
            if "optimizer_state_dict" in saved:
                optimizer.load_state_dict(saved["optimizer_state_dict"])
            start_epoch = int(saved.get("epoch", 0))

    def score_item(item: MCQItem) -> Any:
        encoded = tokenizer(
            option_texts(item),
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        ).to(device)
        outputs = backbone(**encoded)
        hidden = outputs.last_hidden_state
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=device), lengths]
        return classifier(pooled).squeeze(-1)

    def latest_checkpoint() -> Path | None:
        return resolve_latest_epoch_checkpoint(output, "option_classification_head")

    with InterruptGuard(progress, stage="classification_head_kd:training", checkpoint_callback=latest_checkpoint):
        for epoch in range(start_epoch, config.epochs):
            losses: list[float] = []
            for item in train_items:
                logits = score_item(item)
                hard = torch.tensor([item.answer_idx], dtype=torch.long, device=device)
                soft = torch.tensor(_soft_targets(item, config.require_teacher_scores), dtype=torch.float32, device=device)
                hard_loss = F.cross_entropy(logits.unsqueeze(0), hard)
                soft_loss = F.kl_div(F.log_softmax(logits, dim=0), soft, reduction="batchmean")
                loss = hard_loss + config.lambda_soft * soft_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            checkpoint_path = checkpoint_dir / f"option_classification_head_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "backbone": config.model_name_or_path,
                    "backbone_state_dict": backbone.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.__dict__,
                },
                checkpoint_path,
            )
            if config.backup_checkpoints:
                backup_artifact(checkpoint_path, output, name=f"option_classification_head_epoch_{epoch + 1}")
            progress.record_metrics(
                {
                    "epoch": epoch + 1,
                    "train_loss": sum(losses) / len(losses) if losses else 0.0,
                    "completed_epochs": epoch + 1,
                    "total_epochs": config.epochs,
                },
                stage="classification_head_kd:training",
                event="epoch",
                epoch=epoch + 1,
                latest_checkpoint=checkpoint_path,
            )

    backbone.eval()
    classifier.eval()
    with InterruptGuard(progress, stage="classification_head_kd:eval", checkpoint_callback=latest_checkpoint):
        for item in dev_items:
            if progress.has_prediction(item.item_id):
                continue
            with timed() as state:
                with torch.no_grad():
                    logits = score_item(item)
                    probs = torch.softmax(logits, dim=0).detach().cpu()
            idx = int(torch.argmax(probs).item())
            progress.append_prediction(
                prediction_row(
                    item,
                    answer_label(idx),
                    float(probs[idx]),
                    latency_ms=state["latency_ms"],
                    extra=memory_snapshot(),
                )
            )

    torch.save(
        {
            "backbone": config.model_name_or_path,
            "backbone_state_dict": backbone.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "config": config.__dict__,
        },
        final_model,
    )
    final_backup = backup_artifact(final_model, output, name="option_classification_head") if config.backup_weights else None
    predictions = progress.predictions_for_items(item.item_id for item in dev_items)
    metrics = summarize_predictions(predictions)
    progress.finalize_predictions(
        metrics,
        predictions,
        final_artifact=final_model,
        details={"final_backup": str(final_backup) if final_backup else None},
    )
    (output / "classification_head_config.json").write_text(
        json.dumps(config.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train option classification head with hard+soft KD.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--dev-metadata-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-status", default="aug_judge_pass")
    parser.add_argument("--dev-min-status", default="aug_judge_pass")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lambda-soft", type=float, default=0.5)
    parser.add_argument("--allow-missing-teacher-scores", dest="require_teacher_scores", action="store_false")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress-interval-items", type=int, default=1)
    parser.add_argument("--no-weight-backup", action="store_true")
    parser.add_argument("--no-checkpoint-backup", action="store_true")
    parser.set_defaults(require_teacher_scores=True)
    args = parser.parse_args()
    args.resume = not args.no_resume
    args.backup_weights = not args.no_weight_backup
    args.backup_checkpoints = not args.no_checkpoint_backup
    del args.no_resume
    del args.no_weight_backup
    del args.no_checkpoint_backup
    result = train_option_classification_head_kd(OptionClassificationHeadConfig(**vars(args)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
