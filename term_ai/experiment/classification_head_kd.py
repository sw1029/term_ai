from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from term_ai.contracts import answer_label, write_jsonl
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import MCQItem, load_mcq_items, prediction_row
from term_ai.experiment.ops import memory_snapshot, timed


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
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install training dependencies first: pip install -e .[train]") from exc

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_items = [item for item in load_mcq_items(config.metadata_jsonl, min_status=config.min_status) if item.split == "train"]
    dev_items = [item for item in load_mcq_items(config.dev_metadata_jsonl, min_status=config.dev_min_status) if item.split == "dev"]
    if not train_items:
        raise ValueError("no train items for option classification head KD")
    if not dev_items:
        raise ValueError("no dev items for option classification head KD")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(config.model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    classifier = torch.nn.Linear(backbone.config.hidden_size, 1).to(device)
    optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(classifier.parameters()), lr=config.learning_rate)

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

    for _ in range(config.epochs):
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

    predictions: list[dict[str, Any]] = []
    backbone.eval()
    classifier.eval()
    for item in dev_items:
        with timed() as state:
            with torch.no_grad():
                logits = score_item(item)
                probs = torch.softmax(logits, dim=0).detach().cpu()
        idx = int(torch.argmax(probs).item())
        predictions.append(
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
            "classifier_state_dict": classifier.state_dict(),
            "config": config.__dict__,
        },
        output / "option_classification_head.pt",
    )
    metrics = summarize_predictions(predictions)
    write_jsonl(output / "prediction_log.jsonl", predictions)
    (output / "metric_log.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
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
    parser.set_defaults(require_teacher_scores=True)
    args = parser.parse_args()
    result = train_option_classification_head_kd(OptionClassificationHeadConfig(**vars(args)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
