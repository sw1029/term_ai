from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from term_ai.contracts import answer_label, write_jsonl
from term_ai.experiment.baselines import _item_features, _load_embedder
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import MCQItem, load_mcq_items, prediction_row
from term_ai.experiment.ops import memory_snapshot, timed
from term_ai.experiment.test_lock import enforce_final_test_once


class TorchMLPScorer:
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        import torch

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )


def _soft_targets(item: MCQItem) -> np.ndarray:
    if item.teacher_scores and len(item.teacher_scores) == 4:
        scores = np.asarray(item.teacher_scores, dtype=np.float32)
        total = float(scores.sum())
        return scores / total if total > 0 else np.eye(4, dtype=np.float32)[item.answer_idx]
    return np.eye(4, dtype=np.float32)[item.answer_idx]


def train_kd_scorer(
    metadata_path: str | Path,
    output_dir: str | Path,
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
    train_split: str = "train",
    eval_split: str = "dev",
    min_status: str = "aug_judge_pass",
    epochs: int = 50,
    lambda_soft: float = 0.5,
    mu_margin: float = 0.2,
    final_test_once: bool = True,
    test_lock_dir: str | Path | None = None,
    require_teacher_scores: bool = True,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enforce_final_test_once(output, "E1", eval_split, enabled=final_test_once, lock_dir=test_lock_dir)

    items = load_mcq_items(metadata_path, min_status=min_status)
    if require_teacher_scores:
        items = [item for item in items if item.teacher_scores and len(item.teacher_scores) == 4]
    train_items = [item for item in items if item.split == train_split]
    eval_items = [item for item in items if item.split == eval_split]
    if not train_items:
        raise ValueError("no train items for KD scorer; check min_status and teacher_scores")
    if not eval_items:
        raise ValueError("no eval items for KD scorer; check eval_split, min_status, and teacher_scores")
    embedder = _load_embedder(embedding_model)

    sample_features = _item_features(embedder, train_items[0])
    scorer = TorchMLPScorer(input_dim=sample_features.shape[1]).model
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3)

    for _ in range(epochs):
        for item in train_items:
            features = torch.tensor(_item_features(embedder, item), dtype=torch.float32)
            logits = scorer(features).squeeze(-1)
            hard = torch.tensor([item.answer_idx], dtype=torch.long)
            soft = torch.tensor(_soft_targets(item), dtype=torch.float32)
            hard_loss = F.cross_entropy(logits.unsqueeze(0), hard)
            soft_loss = F.kl_div(F.log_softmax(logits, dim=0), soft, reduction="batchmean")
            correct = logits[item.answer_idx]
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[item.answer_idx] = False
            margin_loss = torch.clamp(0.25 - (correct - logits[mask].max()), min=0.0)
            loss = hard_loss + lambda_soft * soft_loss + mu_margin * margin_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    predictions: list[dict[str, Any]] = []
    for item in eval_items:
        with timed() as state:
            features = torch.tensor(_item_features(embedder, item), dtype=torch.float32)
            with torch.no_grad():
                logits = scorer(features).squeeze(-1)
                probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        idx = int(np.argmax(probs))
        predictions.append(
            prediction_row(
                item,
                answer_label(idx),
                float(probs[idx]),
                latency_ms=state["latency_ms"],
                extra=memory_snapshot(),
            )
        )

    torch.save({"state_dict": scorer.state_dict(), "embedding_model": embedding_model}, output / "kd_scorer.pt")
    metrics = summarize_predictions(predictions)
    write_jsonl(output / "prediction_log.jsonl", predictions)
    (output / "metric_log.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train E1 embedding scorer with hard+soft KD loss.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-model", default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--min-status", default="aug_judge_pass")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda-soft", type=float, default=0.5)
    parser.add_argument("--mu-margin", type=float, default=0.2)
    parser.add_argument("--test-lock-dir")
    parser.add_argument("--allow-missing-teacher-scores", action="store_true")
    parser.add_argument("--allow-repeat-test", action="store_true")
    args = parser.parse_args()
    metrics = train_kd_scorer(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        train_split=args.train_split,
        eval_split=args.eval_split,
        min_status=args.min_status,
        epochs=args.epochs,
        lambda_soft=args.lambda_soft,
        mu_margin=args.mu_margin,
        final_test_once=not args.allow_repeat_test,
        test_lock_dir=args.test_lock_dir,
        require_teacher_scores=not args.allow_missing_teacher_scores,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
