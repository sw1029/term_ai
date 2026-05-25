from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from term_ai.contracts import answer_label, write_jsonl
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import MCQItem, load_mcq_items, prediction_row


def _load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install baseline dependencies first: pip install -e .[baseline]") from exc
    return SentenceTransformer(model_name)


def _cosine_matrix(query_vectors: np.ndarray, option_vectors: np.ndarray) -> np.ndarray:
    return np.sum(query_vectors * option_vectors, axis=1)


def _item_features(model: Any, item: MCQItem) -> np.ndarray:
    query = item.query_text()
    query_vec = np.asarray(model.encode([query], normalize_embeddings=True))[0]
    option_vecs = np.asarray(model.encode(item.options, normalize_embeddings=True))
    cosine_scores = _cosine_matrix(np.repeat(query_vec[None, :], len(item.options), axis=0), option_vecs)
    option_positions = np.arange(len(item.options), dtype=float) / max(1, len(item.options) - 1)
    return np.column_stack([cosine_scores, option_positions])


def b0_mxbai_threshold(items: list[MCQItem], model_name: str) -> list[dict[str, Any]]:
    model = _load_embedder(model_name)
    rows: list[dict[str, Any]] = []
    for item in items:
        features = _item_features(model, item)
        scores = features[:, 0]
        idx = int(np.argmax(scores))
        confidence = float((scores[idx] + 1) / 2)
        rows.append(prediction_row(item, answer_label(idx), confidence, extra={"score": float(scores[idx])}))
    return rows


def train_option_scorer(
    items: list[MCQItem],
    model_name: str,
    scorer_type: str,
    model_output: str | Path,
) -> tuple[Any, Any]:
    if scorer_type == "logistic":
        from sklearn.linear_model import LogisticRegression

        clf: Any = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif scorer_type == "mlp":
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    else:
        raise ValueError("scorer_type must be logistic or mlp")

    embedder = _load_embedder(model_name)
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for item in items:
        features = _item_features(embedder, item)
        for idx, feature in enumerate(features):
            x_rows.append(feature)
            y_rows.append(int(idx == item.answer_idx))
    clf.fit(np.asarray(x_rows), np.asarray(y_rows))
    with open(model_output, "wb") as handle:
        pickle.dump({"embedder_model_name": model_name, "scorer_type": scorer_type, "classifier": clf}, handle)
    return embedder, clf


def predict_option_scorer(items: list[MCQItem], embedder: Any, clf: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        features = _item_features(embedder, item)
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(features)[:, 1]
        else:
            probs = clf.decision_function(features)
        idx = int(np.argmax(probs))
        rows.append(prediction_row(item, answer_label(idx), float(probs[idx])))
    return rows


def run_baseline(
    metadata_path: str | Path,
    output_dir: str | Path,
    method: str,
    eval_split: str = "dev",
    train_split: str = "train",
    min_status: str = "aug_auto_pass",
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    items = load_mcq_items(metadata_path, min_status=min_status)
    eval_items = [item for item in items if item.split == eval_split]
    train_items = [item for item in items if item.split == train_split]

    if method == "b0":
        predictions = b0_mxbai_threshold(eval_items, embedding_model)
    elif method in {"logistic", "mlp"}:
        if not train_items:
            raise ValueError(f"no train items available for {method}")
        model_path = output / f"{method}_scorer.pkl"
        embedder, clf = train_option_scorer(train_items, embedding_model, method, model_path)
        predictions = predict_option_scorer(eval_items, embedder, clf)
    else:
        raise ValueError("method must be b0, logistic, or mlp")

    metrics = summarize_predictions(predictions)
    write_jsonl(output / "prediction_log.jsonl", predictions)
    (output / "metric_log.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B0/B1/B2 embedding baselines.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=["b0", "logistic", "mlp"], required=True)
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--min-status", default="aug_auto_pass")
    parser.add_argument("--embedding-model", default="mixedbread-ai/mxbai-embed-large-v1")
    args = parser.parse_args()
    metrics = run_baseline(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        method=args.method,
        eval_split=args.eval_split,
        train_split=args.train_split,
        min_status=args.min_status,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
