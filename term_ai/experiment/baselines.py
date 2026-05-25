from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from term_ai.contracts import RAW_GT_STATUS, answer_label, write_jsonl
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import MCQItem, load_mcq_items, prediction_row
from term_ai.experiment.ops import memory_snapshot, timed
from term_ai.experiment.test_lock import enforce_final_test_once


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


def _b0_predictions_with_model(
    items: list[MCQItem],
    model: Any,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        with timed() as state:
            features = _item_features(model, item)
        scores = features[:, 0]
        idx = int(np.argmax(scores))
        confidence = float((scores[idx] + 1) / 2)
        score = float(scores[idx])
        prediction = answer_label(idx) if threshold is None or score >= threshold else "ABSTAIN"
        rows.append(
            prediction_row(
                item,
                prediction,
                confidence,
                latency_ms=state["latency_ms"],
                extra={
                    "score": score,
                    "threshold": threshold,
                    "abstained": prediction == "ABSTAIN",
                    **memory_snapshot(),
                },
            )
        )
    return rows


def tune_mxbai_threshold(items: list[MCQItem], model: Any) -> dict[str, Any]:
    if not items:
        raise ValueError("no dev items available for threshold tuning")
    scored = []
    for item in items:
        features = _item_features(model, item)
        scores = features[:, 0]
        idx = int(np.argmax(scores))
        scored.append((float(scores[idx]), answer_label(idx), item.label))

    candidates = sorted({score for score, _, _ in scored} | {-1.0, 1.0})
    best = {"threshold": candidates[0], "accuracy": -1.0, "coverage": 0.0}
    for threshold in candidates:
        total = len(scored)
        correct = 0
        covered = 0
        for score, prediction, label in scored:
            if score >= threshold:
                covered += 1
                correct += int(prediction == label)
        # Abstentions are treated as wrong for the automatic scoring metric.
        accuracy = correct / total if total else 0.0
        coverage = covered / total if total else 0.0
        if (accuracy, coverage, -threshold) > (best["accuracy"], best["coverage"], -best["threshold"]):
            best = {"threshold": threshold, "accuracy": accuracy, "coverage": coverage}
    return best


def b0_mxbai_threshold(
    items: list[MCQItem],
    model_name: str,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    model = _load_embedder(model_name)
    return _b0_predictions_with_model(items, model, threshold=threshold)


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
        with timed() as state:
            features = _item_features(embedder, item)
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(features)[:, 1]
        else:
            probs = clf.decision_function(features)
        idx = int(np.argmax(probs))
        rows.append(
            prediction_row(
                item,
                answer_label(idx),
                float(probs[idx]),
                latency_ms=state["latency_ms"],
                extra=memory_snapshot(),
            )
        )
    return rows


def run_baseline(
    metadata_path: str | Path,
    output_dir: str | Path,
    method: str,
    eval_split: str = "dev",
    train_split: str = "train",
    min_status: str = RAW_GT_STATUS,
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
    train_metadata_path: str | Path | None = None,
    threshold_metadata_path: str | Path | None = None,
    threshold_split: str = "dev",
    threshold: float | None = None,
    final_test_once: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enforce_final_test_once(output, f"B0_{method}", eval_split, enabled=final_test_once)
    items = load_mcq_items(metadata_path, min_status=min_status)
    eval_items = [item for item in items if item.split == eval_split]
    train_source = train_metadata_path or metadata_path
    train_items = [item for item in load_mcq_items(train_source, min_status=min_status) if item.split == train_split]

    if method == "b0":
        model = _load_embedder(embedding_model)
        tuning: dict[str, Any] | None = None
        if threshold is None:
            threshold_source = threshold_metadata_path or metadata_path
            threshold_items = [
                item for item in load_mcq_items(threshold_source, min_status=min_status) if item.split == threshold_split
            ]
            tuning = tune_mxbai_threshold(threshold_items, model)
            threshold = float(tuning["threshold"])
        predictions = _b0_predictions_with_model(eval_items, model, threshold=threshold)
    elif method in {"logistic", "mlp"}:
        if not train_items:
            raise ValueError(f"no train items available for {method}")
        model_path = output / f"{method}_scorer.pkl"
        embedder, clf = train_option_scorer(train_items, embedding_model, method, model_path)
        predictions = predict_option_scorer(eval_items, embedder, clf)
    else:
        raise ValueError("method must be b0, logistic, or mlp")

    metrics = summarize_predictions(predictions)
    if method == "b0":
        metrics["threshold"] = threshold
        if tuning is not None:
            metrics["threshold_tuning"] = tuning
            (output / "threshold.json").write_text(json.dumps(tuning, ensure_ascii=False, indent=2), encoding="utf-8")
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
    parser.add_argument("--min-status", default=RAW_GT_STATUS)
    parser.add_argument("--embedding-model", default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--train-metadata")
    parser.add_argument("--threshold-metadata")
    parser.add_argument("--threshold-split", default="dev")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--allow-repeat-test", action="store_true")
    args = parser.parse_args()
    metrics = run_baseline(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        method=args.method,
        eval_split=args.eval_split,
        train_split=args.train_split,
        min_status=args.min_status,
        embedding_model=args.embedding_model,
        train_metadata_path=args.train_metadata,
        threshold_metadata_path=args.threshold_metadata,
        threshold_split=args.threshold_split,
        threshold=args.threshold,
        final_test_once=not args.allow_repeat_test,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
