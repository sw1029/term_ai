from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
from statistics import mean
from typing import Iterable


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lengths differ")
    if not y_true:
        return 0.0
    return sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred) / len(y_true)


def classwise_f1(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    labels = sorted(set(y_true) | set(y_pred))
    result: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return result


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    scores = classwise_f1(y_true, y_pred)
    return mean(scores.values()) if scores else 0.0


def brier_score(correct: list[int], confidence: list[float]) -> float:
    if len(correct) != len(confidence):
        raise ValueError("correct and confidence lengths differ")
    if not correct:
        return 0.0
    return mean((float(prob) - int(label)) ** 2 for label, prob in zip(correct, confidence))


def expected_calibration_error(correct: list[int], confidence: list[float], bins: int = 10) -> float:
    if len(correct) != len(confidence):
        raise ValueError("correct and confidence lengths differ")
    if not correct:
        return 0.0

    bucket_totals = [0 for _ in range(bins)]
    bucket_conf = [0.0 for _ in range(bins)]
    bucket_correct = [0.0 for _ in range(bins)]
    for is_correct, conf in zip(correct, confidence):
        clamped = min(1.0, max(0.0, float(conf)))
        idx = min(bins - 1, int(clamped * bins))
        bucket_totals[idx] += 1
        bucket_conf[idx] += clamped
        bucket_correct[idx] += int(is_correct)

    ece = 0.0
    total = len(correct)
    for idx in range(bins):
        if bucket_totals[idx] == 0:
            continue
        avg_conf = bucket_conf[idx] / bucket_totals[idx]
        avg_acc = bucket_correct[idx] / bucket_totals[idx]
        ece += bucket_totals[idx] / total * abs(avg_acc - avg_conf)
    return ece


def bootstrap_accuracy_ci(
    y_true: list[str],
    y_pred: list[str],
    samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lengths differ")
    if not y_true:
        return (0.0, 0.0)

    rng = random.Random(seed)
    n = len(y_true)
    values: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_true = [y_true[idx] for idx in indices]
        sample_pred = [y_pred[idx] for idx in indices]
        values.append(accuracy(sample_true, sample_pred))
    values.sort()
    lower = values[int((alpha / 2) * samples)]
    upper = values[min(samples - 1, int((1 - alpha / 2) * samples))]
    return (lower, upper)


def latency_summary(latencies_ms: Iterable[float]) -> dict[str, float]:
    values = sorted(float(value) for value in latencies_ms)
    if not values:
        return {"latency_p50": 0.0, "latency_p95": 0.0}
    p50_idx = min(len(values) - 1, math.floor(0.50 * (len(values) - 1)))
    p95_idx = min(len(values) - 1, math.ceil(0.95 * (len(values) - 1)))
    return {"latency_p50": values[p50_idx], "latency_p95": values[p95_idx]}


def summarize_predictions(predictions: list[dict]) -> dict:
    y_true = [str(row["label"]) for row in predictions]
    y_pred = [str(row["prediction"]) for row in predictions]
    correct = [int(truth == pred) for truth, pred in zip(y_true, y_pred)]
    confidence = [float(row.get("confidence", 1.0)) for row in predictions]
    latencies = [float(row["latency_ms"]) for row in predictions if "latency_ms" in row]
    token_speeds = [float(row["tokens_per_sec"]) for row in predictions if "tokens_per_sec" in row]
    peak_vram = [float(row["peak_vram_mb"]) for row in predictions if "peak_vram_mb" in row]
    parse_errors = sum(1 for row in predictions if row.get("parse_error"))

    ci_low, ci_high = bootstrap_accuracy_ci(y_true, y_pred, samples=500) if predictions else (0.0, 0.0)
    task_counts = Counter(str(row.get("task_type", "unknown")) for row in predictions)
    task_accuracy: dict[str, float] = {}
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_task[str(row.get("task_type", "unknown"))].append(row)
    for task_type, rows in by_task.items():
        task_accuracy[task_type] = accuracy(
            [str(row["label"]) for row in rows],
            [str(row["prediction"]) for row in rows],
        )

    summary = {
        "n": len(predictions),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "classwise_f1": classwise_f1(y_true, y_pred),
        "ece": expected_calibration_error(correct, confidence),
        "brier_score": brier_score(correct, confidence),
        "bootstrap_accuracy_95ci": [ci_low, ci_high],
        "task_counts": dict(task_counts),
        "task_accuracy": task_accuracy,
        "parse_error_rate": parse_errors / len(predictions) if predictions else 0.0,
    }
    if token_speeds:
        summary["tokens_per_sec"] = mean(token_speeds)
    if peak_vram:
        summary["peak_VRAM_or_RAM"] = max(peak_vram)
    summary.update(latency_summary(latencies))
    return summary
