from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.progress import ProgressLogger, append_jsonl, load_jsonl


def _load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["item_id"])] = row
    return rows


def _build_hybrid_rows(
    primary: dict[str, dict[str, Any]],
    fallback: dict[str, dict[str, Any]],
    cross_encoder: dict[str, dict[str, Any]],
    low_threshold: float,
    high_threshold: float,
    stress_fallback: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {"primary": 0, "cross_encoder": 0, "fallback": 0}
    for item_id, row in primary.items():
        confidence = float(row.get("confidence", 0.0))
        stress_tags = set(str(tag) for tag in row.get("stress_tags") or [])
        force_fallback = stress_fallback and bool({"short_answer", "high_similarity", "polysemy"} & stress_tags)
        use_cross_encoder = False
        use_fallback = False
        if not force_fallback and confidence >= high_threshold:
            chosen = dict(row)
            counts["primary"] += 1
        elif not force_fallback and confidence >= low_threshold and item_id in cross_encoder:
            chosen = dict(cross_encoder[item_id])
            use_cross_encoder = True
            counts["cross_encoder"] += 1
        elif item_id in fallback:
            chosen = dict(fallback[item_id])
            use_fallback = True
            counts["fallback"] += 1
        elif item_id in cross_encoder:
            chosen = dict(cross_encoder[item_id])
            use_cross_encoder = True
            counts["cross_encoder"] += 1
        else:
            chosen = dict(row)
            counts["primary"] += 1
        chosen["used_cross_encoder"] = use_cross_encoder
        chosen["used_fallback"] = use_fallback
        chosen["primary_confidence"] = row.get("confidence")
        chosen["hybrid_reason"] = (
            "stress_fallback"
            if force_fallback and use_fallback
            else "high_confidence"
            if not use_cross_encoder and not use_fallback
            else "mid_confidence_cross_encoder"
            if use_cross_encoder
            else "low_confidence_fallback"
        )
        rows.append(chosen)
    return rows, counts


def _hybrid_metrics(
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    low_threshold: float,
    high_threshold: float,
    primary_cost_per_1000: float,
    cross_encoder_cost_per_1000: float,
    fallback_cost_per_1000: float,
) -> dict[str, Any]:
    metrics = summarize_predictions(rows)
    total = len(rows)
    cross_encoder_rate = counts["cross_encoder"] / total if total else 0.0
    fallback_rate = counts["fallback"] / total if total else 0.0
    metrics["primary_accept_rate"] = counts["primary"] / total if total else 0.0
    metrics["cross_encoder_rate"] = cross_encoder_rate
    metrics["fallback_rate"] = fallback_rate
    metrics["cost_per_1000_questions"] = (
        primary_cost_per_1000
        + cross_encoder_rate * cross_encoder_cost_per_1000
        + fallback_rate * fallback_cost_per_1000
    )
    metrics["low_confidence_threshold"] = low_threshold
    metrics["high_confidence_threshold"] = high_threshold
    return metrics


def run_hybrid_policy(
    primary_predictions: str | Path,
    fallback_predictions: str | Path,
    output_dir: str | Path,
    confidence_threshold: float = 0.7,
    cross_encoder_predictions: str | Path | None = None,
    low_confidence_threshold: float | None = None,
    high_confidence_threshold: float | None = None,
    primary_cost_per_1000: float = 0.0,
    cross_encoder_cost_per_1000: float = 0.0,
    fallback_cost_per_1000: float = 0.0,
    stress_fallback: bool = True,
    resume: bool = True,
    progress_interval_items: int = 1,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    primary = _load_predictions(primary_predictions)
    fallback = _load_predictions(fallback_predictions)
    cross_encoder = _load_predictions(cross_encoder_predictions) if cross_encoder_predictions else {}
    progress = ProgressLogger(
        output,
        resume=resume,
        progress_interval_items=progress_interval_items,
        stage="H1:hybrid",
        total_count=len(primary),
    )
    completed_metrics = progress.completed_metrics_if_available(primary.keys())
    if completed_metrics is not None:
        return completed_metrics
    low_threshold = confidence_threshold if low_confidence_threshold is None else low_confidence_threshold
    high_threshold = confidence_threshold if high_confidence_threshold is None else high_confidence_threshold
    rows, counts = _build_hybrid_rows(primary, fallback, cross_encoder, low_threshold, high_threshold, stress_fallback)
    metrics = _hybrid_metrics(
        rows,
        counts,
        low_threshold,
        high_threshold,
        primary_cost_per_1000,
        cross_encoder_cost_per_1000,
        fallback_cost_per_1000,
    )

    for row in rows:
        progress.append_prediction(row)
    progress.finalize_predictions(metrics, rows)
    return metrics


def tune_hybrid_policy(
    primary_predictions: str | Path,
    fallback_predictions: str | Path,
    output_dir: str | Path,
    cross_encoder_predictions: str | Path | None = None,
    threshold_grid: list[float] | None = None,
    primary_cost_per_1000: float = 0.0,
    cross_encoder_cost_per_1000: float = 0.0,
    fallback_cost_per_1000: float = 0.0,
    stress_fallback: bool = True,
    resume: bool = True,
    progress_interval_items: int = 1,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    primary = _load_predictions(primary_predictions)
    fallback = _load_predictions(fallback_predictions)
    cross_encoder = _load_predictions(cross_encoder_predictions) if cross_encoder_predictions else {}
    progress = ProgressLogger(
        output,
        resume=resume,
        progress_interval_items=progress_interval_items,
        stage="H1:hybrid_tuning",
        total_count=len(primary),
    )
    completed_metrics = progress.completed_metrics_if_available(primary.keys())
    if completed_metrics is not None:
        return completed_metrics
    grid = threshold_grid or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    trial_partial = output / "hybrid_policy_tuning.partial.jsonl"
    if not resume and trial_partial.exists():
        trial_partial.unlink()
    trials: list[dict[str, Any]] = load_jsonl(trial_partial) if resume else []
    completed_pairs = {
        (float(row["low_confidence_threshold"]), float(row["high_confidence_threshold"]))
        for row in trials
        if "low_confidence_threshold" in row and "high_confidence_threshold" in row
    }
    for low in grid:
        for high in grid:
            if low > high:
                continue
            if (float(low), float(high)) in completed_pairs:
                continue
            rows, counts = _build_hybrid_rows(primary, fallback, cross_encoder, low, high, stress_fallback)
            metrics = _hybrid_metrics(
                rows,
                counts,
                low,
                high,
                primary_cost_per_1000,
                cross_encoder_cost_per_1000,
                fallback_cost_per_1000,
            )
            trial = {
                "low_confidence_threshold": low,
                "high_confidence_threshold": high,
                "accuracy": metrics["accuracy"],
                "fallback_rate": metrics["fallback_rate"],
                "cross_encoder_rate": metrics["cross_encoder_rate"],
                "cost_per_1000_questions": metrics["cost_per_1000_questions"],
                "parse_error_rate": metrics["parse_error_rate"],
            }
            trials.append(trial)
            append_jsonl(trial_partial, trial)
            progress.record_metrics(trial, stage="H1:hybrid_tuning", event="trial")
    if not trials:
        raise ValueError("threshold_grid produced no valid low/high threshold pairs")

    best_trial = max(
        trials,
        key=lambda trial: (
            float(trial["accuracy"]),
            -float(trial["cost_per_1000_questions"]),
            -float(trial["fallback_rate"]),
            -float(trial["cross_encoder_rate"]),
        ),
    )
    best_rows, best_counts = _build_hybrid_rows(
        primary,
        fallback,
        cross_encoder,
        float(best_trial["low_confidence_threshold"]),
        float(best_trial["high_confidence_threshold"]),
        stress_fallback,
    )
    best = _hybrid_metrics(
        best_rows,
        best_counts,
        float(best_trial["low_confidence_threshold"]),
        float(best_trial["high_confidence_threshold"]),
        primary_cost_per_1000,
        cross_encoder_cost_per_1000,
        fallback_cost_per_1000,
    )
    best.pop("_selection_key", None)
    for row in best_rows:
        progress.append_prediction(row)
    progress.finalize_predictions(best, best_rows)
    tuning = {"threshold_grid": grid, "trials": trials, "selected_policy": best}
    (output / "hybrid_policy_tuning.json").write_text(json.dumps(tuning, ensure_ascii=False, indent=2), encoding="utf-8")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1 confidence based hybrid fallback policy.")
    parser.add_argument("--primary-predictions", required=True)
    parser.add_argument("--fallback-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--cross-encoder-predictions")
    parser.add_argument("--low-confidence-threshold", type=float)
    parser.add_argument("--high-confidence-threshold", type=float)
    parser.add_argument("--primary-cost-per-1000", type=float, default=0.0)
    parser.add_argument("--cross-encoder-cost-per-1000", type=float, default=0.0)
    parser.add_argument("--fallback-cost-per-1000", type=float, default=0.0)
    parser.add_argument("--disable-stress-fallback", action="store_true")
    parser.add_argument("--tune-policy", action="store_true")
    parser.add_argument("--threshold-grid", type=float, nargs="*")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress-interval-items", type=int, default=1)
    args = parser.parse_args()
    if args.tune_policy:
        metrics = tune_hybrid_policy(
            args.primary_predictions,
            args.fallback_predictions,
            args.output_dir,
            cross_encoder_predictions=args.cross_encoder_predictions,
            threshold_grid=args.threshold_grid,
            primary_cost_per_1000=args.primary_cost_per_1000,
            cross_encoder_cost_per_1000=args.cross_encoder_cost_per_1000,
            fallback_cost_per_1000=args.fallback_cost_per_1000,
            stress_fallback=not args.disable_stress_fallback,
            resume=not args.no_resume,
            progress_interval_items=args.progress_interval_items,
        )
    else:
        metrics = run_hybrid_policy(
            args.primary_predictions,
            args.fallback_predictions,
            args.output_dir,
            confidence_threshold=args.confidence_threshold,
            cross_encoder_predictions=args.cross_encoder_predictions,
            low_confidence_threshold=args.low_confidence_threshold,
            high_confidence_threshold=args.high_confidence_threshold,
            primary_cost_per_1000=args.primary_cost_per_1000,
            cross_encoder_cost_per_1000=args.cross_encoder_cost_per_1000,
            fallback_cost_per_1000=args.fallback_cost_per_1000,
            stress_fallback=not args.disable_stress_fallback,
            resume=not args.no_resume,
            progress_interval_items=args.progress_interval_items,
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
