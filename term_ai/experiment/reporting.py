from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_metric_logs(runs_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in Path(runs_dir).rglob("metric_log.json"):
        metric = json.loads(path.read_text(encoding="utf-8"))
        metric["run_dir"] = str(path.parent)
        rows.append(metric)
    return rows


def _load_prediction_errors(runs_dir: str | Path, limit: int = 100) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for path in Path(runs_dir).rglob("prediction_log.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("label") != row.get("prediction"):
                    row["run_dir"] = str(path.parent)
                    errors.append(row)
                    if len(errors) >= limit:
                        return errors
    return errors


def _load_explanation_summaries(runs_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in Path(runs_dir).rglob("*explanation_judgment_summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["run_dir"] = str(path.parent)
        summary["summary_path"] = str(path)
        rows.append(summary)
    return rows


def write_final_report_inputs(runs_dir: str | Path, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = _load_metric_logs(runs_dir)
    errors = _load_prediction_errors(runs_dir)
    explanation_summaries = _load_explanation_summaries(runs_dir)
    metrics_path = output / "final_experiment_report_input.json"
    errors_path = output / "error_analysis_input.json"
    explanations_path = output / "explanation_judge_report_input.json"
    deploy_path = output / "deployment_recommendation_input.json"
    report_md_path = output / "final_experiment_report.md"
    deploy_md_path = output / "deployment_recommendation.md"

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    errors_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    explanations_path.write_text(json.dumps(explanation_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    recommendation = {
        "rules": [
            "Prefer embedding scorer if quality is close to API recheck and latency/cost are lower.",
            "Use fallback only for low-confidence or stress-tagged items.",
            "Use quantized local LM only if raw test and stress subset drops are bounded.",
        ],
        "available_metric_logs": len(metrics),
        "error_cases": len(errors),
    }
    deploy_path.write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md_path.write_text(_render_final_report(metrics, errors, explanation_summaries), encoding="utf-8")
    deploy_md_path.write_text(_render_deployment_recommendation(metrics, recommendation), encoding="utf-8")
    return {
        "final_report_input": str(metrics_path),
        "error_analysis_input": str(errors_path),
        "explanation_judge_report_input": str(explanations_path),
        "deployment_recommendation_input": str(deploy_path),
        "final_report": str(report_md_path),
        "deployment_recommendation": str(deploy_md_path),
    }


def _best_metric(metrics: list[dict[str, Any]], key: str, reverse: bool = True) -> dict[str, Any] | None:
    candidates = [row for row in metrics if isinstance(row.get(key), (int, float))]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: float(row[key]), reverse=reverse)[0]


def _error_breakdown(errors: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in errors:
        tags = row.get("stress_tags") or ["untagged"]
        for tag in tags:
            key = str(tag)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _ops_missing(metrics: list[dict[str, Any]]) -> list[str]:
    required = [
        "latency_ms",
        "batch_size_1_latency_ms",
        "tokens_per_sec",
        "ram_mb",
        "cold_start_ms",
    ]
    missing: list[str] = []
    for key in required:
        if not any((row.get("ops_metric_coverage") or {}).get(key, 0) for row in metrics):
            missing.append(key)
    has_cost = any(
        (row.get("ops_metric_coverage") or {}).get("estimated_cost_usd", 0)
        or (row.get("ops_metric_coverage") or {}).get("local_cost_per_hour_usd", 0)
        for row in metrics
    )
    if not has_cost:
        missing.append("estimated_or_local_cost")
    return missing


def _render_final_report(
    metrics: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    explanation_summaries: list[dict[str, Any]],
) -> str:
    best_accuracy = _best_metric(metrics, "accuracy")
    best_latency = _best_metric(metrics, "latency_p95", reverse=False)
    best_cost = _best_metric(metrics, "cost_per_1000_questions", reverse=False)
    best_ece = _best_metric(metrics, "ece", reverse=False)
    error_breakdown = _error_breakdown(errors)
    missing_ops = _ops_missing(metrics)
    lines = [
        "# Final Experiment Report",
        "",
        "## Summary",
        f"- Metric logs collected: {len(metrics)}",
        f"- Error cases sampled: {len(errors)}",
    ]
    if best_accuracy:
        lines.append(
            f"- Best accuracy: {best_accuracy.get('accuracy'):.4f} ({best_accuracy.get('run_dir', 'unknown run')})"
        )
    if best_latency:
        lines.append(
            f"- Lowest p95 latency: {best_latency.get('latency_p95'):.2f} ms ({best_latency.get('run_dir', 'unknown run')})"
        )
    if best_cost:
        lines.append(
            f"- Lowest cost/1000 questions: {best_cost.get('cost_per_1000_questions'):.6f} ({best_cost.get('run_dir', 'unknown run')})"
        )
    if best_ece:
        lines.append(f"- Best calibration ECE: {best_ece.get('ece'):.4f} ({best_ece.get('run_dir', 'unknown run')})")
    lines.extend(["", "## Error Analysis"])
    if error_breakdown:
        for tag, count in sorted(error_breakdown.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {tag}: {count}")
    else:
        lines.append("- No prediction errors were collected.")
    lines.extend(["", "## Explanation Judge"])
    if explanation_summaries:
        for summary in explanation_summaries:
            lines.append(
                "- "
                f"{summary.get('summary_path', summary.get('run_dir', 'unknown'))}: "
                f"final_score_avg={float(summary.get('final_score_avg', 0.0)):.4f}, "
                f"reasoning_faithfulness_avg={float(summary.get('reasoning_faithfulness_avg', 0.0)):.4f}, "
                f"hallucination_fail_rate={float(summary.get('hallucination_fail_rate', 0.0)):.4f}"
            )
    else:
        lines.append("- No explanation judge summaries were collected.")
    lines.extend(["", "## Operational Coverage"])
    if missing_ops:
        lines.append(f"- Missing or empty operational metrics: {', '.join(missing_ops)}")
    else:
        lines.append("- Required operational metrics were present in at least one run.")
    lines.extend(
        [
            "",
            "## Caveats",
            "- Interpret generated cloze results separately from raw GT results.",
            "- Treat this report as invalid for deployment if final test locks or raw/test_cloze split artifacts are missing.",
            "- Treat aug_judge_pass as strict judge validated, not human approved.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_deployment_recommendation(metrics: list[dict[str, Any]], recommendation: dict[str, Any]) -> str:
    best_accuracy = _best_metric(metrics, "accuracy")
    best_cost = _best_metric(metrics, "cost_per_1000_questions", reverse=False)
    lines = [
        "# Deployment Recommendation",
        "",
        "## Decision Rules",
        *[f"- {rule}" for rule in recommendation["rules"]],
        "",
        "## Current Signal",
    ]
    if not metrics:
        lines.append("- No metric logs are available yet; do not choose a deployment model.")
    else:
        if best_accuracy:
            lines.append(f"- Accuracy leader: {best_accuracy.get('run_dir', 'unknown run')}")
        if best_cost:
            lines.append(f"- Cost leader: {best_cost.get('run_dir', 'unknown run')}")
        missing_ops = _ops_missing(metrics)
        if missing_ops:
            lines.append(f"- Do not finalize deployment until these metrics are populated: {', '.join(missing_ops)}")
        lines.append("- Check stress subset and calibration before using confidence-based fallback in production.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final report/error/deployment input artifacts.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    outputs = write_final_report_inputs(args.runs_dir, args.output_dir)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
