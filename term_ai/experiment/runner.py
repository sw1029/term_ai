from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone
from typing import Any

from term_ai.contracts import iter_jsonl, write_jsonl
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.model_matrix import get_model_spec, model_matrix_as_dicts


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_run_dir(base_dir: Path, experiment_id: str, run_name: str | None = None) -> Path:
    suffix = run_name or utc_stamp()
    run_dir = base_dir / f"{experiment_id}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "backups").mkdir()
    return run_dir


def write_run_manifest(run_dir: Path, experiment_id: str, config: dict[str, Any]) -> dict[str, Any]:
    spec = get_model_spec(experiment_id)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "model_spec": spec.to_dict(),
        "config_snapshot": config,
        "contracts": {
            "split_policy": "word-level train/dev/test split",
            "sft_schema": "messages-only JSONL",
            "test_policy": "dev tunes thresholds; test is final comparison only",
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def write_resume_state(run_dir: Path, state: dict[str, Any]) -> None:
    payload = dict(state)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "resume_state.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def backup_checkpoint(checkpoint_path: Path, run_dir: Path) -> Path:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    backup_dir = run_dir / "backups"
    backup_dir.mkdir(exist_ok=True)
    target = backup_dir / f"{checkpoint_path.stem}_{utc_stamp()}{checkpoint_path.suffix}"
    shutil.copy2(checkpoint_path, target)
    return target


def evaluate_prediction_file(predictions_path: Path, run_dir: Path) -> dict[str, Any]:
    predictions = list(iter_jsonl(predictions_path))
    required = {"label", "prediction"}
    for idx, row in enumerate(predictions, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"prediction row {idx} missing fields: {sorted(missing)}")

    metrics = summarize_predictions(predictions)
    write_jsonl(run_dir / "prediction_log.jsonl", predictions)
    (run_dir / "metric_log.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_resume_state(run_dir, {"stage": "evaluated", "prediction_count": len(predictions)})
    return metrics


def init_matrix(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model_matrix_as_dicts(), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment run artifact tooling.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix = subparsers.add_parser("write-matrix")
    matrix.add_argument("--output", default="experiments/model_matrix.json")

    init_run = subparsers.add_parser("init-run")
    init_run.add_argument("--runs-dir", default="runs")
    init_run.add_argument("--experiment-id", required=True)
    init_run.add_argument("--run-name")
    init_run.add_argument("--config", default="{}")

    evaluate = subparsers.add_parser("evaluate-predictions")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--predictions", required=True)

    args = parser.parse_args()
    if args.command == "write-matrix":
        init_matrix(Path(args.output))
        print(json.dumps({"output": args.output}, ensure_ascii=False))
    elif args.command == "init-run":
        run_dir = create_run_dir(Path(args.runs_dir), args.experiment_id, args.run_name)
        config = json.loads(args.config)
        manifest = write_run_manifest(run_dir, args.experiment_id, config)
        write_resume_state(run_dir, {"stage": "initialized"})
        print(json.dumps({"run_dir": str(run_dir), "manifest": manifest}, ensure_ascii=False, indent=2))
    elif args.command == "evaluate-predictions":
        metrics = evaluate_prediction_file(Path(args.predictions), Path(args.run_dir))
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
