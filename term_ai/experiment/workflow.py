from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from term_ai.augmentation.dataset_builder import (
    build_raw_mcq_metadata_from_anchors,
    build_raw_sft_from_anchors,
    build_strict_eval_sets,
)
from term_ai.augmentation.pipeline import auto_filter_metadata, prepare_artifacts
from term_ai.augmentation.orchestrator import generate_train_batch
from term_ai.experiment.reporting import write_final_report_inputs
from term_ai.experiment.runner import init_matrix


def _phase_record(phase: int, name: str, status: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"phase": phase, "name": name, "status": status, "details": details or {}}


def run_master_workflow(config: dict[str, Any], execute: bool = False) -> dict[str, Any]:
    """Coordinate Phase 0~9 without silently running expensive model jobs.

    The workflow always materializes safe contract artifacts. LLM calls, model
    training, and full evaluations run only when execute=True and the relevant
    config section is explicitly enabled.
    """

    runs_dir = Path(config.get("runs_dir", "runs"))
    run_dir = runs_dir / f"master_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    phases: list[dict[str, Any]] = []

    matrix_path = Path(config.get("model_matrix", "experiments/model_matrix.json"))
    init_matrix(matrix_path)
    phases.append(_phase_record(0, "experiment freeze", "completed", {"model_matrix": str(matrix_path)}))

    data_cfg = config.get("data", {})
    input_path = Path(data_cfg.get("input_path", "pharaprased_voca.jsonl"))
    data_dir = Path(data_cfg.get("output_dir", "data"))
    seed = int(data_cfg.get("seed", 42))
    manifest = prepare_artifacts(input_path, data_dir, seed=seed)
    anchors_path = Path(manifest["anchor_path"])
    raw_sft_counts = build_raw_sft_from_anchors(anchors_path, data_dir / "sft", seed=seed)
    raw_mcq_counts = build_raw_mcq_metadata_from_anchors(anchors_path, data_dir / "metadata", seed=seed)
    phases.append(
        _phase_record(
            1,
            "raw anchor, split, raw SFT/MCQ",
            "completed",
            {"prepare": manifest, "raw_sft": raw_sft_counts, "raw_mcq": raw_mcq_counts},
        )
    )

    aug_cfg = config.get("augmentation", {})
    if execute and aug_cfg.get("enabled"):
        total = int(aug_cfg["total"])
        candidate_path = Path(aug_cfg.get("candidate_path", data_dir / "aug" / "train_aug_candidate_v1.jsonl"))
        aug_manifest = generate_train_batch(
            anchors_path=anchors_path,
            output_path=candidate_path,
            total=total,
            model=str(aug_cfg.get("model", "gpt-5.4-mini")),
            env_path=aug_cfg.get("env_path", ".env"),
            requests_per_second=float(aug_cfg.get("requests_per_second", 1.0)),
        )
        auto_path = Path(aug_cfg.get("auto_filter_path", data_dir / "aug" / "train_aug_auto_pass_v1.jsonl"))
        auto_counts = auto_filter_metadata(candidate_path, auto_path)
        phases.append(
            _phase_record(2, "augmentation generation and auto filter", "completed", {"manifest": aug_manifest, "auto_filter": auto_counts})
        )
    else:
        phases.append(_phase_record(2, "augmentation generation and validation", "planned", {"execute_required": True}))

    approved_metadata = config.get("approved_metadata")
    raw_metadata = data_dir / "metadata" / "raw_mcq_v1.jsonl"
    if approved_metadata and Path(approved_metadata).exists():
        eval_counts = build_strict_eval_sets(raw_metadata, approved_metadata, data_dir / "eval")
        phases.append(_phase_record(3, "strict raw/cloze eval views", "completed", eval_counts))
    else:
        phases.append(_phase_record(3, "baselines and strict eval views", "planned", {"needs_approved_metadata": True}))

    for phase, name in [
        (4, "small LM zero-shot"),
        (5, "LoRA SFT"),
        (6, "LoRA SFT + KD"),
        (7, "quantization"),
        (8, "hybrid policy"),
    ]:
        phases.append(_phase_record(phase, name, "planned", {"run_specific_cli": True}))

    report_dir = Path(config.get("report_dir", "reports"))
    report_outputs = write_final_report_inputs(runs_dir, report_dir)
    phases.append(_phase_record(9, "final report inputs", "completed", report_outputs))

    result = {
        "run_dir": str(run_dir),
        "execute": execute,
        "phases": phases,
    }
    (run_dir / "master_workflow_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 0~9 master experiment workflow.")
    parser.add_argument("--config", default="{}")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config)
    result = run_master_workflow(config, execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
