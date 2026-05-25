from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from term_ai.augmentation.dataset_builder import (
    build_kd_metadata_view,
    build_raw_mcq_metadata_from_anchors,
    build_raw_sft_from_anchors,
    build_raw_aug_sft_by_split,
    build_strict_eval_sets,
    build_validated_aug_sft_by_split,
)
from term_ai.augmentation.pipeline import auto_filter_metadata, prepare_artifacts
from term_ai.augmentation.orchestrator import generate_train_batch
from term_ai.experiment.reporting import write_final_report_inputs
from term_ai.experiment.runner import init_matrix


def _phase_record(phase: int, name: str, status: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"phase": phase, "name": name, "status": status, "details": details or {}}


def _run_phase_job(job: dict[str, Any], cwd: Path) -> dict[str, Any]:
    command = job.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"phase job requires a non-empty command list: {job}")
    resolved = [sys.executable if part == "{python}" else str(part) for part in command]
    completed = subprocess.run(resolved, cwd=str(cwd), check=False, capture_output=True, text=True)
    return {
        "name": job.get("name", "unnamed"),
        "phase": int(job.get("phase", -1)),
        "command": resolved,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


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

    strict_judge_metadata = config.get("strict_judge_metadata") or config.get("approved_metadata")
    raw_metadata = data_dir / "metadata" / "raw_mcq_v1.jsonl"
    if strict_judge_metadata and Path(strict_judge_metadata).exists():
        validated_counts = build_validated_aug_sft_by_split(
            strict_judge_metadata,
            data_dir / "sft",
            min_status=str(config.get("trainable_min_status", "aug_judge_pass")),
        )
        raw_aug_counts = build_raw_aug_sft_by_split(
            data_dir / "sft",
            strict_judge_metadata,
            data_dir / "sft",
            min_status=str(config.get("trainable_min_status", "aug_judge_pass")),
        )
        eval_counts = build_strict_eval_sets(
            raw_metadata,
            strict_judge_metadata,
            data_dir / "eval",
            min_status=str(config.get("trainable_min_status", "aug_judge_pass")),
        )
        kd_counts: dict[str, int] | None = None
        raw_teacher_scores = config.get("raw_teacher_scores")
        if raw_teacher_scores:
            kd_counts = build_kd_metadata_view(
                raw_metadata,
                strict_judge_metadata,
                data_dir / "metadata" / "kd_train_view_v1.jsonl",
                min_status=str(config.get("trainable_min_status", "aug_judge_pass")),
                raw_teacher_scores_path=raw_teacher_scores,
                include_raw=True,
                require_raw_teacher_scores=True,
            )
        phases.append(
            _phase_record(
                3,
                "strict judge data views",
                "completed",
                {"validated_sft": validated_counts, "raw_aug_sft": raw_aug_counts, "eval": eval_counts, "kd": kd_counts},
            )
        )
    else:
        phases.append(_phase_record(3, "baselines and strict eval views", "planned", {"needs_strict_judge_metadata": True}))

    configured_jobs = [job for job in config.get("phase_jobs", []) if bool(job.get("enabled", True))]
    jobs_by_phase: dict[int, list[dict[str, Any]]] = {}
    for job in configured_jobs:
        jobs_by_phase.setdefault(int(job.get("phase", -1)), []).append(job)

    for phase, name in [
        (4, "small LM zero-shot"),
        (5, "LoRA SFT"),
        (6, "LoRA SFT + KD"),
        (7, "quantization"),
        (8, "hybrid policy"),
    ]:
        jobs = jobs_by_phase.get(phase, [])
        if execute and jobs:
            results = [_run_phase_job(job, Path.cwd()) for job in jobs]
            failed = [result for result in results if result["returncode"] != 0]
            phases.append(_phase_record(phase, name, "failed" if failed else "completed", {"jobs": results}))
            if failed:
                break
        else:
            phases.append(
                _phase_record(
                    phase,
                    name,
                    "planned",
                    {"configured_jobs": len(jobs), "execute_required": bool(jobs)},
                )
            )

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
