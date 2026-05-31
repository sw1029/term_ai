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
from term_ai.augmentation.judge_llm import judge_metadata
from term_ai.augmentation.pipeline import apply_judge_validation, auto_filter_metadata, prepare_artifacts
from term_ai.augmentation.orchestrator import generate_split_batch
from term_ai.experiment.explanation_judge import judge_explanations, summarize_explanation_judgments
from term_ai.experiment.reporting import write_final_report_inputs
from term_ai.experiment.runner import init_matrix
from term_ai.contracts import write_jsonl


def _phase_record(phase: int, name: str, status: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"phase": phase, "name": name, "status": status, "details": details or {}}


def _run_phase_job(job: dict[str, Any], cwd: Path) -> dict[str, Any]:
    command = job.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"phase job requires a non-empty command list: {job}")
    missing_paths = [str(path) for path in job.get("requires_paths", []) if not Path(path).exists()]
    if missing_paths:
        return {
            "name": job.get("name", "unnamed"),
            "phase": int(job.get("phase", -1)),
            "status": "skipped",
            "reason": "missing required input paths",
            "missing_paths": missing_paths,
        }
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


def _auto_job_command(experiment_id: str, output_dir: Path, eval_split: str, extra: list[str] | None = None) -> list[str]:
    command = [
        "{python}",
        "-m",
        "term_ai.experiment.hydra_app",
        "execution.run=true",
        f"model.experiment_id={experiment_id}",
        f"execution.output_dir={output_dir}",
        f"evaluation.split={eval_split}",
    ]
    if extra:
        command.extend(extra)
    return command


def _default_phase_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Create the planned Phase 4~8 matrix when explicit phase_jobs are absent."""

    auto_cfg = config.get("auto_phase_jobs") or {}
    if not bool(auto_cfg.get("enabled", False)):
        return []
    output_base = Path(auto_cfg.get("output_dir", Path(config.get("runs_dir", "runs")) / "master_matrix"))
    eval_split = str(auto_cfg.get("eval_split", "dev"))
    model_ids = auto_cfg.get("model_ids") or {}
    gemma_model = str(model_ids.get("gemma", "google/gemma-2-2b-it"))
    qwen_model = str(model_ids.get("qwen", "Qwen/Qwen2.5-3B-Instruct"))
    bitnet_model = str(model_ids.get("bitnet", "microsoft/bitnet-b1.58-2B-4T"))
    jobs: list[dict[str, Any]] = []

    def model_extras(experiment_id: str) -> list[str]:
        if "Gemma" in experiment_id:
            return [f"execution.model_name_or_path={gemma_model}"]
        if "Qwen" in experiment_id:
            return [f"execution.model_name_or_path={qwen_model}"]
        if "BitNet" in experiment_id:
            return [f"execution.model_name_or_path={bitnet_model}"]
        return []

    if bool(auto_cfg.get("include_baselines", True)):
        jobs.extend(
            {
                "phase": 4,
                "name": experiment_id,
                "command": _auto_job_command(experiment_id, output_base / experiment_id, eval_split),
            }
            for experiment_id in ("B0", "B1", "B2", "B3")
        )
        jobs.append(
            {
                "phase": 4,
                "name": "B4",
                "requires_paths": [str(output_base / "B0" / "prediction_log.jsonl")],
                "command": _auto_job_command(
                    "B4",
                    output_base / "B4",
                    eval_split,
                    extra=[f"execution.primary_predictions={output_base / 'B0' / 'prediction_log.jsonl'}"],
                ),
            }
        )

    if bool(auto_cfg.get("include_zero_shot", True)):
        jobs.extend(
            {
                "phase": 4,
                "name": experiment_id,
                "command": _auto_job_command(
                    experiment_id,
                    output_base / experiment_id,
                    eval_split,
                    extra=model_extras(experiment_id),
                ),
            }
            for experiment_id in ("G0-Gemma", "G0-Qwen", "G0-BitNet")
        )

    if bool(auto_cfg.get("include_lora_sft", True)):
        jobs.extend(
            {
                "phase": 5,
                "name": experiment_id,
                "command": _auto_job_command(
                    experiment_id,
                    output_base / experiment_id,
                    eval_split,
                    extra=model_extras(experiment_id),
                ),
            }
            for experiment_id in (
                "G1-Gemma",
                "G1-Qwen",
                "G1-BitNet",
                "G2-Gemma",
                "G2-Qwen",
                "G2-BitNet",
            )
        )
    prompt_cfg = auto_cfg.get("prompt_variation") or {}
    if bool(prompt_cfg.get("enabled", True)):
        train_sft = Path(prompt_cfg.get("train_jsonl", "data/sft/raw_judge_aug_train_sft_v1.jsonl"))
        dev_sft = Path(prompt_cfg.get("dev_jsonl", "data/sft/raw_judge_aug_dev_sft_v1.jsonl"))
        command = [
            "{python}",
            "-m",
            "term_ai.experiment.prompt_variation_sweep",
            "--train-jsonl",
            str(train_sft),
            "--dev-jsonl",
            str(dev_sft),
            "--output-dir",
            str(output_base / "prompt_variation"),
            "--model-name-or-path",
            gemma_model,
        ]
        if prompt_cfg.get("eval_metadata"):
            command.extend(["--eval-metadata", str(prompt_cfg["eval_metadata"]), "--eval-split", eval_split])
        if bool(prompt_cfg.get("execute_training", False)):
            command.append("--execute-training")
        jobs.append(
            {
                "phase": 5,
                "name": "prompt-template-variation",
                "requires_paths": [str(train_sft), str(dev_sft)],
                "command": command,
            }
        )

    if bool(auto_cfg.get("include_lora_kd", True)):
        jobs.extend(
            {
                "phase": 6,
                "name": experiment_id,
                "command": _auto_job_command(
                    experiment_id,
                    output_base / experiment_id,
                    eval_split,
                    extra=model_extras(experiment_id),
                ),
            }
            for experiment_id in ("G3-Gemma", "G3-Qwen", "G3-BitNet")
        )
    kd_sweep_cfg = auto_cfg.get("kd_ablation") or {}
    if bool(kd_sweep_cfg.get("enabled", True)):
        kd_train = Path(kd_sweep_cfg.get("metadata_jsonl", "data/metadata/kd_train_view_v1.jsonl"))
        kd_dev = Path(kd_sweep_cfg.get("dev_metadata_jsonl", "data/metadata/kd_dev_view_v1.jsonl"))
        command = [
            "{python}",
            "-m",
            "term_ai.experiment.kd_sweep",
            "--model-name-or-path",
            gemma_model,
            "--metadata-jsonl",
            str(kd_train),
            "--dev-metadata-jsonl",
            str(kd_dev),
            "--output-dir",
            str(output_base / "G3-kd-ablation"),
        ]
        if kd_sweep_cfg.get("eval_metadata"):
            command.extend(["--eval-metadata", str(kd_sweep_cfg["eval_metadata"]), "--eval-split", eval_split])
        if bool(kd_sweep_cfg.get("execute_training", False)):
            command.append("--execute-training")
        jobs.append(
            {
                "phase": 6,
                "name": "G3-KD-ablation",
                "requires_paths": [str(kd_train), str(kd_dev)],
                "command": command,
            }
        )

    if bool(auto_cfg.get("include_quantization", True)):
        g4_source = str(auto_cfg.get("g4_source_experiment", "G3-Gemma"))
        g4_model = bitnet_model if "BitNet" in g4_source else gemma_model if "Gemma" in g4_source else qwen_model
        adapter = output_base / g4_source / "final_adapter"
        g4_extra = [
            f"execution.model_name_or_path={g4_model}",
            f"execution.adapter_path={adapter}",
        ]
        jobs.append(
            {
                "phase": 7,
                "name": "G4",
                "requires_paths": [str(adapter)],
                "command": _auto_job_command(
                    "G4-8bit",
                    output_base / "G4",
                    eval_split,
                    extra=g4_extra,
                ),
            }
        )

    if bool(auto_cfg.get("include_hybrid", True)):
        primary = output_base / "B0" / "prediction_log.jsonl"
        fallback = output_base / "B4" / "prediction_log.jsonl"
        jobs.append(
            {
                "phase": 8,
                "name": "H1",
                "requires_paths": [str(primary), str(fallback)],
                "command": _auto_job_command(
                    "H1",
                    output_base / "H1",
                    eval_split,
                    extra=[
                        f"execution.primary_predictions={primary}",
                        f"execution.fallback_predictions={fallback}",
                    ],
                ),
            }
        )
    return jobs


def _augmentation_split_totals(aug_cfg: dict[str, Any], total: int) -> dict[str, int]:
    split_totals = {str(key): int(value) for key, value in (aug_cfg.get("split_totals") or {}).items()}
    if not split_totals:
        split_totals = {"train": total}
    elif total and "train" not in split_totals:
        split_totals["train"] = total
    invalid = set(split_totals) - {"train", "dev", "test"}
    if invalid:
        raise ValueError(f"invalid augmentation split totals: {sorted(invalid)}")
    return {split: count for split, count in split_totals.items() if count > 0}


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
    generated_strict_judge_metadata: Path | None = None
    if execute and aug_cfg.get("enabled"):
        total = int(aug_cfg.get("total", 0))
        split_results: dict[str, Any] = {}
        judged_metadata_paths: list[Path] = []
        judge_cfg = aug_cfg.get("judge") or {}
        for split, split_total in _augmentation_split_totals(aug_cfg, total).items():
            candidate_path = Path(
                aug_cfg.get("candidate_path")
                or aug_cfg.get("candidate_path_template", str(data_dir / "aug" / "{split}_aug_candidate_v1.jsonl")).format(split=split)
            )
            auto_path = Path(
                aug_cfg.get("auto_filter_path")
                or aug_cfg.get("auto_filter_path_template", str(data_dir / "aug" / "{split}_aug_auto_pass_v1.jsonl")).format(split=split)
            )
            aug_manifest = generate_split_batch(
                anchors_path=anchors_path,
                output_path=candidate_path,
                total=split_total,
                model=str(aug_cfg.get("model", "gpt-5.4-mini")),
                env_path=aug_cfg.get("env_path", ".env"),
                requests_per_second=float(aug_cfg.get("requests_per_second", 1.0)),
                split=split,
            )
            auto_path.parent.mkdir(parents=True, exist_ok=True)
            auto_counts = auto_filter_metadata(candidate_path, auto_path)
            split_result: dict[str, Any] = {"manifest": aug_manifest, "auto_filter": auto_counts}
            if bool(judge_cfg.get("enabled", False)):
                judge_path = Path(
                    str(judge_cfg.get("output_path_template", data_dir / "judge" / "{split}_judge_v1.jsonl")).format(split=split)
                )
                judged_metadata_path = Path(
                    str(judge_cfg.get("metadata_path_template", data_dir / "metadata" / "{split}_aug_judge_pass_v1.jsonl")).format(split=split)
                )
                judge_path.parent.mkdir(parents=True, exist_ok=True)
                judged_metadata_path.parent.mkdir(parents=True, exist_ok=True)
                judge_counts = judge_metadata(
                    auto_path,
                    judge_path,
                    model=str(judge_cfg.get("model", "gpt-5.4-mini")),
                    env_path=judge_cfg.get("env_path", aug_cfg.get("env_path", ".env")),
                    requests_per_second=float(judge_cfg.get("requests_per_second", aug_cfg.get("requests_per_second", 1.0))),
                    limit=judge_cfg.get("limit"),
                    generator_model=str(aug_cfg.get("model", "gpt-5.4-mini")),
                    enforce_model_separation=bool(judge_cfg.get("enforce_model_separation", True)),
                    reasoning_effort=judge_cfg.get("reasoning_effort"),
                )
                apply_counts = apply_judge_validation(auto_path, judge_path, judged_metadata_path)
                split_result["judge"] = judge_counts
                split_result["judge_apply"] = apply_counts
                split_result["judge_validated_metadata"] = str(judged_metadata_path)
                judged_metadata_paths.append(judged_metadata_path)
            split_results[split] = split_result
        if judged_metadata_paths:
            merged_rows: list[dict[str, Any]] = []
            for path in judged_metadata_paths:
                with path.open("r", encoding="utf-8") as handle:
                    merged_rows.extend(json.loads(line) for line in handle if line.strip())
            generated_strict_judge_metadata = Path(
                aug_cfg.get("merged_judge_metadata_path", data_dir / "metadata" / "aug_judge_pass_v1.jsonl")
            )
            write_jsonl(generated_strict_judge_metadata, merged_rows)
            split_results["merged_judge_validated_metadata"] = str(generated_strict_judge_metadata)
        phases.append(
            _phase_record(2, "augmentation generation and validation", "completed", split_results)
        )
    else:
        phases.append(_phase_record(2, "augmentation generation and validation", "planned", {"execute_required": True}))

    strict_judge_metadata = generated_strict_judge_metadata or config.get("strict_judge_metadata") or config.get("approved_metadata")
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

    explicit_jobs = list(config.get("phase_jobs", []) or [])
    configured_jobs = explicit_jobs if explicit_jobs else _default_phase_jobs(config)
    configured_jobs = [job for job in configured_jobs if bool(job.get("enabled", True))]
    jobs_by_phase: dict[int, list[dict[str, Any]]] = {}
    for job in configured_jobs:
        jobs_by_phase.setdefault(int(job.get("phase", -1)), []).append(job)

    for phase, name in [
        (4, "baselines and small LM zero-shot"),
        (5, "LoRA SFT and prompt variation"),
        (6, "LoRA SFT + KD"),
        (7, "quantization"),
        (8, "hybrid policy"),
    ]:
        jobs = jobs_by_phase.get(phase, [])
        if execute and jobs:
            results = []
            for job in jobs:
                result = _run_phase_job(job, Path.cwd())
                results.append(result)
                if result.get("returncode") not in {0, None} and result.get("status") != "skipped":
                    break
            failed = [result for result in results if result.get("returncode") not in {0, None} and result.get("status") != "skipped"]
            phases.append(_phase_record(phase, name, "failed" if failed else "completed", {"jobs": results}))
            if failed:
                break
        else:
            phases.append(
                _phase_record(
                    phase,
                    name,
                    "planned",
                    {"configured_jobs": len(jobs), "execute_required": bool(jobs), "auto_generated": not bool(explicit_jobs)},
                )
            )

    explanation_cfg = config.get("explanation_judge") or {}
    if execute and bool(explanation_cfg.get("enabled", False)):
        if not explanation_cfg.get("judge_model"):
            raise ValueError("explanation_judge.judge_model is required when explanation judge is enabled")
        judgment_outputs: list[dict[str, Any]] = []
        for predictions in explanation_cfg.get("prediction_logs", []):
            predictions_path = Path(predictions)
            output_dir = Path(explanation_cfg.get("output_dir", runs_dir / "explanation_judge"))
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = predictions_path.parent.name or predictions_path.stem
            judgments_path = output_dir / f"{stem}_explanation_judgments.jsonl"
            summary_path = output_dir / f"{stem}_explanation_judgment_summary.json"
            judge_counts = judge_explanations(
                predictions_path,
                judgments_path,
                judge_model=str(explanation_cfg["judge_model"]),
                generator_model=explanation_cfg.get("generator_model"),
                env_path=explanation_cfg.get("env_path", ".env"),
                requests_per_second=float(explanation_cfg.get("requests_per_second", 1.0)),
                limit=explanation_cfg.get("limit"),
                enforce_model_separation=bool(explanation_cfg.get("enforce_model_separation", True)),
            )
            summary = summarize_explanation_judgments(judgments_path, summary_path)
            judgment_outputs.append(
                {
                    "predictions": str(predictions_path),
                    "judgments": str(judgments_path),
                    "summary": str(summary_path),
                    "judge_counts": judge_counts,
                    "summary_metrics": summary,
                }
            )
        phases.append(_phase_record(8, "explanation judge", "completed", {"outputs": judgment_outputs}))
    elif bool(explanation_cfg.get("enabled", False)):
        phases.append(_phase_record(8, "explanation judge", "planned", {"execute_required": True}))

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
