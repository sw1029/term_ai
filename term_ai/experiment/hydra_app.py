from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from term_ai.experiment.runner import create_run_dir, init_matrix, write_resume_state, write_run_manifest
from term_ai.experiment.workflow import run_master_workflow


@hydra.main(version_base=None, config_path="../../configs/experiment", config_name="default")
def main(cfg: DictConfig) -> None:
    if bool(cfg.workflow.master_enabled):
        workflow_config = OmegaConf.to_container(cfg.workflow.config, resolve=True)
        result = run_master_workflow(
            workflow_config,
            execute=bool(cfg.workflow.execute_expensive_steps),
        )
        print(OmegaConf.to_yaml(OmegaConf.create(result), resolve=True))
        return

    if bool(cfg.execution.run):
        experiment_id = str(cfg.model.experiment_id)
        output_dir = Path(cfg.execution.output_dir)
        metadata = str(cfg.execution.metadata)
        eval_split = str(cfg.evaluation.split)
        min_status = "raw_gt" if experiment_id in {"B0", "B1", "B2"} else "aug_human_pass"
        final_test_once = bool(cfg.evaluation.final_test_once) and not bool(cfg.execution.allow_repeat_test)

        if experiment_id in {"B0", "B1", "B2"}:
            from term_ai.experiment.baselines import run_baseline

            method = {"B0": "b0", "B1": "logistic", "B2": "mlp"}[experiment_id]
            result = run_baseline(
                metadata,
                output_dir,
                method=method,
                eval_split=eval_split,
                min_status=min_status,
                train_metadata_path=cfg.execution.train_metadata,
                final_test_once=final_test_once,
            )
        elif experiment_id == "B3":
            from term_ai.experiment.reranker import run_reranker

            result = run_reranker(metadata, output_dir, eval_split=eval_split, final_test_once=final_test_once)
        elif experiment_id == "B4":
            from term_ai.experiment.api_recheck import run_api_recheck

            result = run_api_recheck(
                metadata,
                output_dir,
                eval_split=eval_split,
                primary_predictions_path=cfg.execution.primary_predictions,
                limit=cfg.execution.limit,
                final_test_once=final_test_once,
            )
        elif experiment_id.startswith("G0"):
            from term_ai.experiment.lm_eval import run_hf_zero_shot

            if not cfg.execution.model_name_or_path:
                raise ValueError("execution.model_name_or_path is required for G0")
            result = run_hf_zero_shot(
                metadata,
                output_dir,
                model_name_or_path=str(cfg.execution.model_name_or_path),
                eval_split=eval_split,
                limit=cfg.execution.limit,
                final_test_once=final_test_once,
            )
        elif experiment_id == "E1":
            from term_ai.experiment.kd_scorer import train_kd_scorer

            result = train_kd_scorer(
                metadata,
                output_dir,
                eval_split=eval_split,
                final_test_once=final_test_once,
            )
        elif experiment_id == "H1":
            from term_ai.experiment.hybrid import run_hybrid_policy

            if not cfg.execution.primary_predictions or not cfg.execution.fallback_predictions:
                raise ValueError("H1 requires execution.primary_predictions and execution.fallback_predictions")
            result = run_hybrid_policy(
                cfg.execution.primary_predictions,
                cfg.execution.fallback_predictions,
                output_dir,
            )
        else:
            raise ValueError(f"Hydra runner for {experiment_id} is not implemented; use the dedicated CLI")
        print(OmegaConf.to_yaml(OmegaConf.create(result), resolve=True))
        return

    matrix_path = Path(cfg.model.matrix_path)
    init_matrix(matrix_path)

    run_dir = create_run_dir(Path(cfg.logging.runs_dir), str(cfg.model.experiment_id))
    config_snapshot = OmegaConf.to_container(cfg, resolve=True)
    write_run_manifest(run_dir, str(cfg.model.experiment_id), config_snapshot)
    write_resume_state(run_dir, {"stage": "initialized", "hydra_config_snapshot": True})
    print(OmegaConf.to_yaml(OmegaConf.create({"run_dir": str(run_dir), "matrix_path": str(matrix_path)}), resolve=True))


if __name__ == "__main__":
    main()
