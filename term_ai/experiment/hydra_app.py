from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from term_ai.contracts import DEFAULT_TRAINABLE_AUG_STATUS, RAW_GT_STATUS
from term_ai.experiment.runner import create_run_dir, init_matrix, write_resume_state, write_run_manifest
from term_ai.experiment.workflow import run_master_workflow


def _value(value: object) -> object | None:
    return None if value in {None, ""} else value


def _registry(cfg: DictConfig) -> dict[str, str]:
    if "hf_model_ids" not in cfg.model:
        return {}
    return {
        str(key): str(value)
        for key, value in OmegaConf.to_container(cfg.model.hf_model_ids, resolve=True).items()
    }


def _model_name_or_path(cfg: DictConfig, experiment_id: str) -> str | None:
    configured = _value(cfg.execution.model_name_or_path)
    if configured:
        return str(configured)
    registry = _registry(cfg)
    return registry.get(experiment_id)


def _default_metadata(cfg: DictConfig, experiment_id: str) -> str:
    if _value(cfg.execution.metadata):
        return str(cfg.execution.metadata)
    if (
        experiment_id in {"B0", "B1", "B2", "B3", "B4"}
        or experiment_id.startswith("G0")
        or experiment_id.startswith("G1")
        or experiment_id.startswith("G2")
    ):
        return str(cfg.execution.raw_metadata)
    if experiment_id in {"E1", "G3-Gemma", "G3-Qwen"}:
        return str(cfg.execution.kd_metadata)
    if experiment_id.startswith("G4"):
        return str(cfg.execution.raw_metadata)
    return str(cfg.execution.strict_judge_metadata)


def _default_min_status(cfg: DictConfig, experiment_id: str) -> str:
    if _value(cfg.execution.min_status):
        return str(cfg.execution.min_status)
    if (
        experiment_id in {"B0", "B1", "B2", "B3", "B4"}
        or experiment_id.startswith("G0")
        or experiment_id.startswith("G1")
        or experiment_id.startswith("G2")
        or experiment_id.startswith("G4")
    ):
        return RAW_GT_STATUS
    return DEFAULT_TRAINABLE_AUG_STATUS


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
        metadata = _default_metadata(cfg, experiment_id)
        eval_split = str(cfg.evaluation.split)
        min_status = _default_min_status(cfg, experiment_id)
        final_test_once = bool(cfg.evaluation.final_test_once) and not bool(cfg.execution.allow_repeat_test)
        test_lock_dir = _value(cfg.execution.test_lock_dir)
        execution_resume = bool(cfg.execution.resume)
        progress_interval_items = int(cfg.logging.progress_interval_items)

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
                test_lock_dir=test_lock_dir,
                resume=execution_resume,
                progress_interval_items=progress_interval_items,
                backup_weights=bool(cfg.training.weight_backup),
            )
        elif experiment_id == "B3":
            from term_ai.experiment.reranker import run_reranker

            reranker_cfg = cfg.execution.reranker
            result = run_reranker(
                metadata,
                output_dir,
                eval_split=eval_split,
                min_status=min_status,
                fine_tune=bool(reranker_cfg.fine_tune),
                epochs=int(reranker_cfg.epochs),
                batch_size=int(reranker_cfg.batch_size),
                score_normalization=str(reranker_cfg.score_normalization),
                threshold=_value(reranker_cfg.threshold),
                threshold_split=str(reranker_cfg.threshold_split),
                final_test_once=final_test_once,
                test_lock_dir=test_lock_dir,
                resume=execution_resume,
                progress_interval_items=progress_interval_items,
                save_steps=_value(cfg.training.save_steps),
                save_total_limit=int(cfg.training.save_total_limit),
                backup_weights=bool(cfg.training.weight_backup),
                backup_checkpoints=bool(cfg.training.backup_checkpoints),
            )
        elif experiment_id == "B4":
            from term_ai.experiment.api_recheck import run_api_recheck

            api_cfg = cfg.execution.api_recheck
            result = run_api_recheck(
                metadata,
                output_dir,
                provider=str(api_cfg.provider),
                model=str(api_cfg.model),
                env_path=str(api_cfg.env_path),
                base_url=_value(api_cfg.base_url),
                api_key_env=_value(api_cfg.api_key_env),
                eval_split=eval_split,
                min_status=min_status,
                requests_per_second=float(api_cfg.requests_per_second),
                primary_predictions_path=cfg.execution.primary_predictions,
                confidence_threshold=float(api_cfg.confidence_threshold),
                limit=cfg.execution.limit,
                input_cost_per_1m_tokens=float(api_cfg.input_cost_per_1m_tokens),
                output_cost_per_1m_tokens=float(api_cfg.output_cost_per_1m_tokens),
                pricing_path=_value(api_cfg.pricing_path),
                final_test_once=final_test_once,
                test_lock_dir=test_lock_dir,
                resume=execution_resume,
                progress_interval_items=progress_interval_items,
            )
        elif experiment_id.startswith("G0"):
            from term_ai.experiment.lm_eval import run_hf_zero_shot

            model_name_or_path = _model_name_or_path(cfg, experiment_id)
            if not model_name_or_path:
                raise ValueError("execution.model_name_or_path is required for G0")
            result = run_hf_zero_shot(
                metadata,
                output_dir,
                model_name_or_path=model_name_or_path,
                eval_split=eval_split,
                min_status=min_status,
                limit=cfg.execution.limit,
                final_test_once=final_test_once,
                experiment_id=experiment_id,
                test_lock_dir=test_lock_dir,
                local_cost_per_hour_usd=float(cfg.execution.local_cost_per_hour_usd),
                resume=execution_resume,
                progress_interval_items=progress_interval_items,
            )
        elif experiment_id in {"G1-Gemma", "G1-Qwen", "G2-Gemma", "G2-Qwen"}:
            from term_ai.experiment.training import LoRATrainingConfig, train_lora_sft

            train_jsonl = _value(cfg.execution.train_sft_jsonl) or (
                cfg.data.raw_train_sft_jsonl if experiment_id.startswith("G1") else cfg.data.raw_judge_aug_train_sft_jsonl
            )
            dev_jsonl = _value(cfg.execution.dev_sft_jsonl) or (
                cfg.data.raw_dev_sft_jsonl if experiment_id.startswith("G1") else cfg.data.raw_judge_aug_dev_sft_jsonl
            )
            model_name_or_path = _model_name_or_path(cfg, experiment_id)
            if not model_name_or_path:
                raise ValueError("execution.model_name_or_path is required for G1/G2 LoRA SFT")
            adapter = train_lora_sft(
                LoRATrainingConfig(
                    model_name_or_path=model_name_or_path,
                    train_jsonl=str(train_jsonl),
                    dev_jsonl=str(dev_jsonl),
                    output_dir=str(output_dir),
                    lora_r=int(cfg.training.lora.r),
                    lora_alpha=int(cfg.training.lora.alpha),
                    lora_dropout=float(cfg.training.lora.dropout),
                    resume=bool(cfg.training.resume),
                    backup_weights=bool(cfg.training.weight_backup),
                    backup_checkpoints=bool(cfg.training.backup_checkpoints),
                    save_steps=_value(cfg.training.save_steps),
                    save_total_limit=int(cfg.training.save_total_limit),
                    eval_metadata=metadata,
                    eval_split=eval_split,
                    progress_interval_items=progress_interval_items,
                )
            )
            result = {"final_adapter": str(adapter)}
        elif experiment_id in {"G3-Gemma", "G3-Qwen"}:
            from term_ai.experiment.lora_kd import LoRAKDConfig, train_lora_sft_kd

            model_name_or_path = _model_name_or_path(cfg, experiment_id)
            if not model_name_or_path:
                raise ValueError("execution.model_name_or_path is required for G3 LoRA KD")
            adapter = train_lora_sft_kd(
                LoRAKDConfig(
                    model_name_or_path=model_name_or_path,
                    metadata_jsonl=metadata,
                    dev_metadata_jsonl=str(_value(cfg.execution.kd_dev_metadata) or metadata),
                    output_dir=str(output_dir),
                    min_status=min_status,
                    dev_min_status=min_status,
                    lora_r=int(cfg.training.lora.r),
                    lora_alpha=int(cfg.training.lora.alpha),
                    lora_dropout=float(cfg.training.lora.dropout),
                    lambda_soft=float(cfg.training.kd.lambda_soft),
                    include_rationale=bool(cfg.training.kd.include_rationale),
                    require_teacher_scores=bool(cfg.training.kd.require_teacher_scores),
                    response_format=str(cfg.training.kd.response_format),
                    resume=bool(cfg.training.resume),
                    backup_weights=bool(cfg.training.weight_backup),
                    backup_checkpoints=bool(cfg.training.backup_checkpoints),
                    save_steps=_value(cfg.training.save_steps),
                    save_total_limit=int(cfg.training.save_total_limit),
                    progress_interval_items=progress_interval_items,
                )
            )
            result = {"final_adapter": str(adapter)}
        elif experiment_id.startswith("G4"):
            from term_ai.experiment.quantization import compare_quantization

            model_name_or_path = _model_name_or_path(cfg, experiment_id)
            if not model_name_or_path or not cfg.execution.adapter_path:
                raise ValueError("execution.model_name_or_path and execution.adapter_path are required for G4")
            result = compare_quantization(
                metadata,
                output_dir,
                model_name_or_path=model_name_or_path,
                adapter_path=str(cfg.execution.adapter_path),
                eval_split=eval_split,
                min_status=min_status,
                limit=cfg.execution.limit,
                g3_checkpoint_id=str(cfg.execution.adapter_path),
                require_g3_manifest=bool(cfg.training.quantization.require_g3_manifest),
                final_test_once=final_test_once,
                test_lock_dir=test_lock_dir,
                local_cost_per_hour_usd=float(cfg.execution.local_cost_per_hour_usd),
                resume=execution_resume,
                progress_interval_items=progress_interval_items,
            )
        elif experiment_id == "E1":
            from term_ai.experiment.kd_scorer import train_kd_scorer

            result = train_kd_scorer(
                metadata,
                output_dir,
                eval_split=eval_split,
                min_status=min_status,
                final_test_once=final_test_once,
                test_lock_dir=test_lock_dir,
                require_teacher_scores=bool(cfg.training.kd.require_teacher_scores),
                resume=bool(cfg.training.resume),
                progress_interval_items=progress_interval_items,
                backup_weights=bool(cfg.training.weight_backup),
                backup_checkpoints=bool(cfg.training.backup_checkpoints),
            )
        elif experiment_id == "H1":
            from term_ai.experiment.hybrid import run_hybrid_policy, tune_hybrid_policy

            if not cfg.execution.primary_predictions or not cfg.execution.fallback_predictions:
                raise ValueError("H1 requires execution.primary_predictions and execution.fallback_predictions")
            hybrid_cfg = cfg.execution.hybrid
            if bool(hybrid_cfg.tune_policy):
                result = tune_hybrid_policy(
                    cfg.execution.primary_predictions,
                    cfg.execution.fallback_predictions,
                    output_dir,
                    cross_encoder_predictions=_value(cfg.execution.cross_encoder_predictions),
                    threshold_grid=[float(value) for value in hybrid_cfg.threshold_grid],
                    primary_cost_per_1000=float(hybrid_cfg.primary_cost_per_1000),
                    cross_encoder_cost_per_1000=float(hybrid_cfg.cross_encoder_cost_per_1000),
                    fallback_cost_per_1000=float(hybrid_cfg.fallback_cost_per_1000),
                    resume=execution_resume,
                    progress_interval_items=progress_interval_items,
                )
            else:
                result = run_hybrid_policy(
                    cfg.execution.primary_predictions,
                    cfg.execution.fallback_predictions,
                    output_dir,
                    cross_encoder_predictions=_value(cfg.execution.cross_encoder_predictions),
                    low_confidence_threshold=float(hybrid_cfg.low_confidence_threshold),
                    high_confidence_threshold=float(hybrid_cfg.high_confidence_threshold),
                    primary_cost_per_1000=float(hybrid_cfg.primary_cost_per_1000),
                    cross_encoder_cost_per_1000=float(hybrid_cfg.cross_encoder_cost_per_1000),
                    fallback_cost_per_1000=float(hybrid_cfg.fallback_cost_per_1000),
                    resume=execution_resume,
                    progress_interval_items=progress_interval_items,
                )
        else:
            raise ValueError(f"Hydra runner for {experiment_id} is not implemented")
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
