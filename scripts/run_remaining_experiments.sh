#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run the remaining B0-H1 experiment flow from docs/b0_to_h1_conda_term_toeic_commands.md.

Usage:
  scripts/run_remaining_experiments.sh [all|dev|final|stats|report]

Defaults:
  mode: all
  PYTHON_BIN=python
  FORCE=0
  DRY_RUN=0
  INCLUDE_BITNET=1
  BITNET_MODEL_ID=microsoft/bitnet-b1.58-2B-4T
  RUN_BITNET_G4=0
  G4_FINAL_MODEL=all      # qwen, gemma, bitnet, both, or all

Useful environment overrides:
  PYTHON_BIN=/path/to/python
  FORCE=1
  DRY_RUN=1
  INCLUDE_BITNET=0
  BITNET_MODEL_ID=microsoft/bitnet-b1.58-2B-4T
  RUN_BITNET_G4=0         # skip BitNet G4 bnb quantization if the runtime rejects it
  ENV_PATH=.env
  TEST_LOCK_DIR=runs/_test_locks
  B4_PROVIDER=openai
  B4_MODEL=gpt-5.4-mini
  API_RECHECK_RPS=1
  API_RECHECK_CONFIDENCE_THRESHOLD=0.7
  LOCAL_COST_PER_HOUR_USD=0
  STAT_SAMPLES=1000
  H1_LOW_CONFIDENCE_THRESHOLD=      # empty means read runs/H1_dev/hybrid_policy_tuning.json
  H1_HIGH_CONFIDENCE_THRESHOLD=     # empty means read runs/H1_dev/hybrid_policy_tuning.json
  GENERATE_RAW_TEACHER_SCORES=0
EOF
}

MODE="${1:-all}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
ENV_PATH="${ENV_PATH:-.env}"
TEST_LOCK_DIR="${TEST_LOCK_DIR:-runs/_test_locks}"
INCLUDE_BITNET="${INCLUDE_BITNET:-1}"
BITNET_MODEL_ID="${BITNET_MODEL_ID:-microsoft/bitnet-b1.58-2B-4T}"
RUN_BITNET_G4="${RUN_BITNET_G4:-0}"
G4_FINAL_MODEL="${G4_FINAL_MODEL:-all}"
B4_PROVIDER="${B4_PROVIDER:-openai}"
B4_MODEL="${B4_MODEL:-gpt-5.4-mini}"
API_RECHECK_RPS="${API_RECHECK_RPS:-1}"
API_RECHECK_CONFIDENCE_THRESHOLD="${API_RECHECK_CONFIDENCE_THRESHOLD:-0.7}"
LOCAL_COST_PER_HOUR_USD="${LOCAL_COST_PER_HOUR_USD:-0}"
STAT_SAMPLES="${STAT_SAMPLES:-1000}"
GENERATE_RAW_TEACHER_SCORES="${GENERATE_RAW_TEACHER_SCORES:-0}"
RAW_TEACHER_MODEL="${RAW_TEACHER_MODEL:-gpt-5.4-mini}"
RAW_TEACHER_REASONING_EFFORT="${RAW_TEACHER_REASONING_EFFORT:-low}"
H1_LOW_CONFIDENCE_THRESHOLD="${H1_LOW_CONFIDENCE_THRESHOLD:-}"
H1_HIGH_CONFIDENCE_THRESHOLD="${H1_HIGH_CONFIDENCE_THRESHOLD:-}"

log() {
  printf '[remaining-exp] %s\n' "$*"
}

die() {
  printf '[remaining-exp] ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  if [[ ! -f "$1" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      log "dry-run: missing required file would block real run: $1"
      return 0
    fi
    die "missing required file: $1"
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      log "dry-run: missing required directory would block real run: $1"
      return 0
    fi
    die "missing required directory: $1"
  fi
}

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_step() {
  local marker="$1"
  local desc="$2"
  shift 2
  if [[ "${FORCE}" != "1" && -e "${marker}" ]]; then
    log "skip: ${desc} (${marker} exists)"
    return 0
  fi
  log "run: ${desc}"
  print_cmd "$@"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

ensure_common_inputs() {
  require_file "data/metadata/raw_mcq_v1.jsonl"
  require_file "data/metadata/raw_train_mcq_v1.jsonl"
  require_file "data/metadata/raw_dev_mcq_v1.jsonl"
  require_file "data/metadata/raw_test_mcq_v1.jsonl"
  require_file "data/metadata/train_aug_judge_pass_v1.jsonl"
  require_file "data/metadata/dev_aug_judge_pass_v1.jsonl"
  require_file "data/metadata/test_aug_judge_pass_v1.jsonl"
  require_file "data/sft/raw_train_sft_v1.jsonl"
  require_file "data/sft/raw_judge_aug_train_sft_v1.jsonl"
}

ensure_raw_teacher_scores() {
  if [[ -f "data/metadata/raw_teacher_scores_v1.jsonl" ]]; then
    return 0
  fi
  if [[ "${GENERATE_RAW_TEACHER_SCORES}" != "1" ]]; then
    die "data/metadata/raw_teacher_scores_v1.jsonl is missing. Set GENERATE_RAW_TEACHER_SCORES=1 to generate it through the API."
  fi
  run_step "data/metadata/raw_teacher_scores_v1.jsonl" "raw teacher scores" \
    "${PYTHON_BIN}" -m term_ai.augmentation.raw_teacher_scores \
      --metadata data/metadata/raw_mcq_v1.jsonl \
      --output data/metadata/raw_teacher_scores_v1.jsonl \
      --model "${RAW_TEACHER_MODEL}" \
      --env "${ENV_PATH}" \
      --requests-per-second "${API_RECHECK_RPS}" \
      --reasoning-effort "${RAW_TEACHER_REASONING_EFFORT}" \
      --resume
}

ensure_kd_views() {
  ensure_common_inputs
  ensure_raw_teacher_scores

  run_step "data/metadata/kd_all_view_v1.jsonl" "KD all-split metadata view" \
    "${PYTHON_BIN}" -m term_ai.augmentation.dataset_builder kd-metadata \
      --raw-metadata data/metadata/raw_mcq_v1.jsonl \
      --generated-metadata data/metadata/aug_judge_pass_v1.jsonl \
      --output data/metadata/kd_all_view_v1.jsonl \
      --min-status aug_judge_pass \
      --raw-teacher-scores data/metadata/raw_teacher_scores_v1.jsonl

  run_step "data/metadata/kd_train_view_v1.jsonl" "KD train metadata view" \
    "${PYTHON_BIN}" -m term_ai.augmentation.dataset_builder kd-metadata \
      --raw-metadata data/metadata/raw_train_mcq_v1.jsonl \
      --generated-metadata data/metadata/train_aug_judge_pass_v1.jsonl \
      --output data/metadata/kd_train_view_v1.jsonl \
      --min-status aug_judge_pass \
      --raw-teacher-scores data/metadata/raw_teacher_scores_v1.jsonl

  run_step "data/metadata/kd_dev_view_v1.jsonl" "KD dev metadata view" \
    "${PYTHON_BIN}" -m term_ai.augmentation.dataset_builder kd-metadata \
      --raw-metadata data/metadata/raw_dev_mcq_v1.jsonl \
      --generated-metadata data/metadata/dev_aug_judge_pass_v1.jsonl \
      --output data/metadata/kd_dev_view_v1.jsonl \
      --min-status aug_judge_pass \
      --raw-teacher-scores data/metadata/raw_teacher_scores_v1.jsonl

  run_step "data/metadata/kd_test_view_v1.jsonl" "KD test metadata view" \
    "${PYTHON_BIN}" -m term_ai.augmentation.dataset_builder kd-metadata \
      --raw-metadata data/metadata/raw_test_mcq_v1.jsonl \
      --generated-metadata data/metadata/test_aug_judge_pass_v1.jsonl \
      --output data/metadata/kd_test_view_v1.jsonl \
      --min-status aug_judge_pass \
      --raw-teacher-scores data/metadata/raw_teacher_scores_v1.jsonl
}

run_dev_remaining() {
  ensure_kd_views
  require_file "runs/B0_dev/prediction_log.jsonl"
  require_file "runs/B3_dev/prediction_log.jsonl"
  require_file "runs/B4_dev/prediction_log.jsonl"

  run_bitnet_dev_track

  run_step "runs/E1_dev/metric_log.json" "E1 dev embedding scorer KD" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=E1 \
      execution.output_dir=runs/E1_dev \
      execution.kd_metadata=data/metadata/kd_all_view_v1.jsonl \
      execution.min_status=any \
      evaluation.split=dev \
      training.kd.require_teacher_scores=true

  run_step "runs/H1_dev/hybrid_policy_tuning.json" "H1 dev hybrid policy tuning" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=H1 \
      execution.output_dir=runs/H1_dev \
      evaluation.split=dev \
      execution.primary_predictions=runs/B0_dev/prediction_log.jsonl \
      execution.cross_encoder_predictions=runs/B3_dev/prediction_log.jsonl \
      execution.fallback_predictions=runs/B4_dev/prediction_log.jsonl \
      execution.hybrid.tune_policy=true \
      'execution.hybrid.threshold_grid=[0.3,0.4,0.5,0.6,0.7,0.8]' \
      execution.hybrid.primary_cost_per_1000=0 \
      execution.hybrid.cross_encoder_cost_per_1000=0 \
      execution.hybrid.fallback_cost_per_1000=0

  read_h1_policy
  run_step "runs/H1_dev_fixed/metric_log.json" "H1 dev fixed selected policy" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=H1 \
      execution.output_dir=runs/H1_dev_fixed \
      evaluation.split=dev \
      execution.primary_predictions=runs/B0_dev/prediction_log.jsonl \
      execution.cross_encoder_predictions=runs/B3_dev/prediction_log.jsonl \
      execution.fallback_predictions=runs/B4_dev/prediction_log.jsonl \
      execution.hybrid.tune_policy=false \
      execution.hybrid.low_confidence_threshold="${H1_LOW_CONFIDENCE_THRESHOLD}" \
      execution.hybrid.high_confidence_threshold="${H1_HIGH_CONFIDENCE_THRESHOLD}"
}

run_bitnet_dev_track() {
  if [[ "${INCLUDE_BITNET}" != "1" ]]; then
    log "skip: BitNet dev track (INCLUDE_BITNET=${INCLUDE_BITNET})"
    return 0
  fi

  run_step "runs/G0_BitNet_dev/metric_log.json" "G0 BitNet dev zero-shot" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=G0-BitNet \
      execution.output_dir=runs/G0_BitNet_dev \
      execution.model_name_or_path="${BITNET_MODEL_ID}" \
      evaluation.split=dev \
      execution.local_cost_per_hour_usd="${LOCAL_COST_PER_HOUR_USD}"

  run_step "runs/G1_BitNet_dev/post_train_eval/metric_log.json" "G1 BitNet dev raw LoRA SFT" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=G1-BitNet \
      execution.output_dir=runs/G1_BitNet_dev \
      execution.model_name_or_path="${BITNET_MODEL_ID}" \
      evaluation.split=dev \
      training.save_total_limit=3

  run_step "runs/G2_BitNet_dev/post_train_eval/metric_log.json" "G2 BitNet dev raw+aug LoRA SFT" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=G2-BitNet \
      execution.output_dir=runs/G2_BitNet_dev \
      execution.model_name_or_path="${BITNET_MODEL_ID}" \
      evaluation.split=dev \
      training.save_total_limit=3

  run_step "runs/G3_BitNet_dev/final_adapter/adapter_config.json" "G3 BitNet dev LoRA KD" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=G3-BitNet \
      execution.output_dir=runs/G3_BitNet_dev \
      execution.model_name_or_path="${BITNET_MODEL_ID}" \
      execution.kd_metadata=data/metadata/kd_train_view_v1.jsonl \
      execution.kd_dev_metadata=data/metadata/kd_dev_view_v1.jsonl \
      execution.min_status=any \
      evaluation.split=dev \
      training.kd.lambda_soft=0.5 \
      training.kd.include_rationale=true \
      training.kd.require_teacher_scores=true

  run_lm_eval_adapter_dev "runs/G3_BitNet_dev/post_train_eval/metric_log.json" "G3 BitNet dev post-train eval" \
    runs/G3_BitNet_dev/post_train_eval "${BITNET_MODEL_ID}" runs/G3_BitNet_dev/final_adapter G3-BitNet

  if [[ "${RUN_BITNET_G4}" == "1" ]]; then
    run_step "runs/G4_BitNet_dev/quantization_compare.json" "G4 BitNet dev quantization comparison" \
      "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
        execution.run=true \
        model.experiment_id=G4-8bit \
        execution.output_dir=runs/G4_BitNet_dev \
        execution.model_name_or_path="${BITNET_MODEL_ID}" \
        execution.adapter_path=runs/G3_BitNet_dev/final_adapter \
        evaluation.split=dev \
        execution.local_cost_per_hour_usd="${LOCAL_COST_PER_HOUR_USD}"
  else
    log "skip: G4 BitNet dev quantization comparison (RUN_BITNET_G4=${RUN_BITNET_G4})"
  fi
}

run_lm_eval_adapter_dev() {
  local marker="$1"
  local desc="$2"
  local output_dir="$3"
  local model_name="$4"
  local adapter_path="$5"
  local experiment_id="$6"
  local trust_remote="${7:-false}"
  require_dir "${adapter_path}"
  if [[ "${trust_remote}" == "true" ]]; then
    run_step "${marker}" "${desc}" \
      "${PYTHON_BIN}" -m term_ai.experiment.lm_eval \
        --metadata data/metadata/raw_mcq_v1.jsonl \
        --output-dir "${output_dir}" \
        --model-name-or-path "${model_name}" \
        --adapter-path "${adapter_path}" \
        --eval-split dev \
        --min-status raw_gt \
        --experiment-id "${experiment_id}" \
        --trust-remote-code
  else
    run_step "${marker}" "${desc}" \
      "${PYTHON_BIN}" -m term_ai.experiment.lm_eval \
        --metadata data/metadata/raw_mcq_v1.jsonl \
        --output-dir "${output_dir}" \
        --model-name-or-path "${model_name}" \
        --adapter-path "${adapter_path}" \
        --eval-split dev \
        --min-status raw_gt \
        --experiment-id "${experiment_id}"
  fi
}

read_h1_policy() {
  if [[ -n "${H1_LOW_CONFIDENCE_THRESHOLD}" && -n "${H1_HIGH_CONFIDENCE_THRESHOLD}" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" && ! -f "runs/H1_dev/hybrid_policy_tuning.json" ]]; then
    H1_LOW_CONFIDENCE_THRESHOLD="0.4"
    H1_HIGH_CONFIDENCE_THRESHOLD="0.7"
    log "H1 selected policy unavailable in dry-run; using docs example low=0.4, high=0.7"
    return 0
  fi
  require_file "runs/H1_dev/hybrid_policy_tuning.json"
  local values
  values="$("${PYTHON_BIN}" -c 'import json; d=json.load(open("runs/H1_dev/hybrid_policy_tuning.json", encoding="utf-8")); p=d["selected_policy"]; print(p["low_confidence_threshold"], p["high_confidence_threshold"])')"
  read -r H1_LOW_CONFIDENCE_THRESHOLD H1_HIGH_CONFIDENCE_THRESHOLD <<<"${values}"
  log "H1 selected policy: low=${H1_LOW_CONFIDENCE_THRESHOLD}, high=${H1_HIGH_CONFIDENCE_THRESHOLD}"
}

run_final_baselines() {
  run_step "runs/B0_test_final/metric_log.json" "B0 final test" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=B0 \
      execution.output_dir=runs/B0_test_final \
      evaluation.split=test \
      execution.test_lock_dir="${TEST_LOCK_DIR}"

  run_step "runs/B1_test_final/metric_log.json" "B1 final test" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=B1 \
      execution.output_dir=runs/B1_test_final \
      evaluation.split=test \
      execution.test_lock_dir="${TEST_LOCK_DIR}"

  run_step "runs/B2_test_final/metric_log.json" "B2 final test" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=B2 \
      execution.output_dir=runs/B2_test_final \
      evaluation.split=test \
      execution.test_lock_dir="${TEST_LOCK_DIR}"

  run_step "runs/B3_test_final/metric_log.json" "B3 final test" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=B3 \
      execution.output_dir=runs/B3_test_final \
      evaluation.split=test \
      execution.reranker.fine_tune=false \
      execution.test_lock_dir="${TEST_LOCK_DIR}"

  run_step "runs/B4_test_final/metric_log.json" "B4 final test API recheck" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=B4 \
      execution.output_dir=runs/B4_test_final \
      evaluation.split=test \
      execution.primary_predictions=runs/B0_test_final/prediction_log.jsonl \
      execution.api_recheck.provider="${B4_PROVIDER}" \
      execution.api_recheck.model="${B4_MODEL}" \
      execution.api_recheck.env_path="${ENV_PATH}" \
      execution.api_recheck.requests_per_second="${API_RECHECK_RPS}" \
      execution.api_recheck.confidence_threshold="${API_RECHECK_CONFIDENCE_THRESHOLD}" \
      execution.test_lock_dir="${TEST_LOCK_DIR}"
}

run_final_zero_shot() {
  run_step "runs/G0_Gemma_test_final/metric_log.json" "G0 Gemma final test" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=G0-Gemma \
      execution.output_dir=runs/G0_Gemma_test_final \
      execution.model_name_or_path=google/gemma-2-2b-it \
      evaluation.split=test \
      execution.test_lock_dir="${TEST_LOCK_DIR}"

  run_step "runs/G0_Qwen_test_final/metric_log.json" "G0 Qwen final test" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=G0-Qwen \
      execution.output_dir=runs/G0_Qwen_test_final \
      execution.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
      evaluation.split=test \
      execution.test_lock_dir="${TEST_LOCK_DIR}"

  if [[ "${INCLUDE_BITNET}" == "1" ]]; then
    run_step "runs/G0_BitNet_test_final/metric_log.json" "G0 BitNet final test" \
      "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
        execution.run=true \
        model.experiment_id=G0-BitNet \
        execution.output_dir=runs/G0_BitNet_test_final \
        execution.model_name_or_path="${BITNET_MODEL_ID}" \
        evaluation.split=test \
        execution.test_lock_dir="${TEST_LOCK_DIR}"
  fi
}

run_lm_eval_adapter() {
  local marker="$1"
  local desc="$2"
  local output_dir="$3"
  local model_name="$4"
  local adapter_path="$5"
  local experiment_id="$6"
  local trust_remote="${7:-false}"
  require_dir "${adapter_path}"
  if [[ "${trust_remote}" == "true" ]]; then
    run_step "${marker}" "${desc}" \
      "${PYTHON_BIN}" -m term_ai.experiment.lm_eval \
        --metadata data/metadata/raw_mcq_v1.jsonl \
        --output-dir "${output_dir}" \
        --model-name-or-path "${model_name}" \
        --adapter-path "${adapter_path}" \
        --eval-split test \
        --min-status raw_gt \
        --experiment-id "${experiment_id}" \
        --test-lock-dir "${TEST_LOCK_DIR}" \
        --trust-remote-code
  else
    run_step "${marker}" "${desc}" \
      "${PYTHON_BIN}" -m term_ai.experiment.lm_eval \
        --metadata data/metadata/raw_mcq_v1.jsonl \
        --output-dir "${output_dir}" \
        --model-name-or-path "${model_name}" \
        --adapter-path "${adapter_path}" \
        --eval-split test \
        --min-status raw_gt \
        --experiment-id "${experiment_id}" \
        --test-lock-dir "${TEST_LOCK_DIR}"
  fi
}

run_final_adapters() {
  run_lm_eval_adapter "runs/G1_Gemma_test_final/metric_log.json" "G1 Gemma adapter final test" \
    runs/G1_Gemma_test_final google/gemma-2-2b-it runs/G1_Gemma_dev/final_adapter G1-Gemma
  run_lm_eval_adapter "runs/G1_Qwen_test_final/metric_log.json" "G1 Qwen adapter final test" \
    runs/G1_Qwen_test_final Qwen/Qwen2.5-3B-Instruct runs/G1_Qwen_dev/final_adapter G1-Qwen
  run_lm_eval_adapter "runs/G2_Gemma_test_final/metric_log.json" "G2 Gemma adapter final test" \
    runs/G2_Gemma_test_final google/gemma-2-2b-it runs/G2_Gemma_dev/final_adapter G2-Gemma
  run_lm_eval_adapter "runs/G2_Qwen_test_final/metric_log.json" "G2 Qwen adapter final test" \
    runs/G2_Qwen_test_final Qwen/Qwen2.5-3B-Instruct runs/G2_Qwen_dev/final_adapter G2-Qwen
  run_lm_eval_adapter "runs/G3_Gemma_test_final/metric_log.json" "G3 Gemma adapter final test" \
    runs/G3_Gemma_test_final google/gemma-2-2b-it runs/G3_Gemma_dev/final_adapter G3-Gemma
  run_lm_eval_adapter "runs/G3_Qwen_test_final/metric_log.json" "G3 Qwen adapter final test" \
    runs/G3_Qwen_test_final Qwen/Qwen2.5-3B-Instruct runs/G3_Qwen_dev/final_adapter G3-Qwen
  if [[ "${INCLUDE_BITNET}" == "1" ]]; then
    run_lm_eval_adapter "runs/G1_BitNet_test_final/metric_log.json" "G1 BitNet adapter final test" \
      runs/G1_BitNet_test_final "${BITNET_MODEL_ID}" runs/G1_BitNet_dev/final_adapter G1-BitNet
    run_lm_eval_adapter "runs/G2_BitNet_test_final/metric_log.json" "G2 BitNet adapter final test" \
      runs/G2_BitNet_test_final "${BITNET_MODEL_ID}" runs/G2_BitNet_dev/final_adapter G2-BitNet
    run_lm_eval_adapter "runs/G3_BitNet_test_final/metric_log.json" "G3 BitNet adapter final test" \
      runs/G3_BitNet_test_final "${BITNET_MODEL_ID}" runs/G3_BitNet_dev/final_adapter G3-BitNet
  fi
}

run_g4_final_one() {
  local family="$1"
  local allow_repeat="${2:-false}"
  local output_dir model_name adapter_path marker
  case "${family}" in
    gemma)
      output_dir="runs/G4_Gemma_test_final"
      model_name="google/gemma-2-2b-it"
      adapter_path="runs/G3_Gemma_dev/final_adapter"
      ;;
    qwen)
      output_dir="runs/G4_Qwen_test_final"
      model_name="Qwen/Qwen2.5-3B-Instruct"
      adapter_path="runs/G3_Qwen_dev/final_adapter"
      ;;
    bitnet)
      output_dir="runs/G4_BitNet_test_final"
      model_name="${BITNET_MODEL_ID}"
      adapter_path="runs/G3_BitNet_dev/final_adapter"
      ;;
    *)
      die "unknown G4 family: ${family}"
      ;;
  esac
  if [[ "${family}" == "bitnet" && "${RUN_BITNET_G4}" != "1" ]]; then
    log "skip: G4 bitnet final quantization comparison (RUN_BITNET_G4=${RUN_BITNET_G4})"
    return 0
  fi
  require_dir "${adapter_path}"
  marker="${output_dir}/quantization_compare.json"
  if [[ "${allow_repeat}" == "true" && "${family}" == "bitnet" ]]; then
    run_step "${marker}" "G4 ${family} final quantization comparison with repeat test lock override" \
      "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
        execution.run=true \
        model.experiment_id=G4-8bit \
        execution.output_dir="${output_dir}" \
        execution.model_name_or_path="${model_name}" \
        execution.adapter_path="${adapter_path}" \
        evaluation.split=test \
        execution.test_lock_dir="${TEST_LOCK_DIR}" \
        execution.allow_repeat_test=true \
        execution.local_cost_per_hour_usd="${LOCAL_COST_PER_HOUR_USD}"
  elif [[ "${family}" == "bitnet" ]]; then
    run_step "${marker}" "G4 ${family} final quantization comparison" \
      "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
        execution.run=true \
        model.experiment_id=G4-8bit \
        execution.output_dir="${output_dir}" \
        execution.model_name_or_path="${model_name}" \
        execution.adapter_path="${adapter_path}" \
        evaluation.split=test \
        execution.test_lock_dir="${TEST_LOCK_DIR}" \
        execution.local_cost_per_hour_usd="${LOCAL_COST_PER_HOUR_USD}"
  elif [[ "${allow_repeat}" == "true" ]]; then
    run_step "${marker}" "G4 ${family} final quantization comparison with repeat test lock override" \
      "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
        execution.run=true \
        model.experiment_id=G4-8bit \
        execution.output_dir="${output_dir}" \
        execution.model_name_or_path="${model_name}" \
        execution.adapter_path="${adapter_path}" \
        evaluation.split=test \
        execution.test_lock_dir="${TEST_LOCK_DIR}" \
        execution.allow_repeat_test=true \
        execution.local_cost_per_hour_usd="${LOCAL_COST_PER_HOUR_USD}"
  else
    run_step "${marker}" "G4 ${family} final quantization comparison" \
      "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
        execution.run=true \
        model.experiment_id=G4-8bit \
        execution.output_dir="${output_dir}" \
        execution.model_name_or_path="${model_name}" \
        execution.adapter_path="${adapter_path}" \
        evaluation.split=test \
        execution.test_lock_dir="${TEST_LOCK_DIR}" \
        execution.local_cost_per_hour_usd="${LOCAL_COST_PER_HOUR_USD}"
  fi
}

run_final_g4() {
  case "${G4_FINAL_MODEL}" in
    qwen)
      run_g4_final_one qwen false
      ;;
    gemma)
      run_g4_final_one gemma false
      ;;
    bitnet)
      run_g4_final_one bitnet false
      ;;
    both)
      run_g4_final_one qwen false
      run_g4_final_one gemma true
      ;;
    all)
      run_g4_final_one qwen false
      run_g4_final_one gemma true
      if [[ "${INCLUDE_BITNET}" == "1" ]]; then
        run_g4_final_one bitnet true
      fi
      ;;
    *)
      die "G4_FINAL_MODEL must be qwen, gemma, bitnet, both, or all"
      ;;
  esac
}

run_final_e1_h1() {
  ensure_kd_views
  run_step "runs/E1_test_final/metric_log.json" "E1 final test" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=E1 \
      execution.output_dir=runs/E1_test_final \
      execution.kd_metadata=data/metadata/kd_all_view_v1.jsonl \
      execution.min_status=any \
      evaluation.split=test \
      training.kd.require_teacher_scores=true \
      execution.test_lock_dir="${TEST_LOCK_DIR}"

  read_h1_policy
  run_step "runs/H1_test_final/metric_log.json" "H1 final test fixed dev-selected policy" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=H1 \
      execution.output_dir=runs/H1_test_final \
      evaluation.split=test \
      execution.primary_predictions=runs/B0_test_final/prediction_log.jsonl \
      execution.cross_encoder_predictions=runs/B3_test_final/prediction_log.jsonl \
      execution.fallback_predictions=runs/B4_test_final/prediction_log.jsonl \
      execution.hybrid.tune_policy=false \
      execution.hybrid.low_confidence_threshold="${H1_LOW_CONFIDENCE_THRESHOLD}" \
      execution.hybrid.high_confidence_threshold="${H1_HIGH_CONFIDENCE_THRESHOLD}" \
      execution.test_lock_dir="${TEST_LOCK_DIR}"
}

run_final_tests() {
  ensure_common_inputs
  run_final_baselines
  run_final_zero_shot
  run_final_adapters
  run_final_g4
  run_final_e1_h1
}

run_stats() {
  run_step "reports/B3_vs_G3_Qwen_test_statistics.json" "statistics B3 vs G3 Qwen final test" \
    "${PYTHON_BIN}" -m term_ai.experiment.statistics \
      --predictions-a runs/B3_test_final/prediction_log.jsonl \
      --predictions-b runs/G3_Qwen_test_final/prediction_log.jsonl \
      --output reports/B3_vs_G3_Qwen_test_statistics.json \
      --samples "${STAT_SAMPLES}"

  run_step "reports/B3_vs_G3_Gemma_test_statistics.json" "statistics B3 vs G3 Gemma final test" \
    "${PYTHON_BIN}" -m term_ai.experiment.statistics \
      --predictions-a runs/B3_test_final/prediction_log.jsonl \
      --predictions-b runs/G3_Gemma_test_final/prediction_log.jsonl \
      --output reports/B3_vs_G3_Gemma_test_statistics.json \
      --samples "${STAT_SAMPLES}"

  if [[ -f "runs/G3_BitNet_test_final/prediction_log.jsonl" || "${DRY_RUN}" == "1" ]]; then
    run_step "reports/B3_vs_G3_BitNet_test_statistics.json" "statistics B3 vs G3 BitNet final test" \
      "${PYTHON_BIN}" -m term_ai.experiment.statistics \
        --predictions-a runs/B3_test_final/prediction_log.jsonl \
        --predictions-b runs/G3_BitNet_test_final/prediction_log.jsonl \
        --output reports/B3_vs_G3_BitNet_test_statistics.json \
        --samples "${STAT_SAMPLES}"
  fi

  run_step "reports/B0_vs_H1_test_statistics.json" "statistics B0 vs H1 final test" \
    "${PYTHON_BIN}" -m term_ai.experiment.statistics \
      --predictions-a runs/B0_test_final/prediction_log.jsonl \
      --predictions-b runs/H1_test_final/prediction_log.jsonl \
      --output reports/B0_vs_H1_test_statistics.json \
      --samples "${STAT_SAMPLES}"

  if [[ -f "runs/G4_Qwen_test_final/4bit/prediction_log.jsonl" ]]; then
    run_step "reports/G3_Qwen_vs_G4_Qwen_4bit_test_statistics.json" "statistics G3 Qwen vs G4 Qwen 4bit final test" \
      "${PYTHON_BIN}" -m term_ai.experiment.statistics \
        --predictions-a runs/G3_Qwen_test_final/prediction_log.jsonl \
        --predictions-b runs/G4_Qwen_test_final/4bit/prediction_log.jsonl \
        --output reports/G3_Qwen_vs_G4_Qwen_4bit_test_statistics.json \
        --samples "${STAT_SAMPLES}"
  fi

  if [[ -f "runs/G4_Gemma_test_final/4bit/prediction_log.jsonl" ]]; then
    run_step "reports/G3_Gemma_vs_G4_Gemma_4bit_test_statistics.json" "statistics G3 Gemma vs G4 Gemma 4bit final test" \
      "${PYTHON_BIN}" -m term_ai.experiment.statistics \
        --predictions-a runs/G3_Gemma_test_final/prediction_log.jsonl \
        --predictions-b runs/G4_Gemma_test_final/4bit/prediction_log.jsonl \
        --output reports/G3_Gemma_vs_G4_Gemma_4bit_test_statistics.json \
        --samples "${STAT_SAMPLES}"
  fi

  if [[ -f "runs/G4_BitNet_test_final/4bit/prediction_log.jsonl" ]]; then
    run_step "reports/G3_BitNet_vs_G4_BitNet_4bit_test_statistics.json" "statistics G3 BitNet vs G4 BitNet 4bit final test" \
      "${PYTHON_BIN}" -m term_ai.experiment.statistics \
        --predictions-a runs/G3_BitNet_test_final/prediction_log.jsonl \
        --predictions-b runs/G4_BitNet_test_final/4bit/prediction_log.jsonl \
        --output reports/G3_BitNet_vs_G4_BitNet_4bit_test_statistics.json \
        --samples "${STAT_SAMPLES}"
  fi
}

run_report() {
  run_step "reports/final_experiment_report.md" "final report input generation" \
    "${PYTHON_BIN}" -m term_ai.experiment.reporting \
      --runs-dir runs \
      --output-dir reports
}

case "${MODE}" in
  dev|dev_remaining)
    run_dev_remaining
    ;;
  final|final_test)
    run_final_tests
    ;;
  stats)
    run_stats
    ;;
  report|reports)
    run_report
    ;;
  all)
    run_dev_remaining
    run_final_tests
    run_stats
    run_report
    ;;
  *)
    usage
    die "unknown mode: ${MODE}"
    ;;
esac

log "done: ${MODE}"
