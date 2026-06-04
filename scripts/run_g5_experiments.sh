#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run G5 3B-teacher -> 0.5B/1.5B student compression experiments.

Usage:
  scripts/run_g5_experiments.sh [download|teacher|dev|final|stats|all]

Defaults:
  mode: all
  PYTHON_BIN=python
  FORCE=0
  DRY_RUN=0
  TEST_LOCK_DIR=runs/_test_locks
  LOCAL_COST_PER_HOUR_USD=0
  STAT_SAMPLES=1000
  G5_STUDENTS="Qwen0p5 Qwen1p5"
  G5_TEMPERATURES="1 2 4"
  G5_TEACHER_MODEL=Qwen/Qwen2.5-3B-Instruct
  G5_TEACHER_ADAPTER=runs/G3_Qwen_dev/final_adapter
  G5_LAMBDA_SOFT=0.5
  G5_DROP_RATIONALE=1
  G5_DOWNLOAD_IF_MISSING=1

Recommended local env:
  PYTHON_BIN=/home/swfool/anaconda3/envs/term_toeic/bin/python scripts/run_g5_experiments.sh all
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
TEST_LOCK_DIR="${TEST_LOCK_DIR:-runs/_test_locks}"
LOCAL_COST_PER_HOUR_USD="${LOCAL_COST_PER_HOUR_USD:-0}"
STAT_SAMPLES="${STAT_SAMPLES:-1000}"
G5_STUDENTS="${G5_STUDENTS:-Qwen0p5 Qwen1p5}"
G5_TEMPERATURES="${G5_TEMPERATURES:-1 2 4}"
G5_TEACHER_MODEL="${G5_TEACHER_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
G5_TEACHER_ADAPTER="${G5_TEACHER_ADAPTER:-runs/G3_Qwen_dev/final_adapter}"
G5_LAMBDA_SOFT="${G5_LAMBDA_SOFT:-0.5}"
G5_DROP_RATIONALE="${G5_DROP_RATIONALE:-1}"
G5_DOWNLOAD_IF_MISSING="${G5_DOWNLOAD_IF_MISSING:-1}"

log() {
  printf '[g5-exp] %s\n' "$*"
}

die() {
  printf '[g5-exp] ERROR: %s\n' "$*" >&2
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

student_model() {
  case "$1" in
    Qwen0p5) printf 'Qwen/Qwen2.5-0.5B-Instruct' ;;
    Qwen1p5) printf 'Qwen/Qwen2.5-1.5B-Instruct' ;;
    *) die "unknown G5 student: $1" ;;
  esac
}

run_name() {
  local experiment_id="$1"
  local suffix="$2"
  printf 'runs/%s_%s' "${experiment_id//-/_}" "${suffix}"
}

teacher_metadata_path() {
  local split="$1"
  local temperature="$2"
  printf 'data/metadata/g5_qwen3b_teacher_kd_%s_t%s_v1.jsonl' "${split}" "${temperature}"
}

kd_rationale_override() {
  if [[ "${G5_DROP_RATIONALE}" == "1" ]]; then
    printf 'training.kd.include_rationale=false'
  else
    printf 'training.kd.include_rationale=true'
  fi
}

ensure_g5_inputs() {
  require_file "data/metadata/raw_mcq_v1.jsonl"
  require_file "data/metadata/kd_train_view_v1.jsonl"
  require_file "data/metadata/kd_dev_view_v1.jsonl"
  require_file "data/sft/raw_train_sft_v1.jsonl"
  require_file "data/sft/raw_dev_sft_v1.jsonl"
  require_file "data/sft/raw_judge_aug_train_sft_v1.jsonl"
  require_file "data/sft/raw_judge_aug_dev_sft_v1.jsonl"
  require_dir "${G5_TEACHER_ADAPTER}"
}

run_download() {
  local args=(
    "${PYTHON_BIN}" -m term_ai.experiment.model_download
    --model-id "${G5_TEACHER_MODEL}"
    --model-id Qwen/Qwen2.5-0.5B-Instruct
    --model-id Qwen/Qwen2.5-1.5B-Instruct
    --output runs/G5_model_download.json
  )
  if [[ "${G5_DOWNLOAD_IF_MISSING}" == "1" ]]; then
    args+=(--download-if-missing)
  fi
  run_step "runs/G5_model_download.json" "G5 model cache verification/download" "${args[@]}"
}

run_teacher_logits() {
  ensure_g5_inputs
  local temp
  for temp in ${G5_TEMPERATURES}; do
    run_step "$(teacher_metadata_path train "${temp}")" "G5 train teacher logits T=${temp}" \
      "${PYTHON_BIN}" -m term_ai.experiment.g5_teacher_logits \
        --metadata-jsonl data/metadata/kd_train_view_v1.jsonl \
        --output "$(teacher_metadata_path train "${temp}")" \
        --model-name-or-path "${G5_TEACHER_MODEL}" \
        --adapter-path "${G5_TEACHER_ADAPTER}" \
        --min-status any \
        --temperature "${temp}"

    run_step "$(teacher_metadata_path dev "${temp}")" "G5 dev teacher logits T=${temp}" \
      "${PYTHON_BIN}" -m term_ai.experiment.g5_teacher_logits \
        --metadata-jsonl data/metadata/kd_dev_view_v1.jsonl \
        --output "$(teacher_metadata_path dev "${temp}")" \
        --model-name-or-path "${G5_TEACHER_MODEL}" \
        --adapter-path "${G5_TEACHER_ADAPTER}" \
        --min-status any \
        --temperature "${temp}"
  done
}

run_g5_hydra_dev() {
  local experiment_id="$1"
  local output_dir="$2"
  local model_id="$3"
  shift 3
  run_step "${output_dir}/metric_log.json" "G5 dev ${experiment_id}" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id="${experiment_id}" \
      execution.output_dir="${output_dir}" \
      execution.model_name_or_path="${model_id}" \
      evaluation.split=dev \
      execution.local_cost_per_hour_usd="${LOCAL_COST_PER_HOUR_USD}" \
      "$@"
}

run_g5_train_dev() {
  local experiment_id="$1"
  local output_dir="$2"
  local model_id="$3"
  shift 3
  run_step "${output_dir}/final_adapter/adapter_config.json" "G5 train ${experiment_id}" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id="${experiment_id}" \
      execution.output_dir="${output_dir}" \
      execution.model_name_or_path="${model_id}" \
      evaluation.split=dev \
      training.save_total_limit=3 \
      "$@"
}

run_g5_sft_dev() {
  local experiment_id="$1"
  local output_dir="$2"
  local model_id="$3"
  shift 3
  run_step "${output_dir}/post_train_eval/metric_log.json" "G5 train/eval ${experiment_id}" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id="${experiment_id}" \
      execution.output_dir="${output_dir}" \
      execution.model_name_or_path="${model_id}" \
      evaluation.split=dev \
      training.save_total_limit=3 \
      "$@"
}

run_lm_eval_adapter_dev() {
  local experiment_id="$1"
  local output_dir="$2"
  local model_id="$3"
  local adapter_path="$4"
  require_dir "${adapter_path}"
  run_step "${output_dir}/post_train_eval/metric_log.json" "G5 post-train dev eval ${experiment_id}" \
    "${PYTHON_BIN}" -m term_ai.experiment.lm_eval \
      --metadata data/metadata/raw_mcq_v1.jsonl \
      --output-dir "${output_dir}/post_train_eval" \
      --model-name-or-path "${model_id}" \
      --adapter-path "${adapter_path}" \
      --eval-split dev \
      --min-status raw_gt \
      --experiment-id "${experiment_id}"
}

run_dev() {
  ensure_g5_inputs
  local student model_id output_dir experiment_id temp
  for student in ${G5_STUDENTS}; do
    model_id="$(student_model "${student}")"

    experiment_id="G5-${student}-ZS"
    output_dir="$(run_name "${experiment_id}" dev)"
    run_g5_hydra_dev "${experiment_id}" "${output_dir}" "${model_id}"

    experiment_id="G5-${student}-G1"
    output_dir="$(run_name "${experiment_id}" dev)"
    run_g5_sft_dev "${experiment_id}" "${output_dir}" "${model_id}"

    experiment_id="G5-${student}-G2"
    output_dir="$(run_name "${experiment_id}" dev)"
    run_g5_sft_dev "${experiment_id}" "${output_dir}" "${model_id}"

    experiment_id="G5-${student}-GPTKD"
    output_dir="$(run_name "${experiment_id}" dev)"
    run_g5_train_dev "${experiment_id}" "${output_dir}" "${model_id}" \
      execution.kd_metadata=data/metadata/kd_train_view_v1.jsonl \
      execution.kd_dev_metadata=data/metadata/kd_dev_view_v1.jsonl \
      execution.min_status=any \
      training.kd.lambda_soft="${G5_LAMBDA_SOFT}" \
      "$(kd_rationale_override)" \
      training.kd.require_teacher_scores=true
    run_lm_eval_adapter_dev "${experiment_id}" "${output_dir}" "${model_id}" "${output_dir}/final_adapter"

    for temp in ${G5_TEMPERATURES}; do
      require_file "$(teacher_metadata_path train "${temp}")"
      require_file "$(teacher_metadata_path dev "${temp}")"
      experiment_id="G5-${student}-3BKD-T${temp}"
      output_dir="$(run_name "${experiment_id}" dev)"
      run_g5_train_dev "${experiment_id}" "${output_dir}" "${model_id}" \
        execution.min_status=any \
        training.kd.lambda_soft="${G5_LAMBDA_SOFT}" \
        "$(kd_rationale_override)" \
        training.kd.require_teacher_scores=true
      run_lm_eval_adapter_dev "${experiment_id}" "${output_dir}" "${model_id}" "${output_dir}/final_adapter"
    done
  done
}

run_lm_eval_final_adapter() {
  local experiment_id="$1"
  local dev_dir="$2"
  local final_dir="$3"
  local model_id="$4"
  require_dir "${dev_dir}/final_adapter"
  run_step "${final_dir}/metric_log.json" "G5 final test ${experiment_id}" \
    "${PYTHON_BIN}" -m term_ai.experiment.lm_eval \
      --metadata data/metadata/raw_mcq_v1.jsonl \
      --output-dir "${final_dir}" \
      --model-name-or-path "${model_id}" \
      --adapter-path "${dev_dir}/final_adapter" \
      --eval-split test \
      --min-status raw_gt \
      --experiment-id "${experiment_id}" \
      --test-lock-dir "${TEST_LOCK_DIR}" \
      --local-cost-per-hour-usd "${LOCAL_COST_PER_HOUR_USD}"
}

run_final() {
  ensure_g5_inputs
  local student model_id experiment_id dev_dir final_dir temp
  for student in ${G5_STUDENTS}; do
    model_id="$(student_model "${student}")"

    experiment_id="G5-${student}-ZS"
    final_dir="$(run_name "${experiment_id}" test_final)"
    run_step "${final_dir}/metric_log.json" "G5 final test ${experiment_id}" \
      "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
        execution.run=true \
        model.experiment_id="${experiment_id}" \
        execution.output_dir="${final_dir}" \
        execution.model_name_or_path="${model_id}" \
        evaluation.split=test \
        execution.test_lock_dir="${TEST_LOCK_DIR}" \
        execution.local_cost_per_hour_usd="${LOCAL_COST_PER_HOUR_USD}"

    for experiment_id in "G5-${student}-G1" "G5-${student}-G2" "G5-${student}-GPTKD"; do
      dev_dir="$(run_name "${experiment_id}" dev)"
      final_dir="$(run_name "${experiment_id}" test_final)"
      run_lm_eval_final_adapter "${experiment_id}" "${dev_dir}" "${final_dir}" "${model_id}"
    done

    for temp in ${G5_TEMPERATURES}; do
      experiment_id="G5-${student}-3BKD-T${temp}"
      dev_dir="$(run_name "${experiment_id}" dev)"
      final_dir="$(run_name "${experiment_id}" test_final)"
      run_lm_eval_final_adapter "${experiment_id}" "${dev_dir}" "${final_dir}" "${model_id}"
    done
  done
}

run_stats_one() {
  local experiment_id="$1"
  local final_dir="$2"
  local prediction_log="${final_dir}/prediction_log.jsonl"
  if [[ ! -f "${prediction_log}" && "${DRY_RUN}" != "1" ]]; then
    log "skip stats: missing ${prediction_log}"
    return 0
  fi
  if [[ -f "runs/B3_test_final/prediction_log.jsonl" || "${DRY_RUN}" == "1" ]]; then
    run_step "reports/B3_vs_${experiment_id//-/_}_test_statistics.json" "statistics B3 vs ${experiment_id}" \
      "${PYTHON_BIN}" -m term_ai.experiment.statistics \
        --predictions-a runs/B3_test_final/prediction_log.jsonl \
        --predictions-b "${prediction_log}" \
        --output "reports/B3_vs_${experiment_id//-/_}_test_statistics.json" \
        --samples "${STAT_SAMPLES}"
  fi
  if [[ -f "runs/G4_Qwen_test_final/4bit/prediction_log.jsonl" || "${DRY_RUN}" == "1" ]]; then
    run_step "reports/G4_Qwen_4bit_vs_${experiment_id//-/_}_test_statistics.json" "statistics G4 Qwen 4bit vs ${experiment_id}" \
      "${PYTHON_BIN}" -m term_ai.experiment.statistics \
        --predictions-a runs/G4_Qwen_test_final/4bit/prediction_log.jsonl \
        --predictions-b "${prediction_log}" \
        --output "reports/G4_Qwen_4bit_vs_${experiment_id//-/_}_test_statistics.json" \
        --samples "${STAT_SAMPLES}"
  fi
}

run_stats() {
  local student experiment_id final_dir temp
  for student in ${G5_STUDENTS}; do
    for experiment_id in "G5-${student}-ZS" "G5-${student}-G1" "G5-${student}-G2" "G5-${student}-GPTKD"; do
      final_dir="$(run_name "${experiment_id}" test_final)"
      run_stats_one "${experiment_id}" "${final_dir}"
    done
    for temp in ${G5_TEMPERATURES}; do
      experiment_id="G5-${student}-3BKD-T${temp}"
      final_dir="$(run_name "${experiment_id}" test_final)"
      run_stats_one "${experiment_id}" "${final_dir}"
    done
  done
}

case "${MODE}" in
  download)
    run_download
    ;;
  teacher)
    run_teacher_logits
    ;;
  dev)
    run_dev
    ;;
  final|final_test)
    run_final
    ;;
  stats)
    run_stats
    ;;
  all)
    run_download
    run_teacher_logits
    run_dev
    run_final
    run_stats
    ;;
  *)
    usage
    die "unknown mode: ${MODE}"
    ;;
esac

log "done: ${MODE}"
