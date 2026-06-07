#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run no-API, no-retrain local evaluation checks using existing checkpoints/adapters.

Usage:
  scripts/run_no_api_no_retrain_check.sh [prepare|quick|full|hybrid|stats|all]

Defaults:
  mode: full
  PYTHON_BIN=python
  SOURCE_METADATA=data/metadata/kd_all_view_v1.jsonl
  EVAL_SPLIT=test
  MIN_STATUS=any
  LIMIT=500
  OUT_ROOT=runs/no_api_no_retrain_check
  SEED=42
  FORCE=0
  DRY_RUN=0
  LOCAL_COST_PER_HOUR_USD=0
  STAT_SAMPLES=1000
  HYBRID_FALLBACK=G5_Qwen0p5_G1
  HYBRID_THRESHOLD_GRID=[0.75,0.8,0.85,0.9,0.95,0.99,1.01]

Modes:
  prepare  Build the local check metadata only.
  quick    B0, B3, G5_Qwen0p5_G1, G5_Qwen1p5_G1.
  full     quick + B2, G3_Qwen, G4_Qwen_4bit. Default.
  hybrid   Local-only H1 sweep using B0/B3 and HYBRID_FALLBACK predictions.
  stats    Paired statistics for selected prediction logs.
  all      full + hybrid + stats.

This script intentionally does not run B4/API, E1 KD scorer training, LoRA/KD
training, G5 teacher-logit generation, or raw teacher score generation.
EOF
}

MODE="${1:-full}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_METADATA="${SOURCE_METADATA:-data/metadata/kd_all_view_v1.jsonl}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MIN_STATUS="${MIN_STATUS:-any}"
LIMIT="${LIMIT:-500}"
OUT_ROOT="${OUT_ROOT:-runs/no_api_no_retrain_check}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
LOCAL_COST_PER_HOUR_USD="${LOCAL_COST_PER_HOUR_USD:-0}"
STAT_SAMPLES="${STAT_SAMPLES:-1000}"
PROGRESS_INTERVAL_ITEMS="${PROGRESS_INTERVAL_ITEMS:-1}"
TEST_LOCK_DIR="${TEST_LOCK_DIR:-runs/_test_locks}"
HYBRID_FALLBACK="${HYBRID_FALLBACK:-G5_Qwen0p5_G1}"
HYBRID_THRESHOLD_GRID="${HYBRID_THRESHOLD_GRID:-[0.75,0.8,0.85,0.9,0.95,0.99,1.01]}"

B2_SCORER_SOURCE="${B2_SCORER_SOURCE:-runs/B2_dev/mlp_scorer.pkl}"
QWEN3B_MODEL="${QWEN3B_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
QWEN0P5_MODEL="${QWEN0P5_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
QWEN1P5_MODEL="${QWEN1P5_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
G3_QWEN_ADAPTER="${G3_QWEN_ADAPTER:-runs/G3_Qwen_dev/final_adapter}"
G5_QWEN0P5_G1_ADAPTER="${G5_QWEN0P5_G1_ADAPTER:-runs/G5_Qwen0p5_G1_dev/final_adapter}"
G5_QWEN1P5_G1_ADAPTER="${G5_QWEN1P5_G1_ADAPTER:-runs/G5_Qwen1p5_G1_dev/final_adapter}"

PREPARED_METADATA="${OUT_ROOT}/_inputs/eval_metadata.jsonl"
REPORT_DIR="${OUT_ROOT}/reports"

log() {
  printf '[no-api-no-retrain] %s\n' "$*"
}

die() {
  printf '[no-api-no-retrain] ERROR: %s\n' "$*" >&2
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

prepare_metadata() {
  require_file "${SOURCE_METADATA}"
  if [[ "${FORCE}" != "1" && -f "${PREPARED_METADATA}" ]]; then
    log "skip: prepare metadata (${PREPARED_METADATA} exists)"
    return 0
  fi
  log "run: prepare metadata"
  print_cmd "${PYTHON_BIN}" - "${SOURCE_METADATA}" "${PREPARED_METADATA}" "${EVAL_SPLIT}" "${LIMIT}" "${SEED}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" - "${SOURCE_METADATA}" "${PREPARED_METADATA}" "${EVAL_SPLIT}" "${LIMIT}" "${SEED}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import random
import sys

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
eval_split = sys.argv[3]
limit = int(sys.argv[4])
seed = int(sys.argv[5])

rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
prefix_rows = [row for row in rows if row.get("split") != eval_split]
test_rows = [row for row in rows if row.get("split") == eval_split]

def is_raw(row: dict) -> bool:
    payload = row.get("payload") or {}
    return (
        row.get("status") == "raw_gt"
        or row.get("source") == "raw_gt"
        or payload.get("source_task_type") == "Raw Meaning Selection"
    )

generated = [row for row in test_rows if not is_raw(row)]
raw = [row for row in test_rows if is_raw(row)]
rng = random.Random(seed)

if limit <= 0:
    selected_test = generated + raw
elif len(generated) >= limit:
    selected_test = rng.sample(generated, limit)
else:
    selected_test = list(generated)
    remaining = limit - len(selected_test)
    selected_test.extend(rng.sample(raw, min(remaining, len(raw))))

rng.shuffle(selected_test)
output_rows = prefix_rows + selected_test
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8", newline="\n") as handle:
    for row in output_rows:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

summary = {
    "source": str(source_path),
    "output": str(output_path),
    "eval_split": eval_split,
    "limit": limit,
    "seed": seed,
    "prefix_rows": len(prefix_rows),
    "selected_test_rows": len(selected_test),
    "selected_generated": sum(1 for row in selected_test if not is_raw(row)),
    "selected_raw": sum(1 for row in selected_test if is_raw(row)),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

ensure_prepared_metadata() {
  if [[ ! -f "${PREPARED_METADATA}" ]]; then
    prepare_metadata
  fi
  require_file "${PREPARED_METADATA}"
}

run_b0() {
  ensure_prepared_metadata
  run_step "${OUT_ROOT}/B0/metric_log.json" "B0 check eval" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=B0 \
      execution.output_dir="${OUT_ROOT}/B0" \
      execution.metadata="${PREPARED_METADATA}" \
      execution.min_status="${MIN_STATUS}" \
      evaluation.split="${EVAL_SPLIT}" \
      execution.allow_repeat_test=true \
      execution.test_lock_dir="${TEST_LOCK_DIR}" \
      logging.progress_interval_items="${PROGRESS_INTERVAL_ITEMS}"
}

prepare_b2_scorer() {
  local output_dir="${OUT_ROOT}/B2"
  local target="${output_dir}/mlp_scorer.pkl"
  require_file "${B2_SCORER_SOURCE}"
  if [[ "${FORCE}" != "1" && -f "${target}" ]]; then
    log "skip: B2 scorer copy (${target} exists)"
    return 0
  fi
  log "run: B2 scorer copy"
  print_cmd cp "${B2_SCORER_SOURCE}" "${target}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  mkdir -p "${output_dir}"
  cp "${B2_SCORER_SOURCE}" "${target}"
}

run_b2() {
  ensure_prepared_metadata
  prepare_b2_scorer
  run_step "${OUT_ROOT}/B2/metric_log.json" "B2 check eval with existing scorer" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=B2 \
      execution.output_dir="${OUT_ROOT}/B2" \
      execution.resume=true \
      execution.metadata="${PREPARED_METADATA}" \
      execution.min_status="${MIN_STATUS}" \
      execution.train_metadata=data/metadata/raw_mcq_v1.jsonl \
      evaluation.split="${EVAL_SPLIT}" \
      execution.allow_repeat_test=true \
      execution.test_lock_dir="${TEST_LOCK_DIR}" \
      logging.progress_interval_items="${PROGRESS_INTERVAL_ITEMS}"
}

run_b3() {
  ensure_prepared_metadata
  run_step "${OUT_ROOT}/B3/metric_log.json" "B3 check eval without fine-tune" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=B3 \
      execution.output_dir="${OUT_ROOT}/B3" \
      execution.metadata="${PREPARED_METADATA}" \
      execution.min_status="${MIN_STATUS}" \
      evaluation.split="${EVAL_SPLIT}" \
      execution.reranker.fine_tune=false \
      execution.reranker.threshold=0.0 \
      execution.allow_repeat_test=true \
      execution.test_lock_dir="${TEST_LOCK_DIR}" \
      logging.progress_interval_items="${PROGRESS_INTERVAL_ITEMS}"
}

run_lm_adapter() {
  local run_name="$1"
  local model_id="$2"
  local adapter_path="$3"
  local experiment_id="$4"
  local quantization="${5:-}"
  ensure_prepared_metadata
  require_dir "${adapter_path}"
  local output_dir="${OUT_ROOT}/${run_name}"
  local marker="${output_dir}/metric_log.json"
  local args=(
    "${PYTHON_BIN}" -m term_ai.experiment.lm_eval
    --metadata "${PREPARED_METADATA}"
    --output-dir "${output_dir}"
    --model-name-or-path "${model_id}"
    --adapter-path "${adapter_path}"
    --eval-split "${EVAL_SPLIT}"
    --min-status "${MIN_STATUS}"
    --experiment-id "${experiment_id}"
    --allow-repeat-test
    --test-lock-dir "${TEST_LOCK_DIR}"
    --local-cost-per-hour-usd "${LOCAL_COST_PER_HOUR_USD}"
    --progress-interval-items "${PROGRESS_INTERVAL_ITEMS}"
  )
  if [[ -n "${quantization}" ]]; then
    args+=(--quantization "${quantization}")
  fi
  run_step "${marker}" "${run_name} lm_eval check" "${args[@]}"
}

run_g3_qwen() {
  run_lm_adapter "G3_Qwen" "${QWEN3B_MODEL}" "${G3_QWEN_ADAPTER}" "G3-Qwen"
}

run_g4_qwen_4bit() {
  run_lm_adapter "G4_Qwen_4bit" "${QWEN3B_MODEL}" "${G3_QWEN_ADAPTER}" "G4-Qwen-4bit" "4bit"
}

run_g5_qwen0p5_g1() {
  run_lm_adapter "G5_Qwen0p5_G1" "${QWEN0P5_MODEL}" "${G5_QWEN0P5_G1_ADAPTER}" "G5-Qwen0p5-G1"
}

run_g5_qwen1p5_g1() {
  run_lm_adapter "G5_Qwen1p5_G1" "${QWEN1P5_MODEL}" "${G5_QWEN1P5_G1_ADAPTER}" "G5-Qwen1p5-G1"
}

run_quick() {
  run_b0
  run_b3
  run_g5_qwen0p5_g1
  run_g5_qwen1p5_g1
}

run_full() {
  run_quick
  run_b2
  run_g3_qwen
  run_g4_qwen_4bit
}

fallback_prediction_log() {
  case "${HYBRID_FALLBACK}" in
    G5_Qwen0p5_G1) printf '%s/G5_Qwen0p5_G1/prediction_log.jsonl' "${OUT_ROOT}" ;;
    G5_Qwen1p5_G1) printf '%s/G5_Qwen1p5_G1/prediction_log.jsonl' "${OUT_ROOT}" ;;
    G3_Qwen) printf '%s/G3_Qwen/prediction_log.jsonl' "${OUT_ROOT}" ;;
    G4_Qwen_4bit) printf '%s/G4_Qwen_4bit/prediction_log.jsonl' "${OUT_ROOT}" ;;
    *) printf '%s' "${HYBRID_FALLBACK}" ;;
  esac
}

run_hybrid() {
  local fallback_log
  fallback_log="$(fallback_prediction_log)"
  require_file "${OUT_ROOT}/B0/prediction_log.jsonl"
  require_file "${OUT_ROOT}/B3/prediction_log.jsonl"
  require_file "${fallback_log}"
  run_step "${OUT_ROOT}/H1_local_hybrid/hybrid_policy_tuning.json" "H1 local-only hybrid sweep" \
    "${PYTHON_BIN}" -m term_ai.experiment.hydra_app \
      execution.run=true \
      model.experiment_id=H1 \
      execution.output_dir="${OUT_ROOT}/H1_local_hybrid" \
      execution.primary_predictions="${OUT_ROOT}/B0/prediction_log.jsonl" \
      execution.cross_encoder_predictions="${OUT_ROOT}/B3/prediction_log.jsonl" \
      execution.fallback_predictions="${fallback_log}" \
      execution.hybrid.tune_policy=true \
      "execution.hybrid.threshold_grid=${HYBRID_THRESHOLD_GRID}" \
      execution.hybrid.primary_cost_per_1000=0 \
      execution.hybrid.cross_encoder_cost_per_1000=0 \
      execution.hybrid.fallback_cost_per_1000=0 \
      logging.progress_interval_items="${PROGRESS_INTERVAL_ITEMS}"
}

stats_one() {
  local name="$1"
  local predictions_a="$2"
  local predictions_b="$3"
  local output="${REPORT_DIR}/${name}.json"
  if [[ ! -f "${predictions_a}" || ! -f "${predictions_b}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      log "dry-run: stats would require ${predictions_a} and ${predictions_b}"
    else
      log "skip stats ${name}: missing prediction log"
    fi
    return 0
  fi
  run_step "${output}" "statistics ${name}" \
    "${PYTHON_BIN}" -m term_ai.experiment.statistics \
      --predictions-a "${predictions_a}" \
      --predictions-b "${predictions_b}" \
      --output "${output}" \
      --samples "${STAT_SAMPLES}" \
      --seed "${SEED}"
}

run_stats() {
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${REPORT_DIR}"
  fi
  stats_one "B0_vs_B3" "${OUT_ROOT}/B0/prediction_log.jsonl" "${OUT_ROOT}/B3/prediction_log.jsonl"
  stats_one "B0_vs_G5_Qwen0p5_G1" "${OUT_ROOT}/B0/prediction_log.jsonl" "${OUT_ROOT}/G5_Qwen0p5_G1/prediction_log.jsonl"
  stats_one "B0_vs_G5_Qwen1p5_G1" "${OUT_ROOT}/B0/prediction_log.jsonl" "${OUT_ROOT}/G5_Qwen1p5_G1/prediction_log.jsonl"
  stats_one "B3_vs_G5_Qwen0p5_G1" "${OUT_ROOT}/B3/prediction_log.jsonl" "${OUT_ROOT}/G5_Qwen0p5_G1/prediction_log.jsonl"
  stats_one "G3_Qwen_vs_G5_Qwen0p5_G1" "${OUT_ROOT}/G3_Qwen/prediction_log.jsonl" "${OUT_ROOT}/G5_Qwen0p5_G1/prediction_log.jsonl"
  stats_one "G3_Qwen_vs_G4_Qwen_4bit" "${OUT_ROOT}/G3_Qwen/prediction_log.jsonl" "${OUT_ROOT}/G4_Qwen_4bit/prediction_log.jsonl"
}

case "${MODE}" in
  prepare)
    prepare_metadata
    ;;
  quick)
    run_quick
    ;;
  full)
    run_full
    ;;
  hybrid)
    run_hybrid
    ;;
  stats)
    run_stats
    ;;
  all)
    run_full
    run_hybrid
    run_stats
    ;;
  *)
    usage
    die "unknown mode: ${MODE}"
    ;;
esac

log "done: ${MODE}"
