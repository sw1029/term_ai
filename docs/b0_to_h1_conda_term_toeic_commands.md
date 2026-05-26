# B0 ~ H1 실험 실행 명령어 모음: conda `term_toeic`

이 문서는 `README.md`의 실험 설계 기준을 Linux/bash와 conda 환경 `term_toeic` 기준으로 옮긴 실행 명령 모음입니다. 모든 명령은 저장소 루트(`/home/swfool/prj/term_ai`)에서 실행하는 것을 전제로 합니다.

핵심 원칙은 다음과 같습니다.

- `dev` split에서 threshold, fallback policy, prompt, early stopping을 먼저 결정합니다.
- `test` split은 마지막 확정 비교에서 한 번만 사용합니다.
- generated 학습 데이터는 `aug_judge_pass` 이상만 사용합니다. 이 프로젝트에서 `aug_judge_pass`는 human approved가 아니라 strict judge validated입니다.
- G4 quantization은 반드시 동일한 G3 KD adapter checkpoint에서 `fp16`, `8bit`, `4bit`만 바꿔 비교합니다.
- H1 hybrid policy는 dev prediction log로 튜닝하고, test에서는 dev에서 선택한 threshold를 고정합니다.

## 0. 공통 실행 방식

`conda activate` 없이 재현 가능한 형태를 우선 사용합니다.

```bash
python -m <module> ...
```

반복 입력이 길면 현재 셸에서만 아래 alias를 써도 됩니다.

```bash
PY="python"
```

설치와 기본 검증:

```bash
python -m pip install -e ".[dev,llm,baseline,metrics,train]"
python -m pytest
```

API 호출이 필요한 B4, raw teacher score, judge 계열은 `.env` 또는 환경변수에 key가 있어야 합니다.

```bash
OPENAI_API_KEY=...
QWEN_API_KEY=...
DASHSCOPE_API_KEY=...
HF_TOKEN=...
```

## 1. 입력 artifact 확인 및 재생성

이미 `data/` 산출물이 있으면 이 섹션은 건너뛰어도 됩니다. 완전 재생성이 필요할 때만 실행합니다.

```bash
python -m term_ai.augmentation.pipeline prepare \
  --input pharaprased_voca.jsonl \
  --output-dir data \
  --seed 42

python -m term_ai.augmentation.dataset_builder raw-sft \
  --anchors data/processed/anchors_v1.jsonl \
  --output-dir data/sft \
  --seed 42

python -m term_ai.augmentation.dataset_builder raw-mcq \
  --anchors data/processed/anchors_v1.jsonl \
  --output-dir data/metadata \
  --seed 42
```

strict judge validated augmentation이 이미 없다면 README의 Phase 2 절차로 `train/dev/test_aug_judge_pass_v1.jsonl`을 만든 뒤, Linux에서는 다음처럼 합칩니다.

```bash
cat \
  data/metadata/train_aug_judge_pass_v1.jsonl \
  data/metadata/dev_aug_judge_pass_v1.jsonl \
  data/metadata/test_aug_judge_pass_v1.jsonl \
  > data/metadata/aug_judge_pass_v1.jsonl
```

SFT와 eval view를 갱신합니다.

```bash
python -m term_ai.augmentation.dataset_builder validated-sft \
  --metadata data/metadata/aug_judge_pass_v1.jsonl \
  --output-dir data/sft \
  --min-status aug_judge_pass

python -m term_ai.augmentation.dataset_builder raw-aug-sft \
  --raw-sft-dir data/sft \
  --metadata data/metadata/aug_judge_pass_v1.jsonl \
  --output-dir data/sft \
  --min-status aug_judge_pass \
  --output-prefix raw_judge_aug

python -m term_ai.augmentation.dataset_builder eval-sets \
  --raw-metadata data/metadata/raw_mcq_v1.jsonl \
  --validated-metadata data/metadata/aug_judge_pass_v1.jsonl \
  --output-dir data/eval \
  --min-status aug_judge_pass
```

## 2. KD view 준비

E1/G3는 raw + judge-validated augmentation + teacher score 조건을 만족해야 합니다. raw MCQ에 teacher score가 없으면 먼저 생성합니다.

```bash
python -m term_ai.augmentation.raw_teacher_scores \
  --metadata data/metadata/raw_mcq_v1.jsonl \
  --output data/metadata/raw_teacher_scores_v1.jsonl \
  --model gpt-5.4-mini \
  --env .env \
  --requests-per-second 1 \
  --reasoning-effort low \
  --resume
```

G3 KD 학습에서는 train/dev가 섞이면 안 되므로 split별 KD view를 만듭니다. E1은 같은 metadata 안에서 train split으로 학습하고 dev/test split을 평가하므로 all-split KD view를 별도로 둡니다.

```bash
python -m term_ai.augmentation.dataset_builder kd-metadata \
  --raw-metadata data/metadata/raw_mcq_v1.jsonl \
  --generated-metadata data/metadata/aug_judge_pass_v1.jsonl \
  --output data/metadata/kd_all_view_v1.jsonl \
  --min-status aug_judge_pass \
  --raw-teacher-scores data/metadata/raw_teacher_scores_v1.jsonl
```

```bash
python -m term_ai.augmentation.dataset_builder kd-metadata \
  --raw-metadata data/metadata/raw_train_mcq_v1.jsonl \
  --generated-metadata data/metadata/train_aug_judge_pass_v1.jsonl \
  --output data/metadata/kd_train_view_v1.jsonl \
  --min-status aug_judge_pass \
  --raw-teacher-scores data/metadata/raw_teacher_scores_v1.jsonl

python -m term_ai.augmentation.dataset_builder kd-metadata \
  --raw-metadata data/metadata/raw_dev_mcq_v1.jsonl \
  --generated-metadata data/metadata/dev_aug_judge_pass_v1.jsonl \
  --output data/metadata/kd_dev_view_v1.jsonl \
  --min-status aug_judge_pass \
  --raw-teacher-scores data/metadata/raw_teacher_scores_v1.jsonl

python -m term_ai.augmentation.dataset_builder kd-metadata \
  --raw-metadata data/metadata/raw_test_mcq_v1.jsonl \
  --generated-metadata data/metadata/test_aug_judge_pass_v1.jsonl \
  --output data/metadata/kd_test_view_v1.jsonl \
  --min-status aug_judge_pass \
  --raw-teacher-scores data/metadata/raw_teacher_scores_v1.jsonl
```

## 3. Dev split: B0 ~ B4 baseline

B0/B1/B2는 `raw_gt` MCQ metadata를 기본 입력으로 사용합니다.

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=B0 \
  execution.output_dir=runs/B0_dev \
  evaluation.split=dev

python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=B1 \
  execution.output_dir=runs/B1_dev \
  evaluation.split=dev

python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=B2 \
  execution.output_dir=runs/B2_dev \
  evaluation.split=dev
```

B3 reranker는 기본적으로 `BAAI/bge-reranker-v2-m3` zero-shot reranker를 사용합니다.

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=B3 \
  execution.output_dir=runs/B3_dev \
  evaluation.split=dev \
  execution.reranker.fine_tune=false
```

B3 fine-tune 비교가 필요하면 별도 output으로 실행합니다.

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=B3 \
  execution.output_dir=runs/B3_finetuned_dev \
  evaluation.split=dev \
  execution.reranker.fine_tune=true \
  execution.reranker.epochs=1 \
  execution.reranker.batch_size=8
```

B4는 API recheck fallback입니다. B0 prediction을 primary로 주면 낮은 confidence/stress item만 재검사합니다.

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=B4 \
  execution.output_dir=runs/B4_dev \
  evaluation.split=dev \
  execution.primary_predictions=runs/B0_dev/prediction_log.jsonl \
  execution.api_recheck.provider=openai \
  execution.api_recheck.model=gpt-5.4-mini \
  execution.api_recheck.env_path=.env \
  execution.api_recheck.requests_per_second=1 \
  execution.api_recheck.confidence_threshold=0.7
```

OpenAI-compatible Qwen/DashScope endpoint를 쓰는 경우:

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=B4 \
  execution.output_dir=runs/B4_qwen_dev \
  evaluation.split=dev \
  execution.primary_predictions=runs/B0_dev/prediction_log.jsonl \
  execution.api_recheck.provider=qwen-compatible \
  execution.api_recheck.model=YOUR_QWEN_API_MODEL \
  execution.api_recheck.base_url=https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1 \
  execution.api_recheck.api_key_env=QWEN_API_KEY \
  execution.api_recheck.env_path=.env \
  execution.api_recheck.requests_per_second=1
```

## 4. Dev split: G0 zero-shot Small LM

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G0-Gemma \
  execution.output_dir=runs/G0_Gemma_dev \
  execution.model_name_or_path=google/gemma-2-2b-it \
  evaluation.split=dev \
  execution.local_cost_per_hour_usd=0

python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G0-Qwen \
  execution.output_dir=runs/G0_Qwen_dev \
  execution.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
  evaluation.split=dev \
  execution.local_cost_per_hour_usd=0
```

## 5. Dev split: G1 raw LoRA SFT

Hydra 실행은 LoRA adapter를 학습하고, `dev` post-train evaluation을 `post_train_eval/` 아래에 기록합니다.

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G1-Gemma \
  execution.output_dir=runs/G1_Gemma_dev \
  execution.model_name_or_path=google/gemma-2-2b-it \
  evaluation.split=dev \
  training.save_total_limit=3

python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G1-Qwen \
  execution.output_dir=runs/G1_Qwen_dev \
  execution.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
  evaluation.split=dev \
  training.save_total_limit=3
```

주요 산출물:

- `runs/G1_Gemma_dev/final_adapter`
- `runs/G1_Gemma_dev/post_train_eval/prediction_log.jsonl`
- `runs/G1_Qwen_dev/final_adapter`
- `runs/G1_Qwen_dev/post_train_eval/prediction_log.jsonl`

## 6. Dev split: G2 raw + judge-validated augmentation LoRA SFT

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G2-Gemma \
  execution.output_dir=runs/G2_Gemma_dev \
  execution.model_name_or_path=google/gemma-2-2b-it \
  evaluation.split=dev \
  training.save_total_limit=3

python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G2-Qwen \
  execution.output_dir=runs/G2_Qwen_dev \
  execution.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
  evaluation.split=dev \
  training.save_total_limit=3
```

주요 산출물:

- `runs/G2_Gemma_dev/final_adapter`
- `runs/G2_Gemma_dev/post_train_eval/prediction_log.jsonl`
- `runs/G2_Qwen_dev/final_adapter`
- `runs/G2_Qwen_dev/post_train_eval/prediction_log.jsonl`

## 7. Dev split: G3 raw + aug + teacher score LoRA KD

G3는 KD adapter 학습 단계입니다. `execution.min_status=any`를 명시해 split별 KD view 안의 `raw_gt`와 `aug_judge_pass`를 함께 사용합니다.

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G3-Gemma \
  execution.output_dir=runs/G3_Gemma_dev \
  execution.model_name_or_path=google/gemma-2-2b-it \
  execution.kd_metadata=data/metadata/kd_train_view_v1.jsonl \
  execution.kd_dev_metadata=data/metadata/kd_dev_view_v1.jsonl \
  execution.min_status=any \
  evaluation.split=dev \
  training.kd.lambda_soft=0.5 \
  training.kd.include_rationale=true \
  training.kd.require_teacher_scores=true

python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G3-Qwen \
  execution.output_dir=runs/G3_Qwen_dev \
  execution.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
  execution.kd_metadata=data/metadata/kd_train_view_v1.jsonl \
  execution.kd_dev_metadata=data/metadata/kd_dev_view_v1.jsonl \
  execution.min_status=any \
  evaluation.split=dev \
  training.kd.lambda_soft=0.5 \
  training.kd.include_rationale=true \
  training.kd.require_teacher_scores=true
```

G3 자체의 dev prediction log가 필요하면 학습 완료 후 adapter를 별도로 평가합니다.

```bash
python -m term_ai.experiment.lm_eval \
  --metadata data/metadata/raw_mcq_v1.jsonl \
  --output-dir runs/G3_Gemma_dev/post_train_eval \
  --model-name-or-path google/gemma-2-2b-it \
  --adapter-path runs/G3_Gemma_dev/final_adapter \
  --eval-split dev \
  --min-status raw_gt \
  --experiment-id G3-Gemma

python -m term_ai.experiment.lm_eval \
  --metadata data/metadata/raw_mcq_v1.jsonl \
  --output-dir runs/G3_Qwen_dev/post_train_eval \
  --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
  --adapter-path runs/G3_Qwen_dev/final_adapter \
  --eval-split dev \
  --min-status raw_gt \
  --experiment-id G3-Qwen
```

## 8. Dev split: G4 quantization comparison

G4는 같은 G3 adapter를 `fp16`, `8bit`, `4bit`로 각각 평가합니다. Gemma G3 checkpoint 기준:

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G4-8bit \
  execution.output_dir=runs/G4_Gemma_dev \
  execution.model_name_or_path=google/gemma-2-2b-it \
  execution.adapter_path=runs/G3_Gemma_dev/final_adapter \
  evaluation.split=dev \
  execution.local_cost_per_hour_usd=0
```

Qwen G3 checkpoint 기준:

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G4-8bit \
  execution.output_dir=runs/G4_Qwen_dev \
  execution.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
  execution.adapter_path=runs/G3_Qwen_dev/final_adapter \
  evaluation.split=dev \
  execution.local_cost_per_hour_usd=0
```

주요 산출물:

- `runs/G4_Gemma_dev/fp16/prediction_log.jsonl`
- `runs/G4_Gemma_dev/8bit/prediction_log.jsonl`
- `runs/G4_Gemma_dev/4bit/prediction_log.jsonl`
- `runs/G4_Gemma_dev/quantization_compare.json`

## 9. Dev split: E1 embedding scorer KD

E1도 KD view의 raw + generated teacher scores를 같이 쓰려면 `execution.min_status=any`를 명시합니다.

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=E1 \
  execution.output_dir=runs/E1_dev \
  execution.kd_metadata=data/metadata/kd_all_view_v1.jsonl \
  execution.min_status=any \
  evaluation.split=dev \
  training.kd.require_teacher_scores=true
```

## 10. Dev split: H1 hybrid scorer + fallback policy

H1은 prediction log를 조합합니다. 기본 조합은 B0 primary + B4 fallback이고, B3 reranker를 cross-encoder 중간 경로로 추가할 수 있습니다.

Dev에서 threshold grid를 튜닝합니다.

```bash
python -m term_ai.experiment.hydra_app \
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
```

튜닝 결과는 `runs/H1_dev/hybrid_policy_tuning.json`의 `selected_policy.low_confidence_threshold`와 `selected_policy.high_confidence_threshold`에서 확인합니다. 예를 들어 dev에서 `low=0.4`, `high=0.7`이 선택되었다면 고정 policy 실행은 다음과 같습니다.

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=H1 \
  execution.output_dir=runs/H1_dev_fixed \
  evaluation.split=dev \
  execution.primary_predictions=runs/B0_dev/prediction_log.jsonl \
  execution.cross_encoder_predictions=runs/B3_dev/prediction_log.jsonl \
  execution.fallback_predictions=runs/B4_dev/prediction_log.jsonl \
  execution.hybrid.tune_policy=false \
  execution.hybrid.low_confidence_threshold=0.4 \
  execution.hybrid.high_confidence_threshold=0.7
```

## 11. Final test 실행

아래 명령은 dev에서 모든 선택을 끝낸 뒤 마지막에만 실행합니다. 공통 test lock 디렉터리는 `runs/_test_locks`입니다. 반복 test가 필요할 때만 `execution.allow_repeat_test=true` 또는 `--allow-repeat-test`를 쓰고, 최종 보고서에 반드시 명시합니다.

### 11.1 Baseline final test

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=B0 \
  execution.output_dir=runs/B0_test_final \
  evaluation.split=test \
  execution.test_lock_dir=runs/_test_locks

python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=B1 \
  execution.output_dir=runs/B1_test_final \
  evaluation.split=test \
  execution.test_lock_dir=runs/_test_locks

python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=B2 \
  execution.output_dir=runs/B2_test_final \
  evaluation.split=test \
  execution.test_lock_dir=runs/_test_locks

python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=B3 \
  execution.output_dir=runs/B3_test_final \
  evaluation.split=test \
  execution.reranker.fine_tune=false \
  execution.test_lock_dir=runs/_test_locks
```

B4 test는 test용 B0 prediction을 primary로 사용합니다.

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=B4 \
  execution.output_dir=runs/B4_test_final \
  evaluation.split=test \
  execution.primary_predictions=runs/B0_test_final/prediction_log.jsonl \
  execution.api_recheck.provider=openai \
  execution.api_recheck.model=gpt-5.4-mini \
  execution.api_recheck.env_path=.env \
  execution.api_recheck.requests_per_second=1 \
  execution.test_lock_dir=runs/_test_locks
```

### 11.2 G0 final test

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G0-Gemma \
  execution.output_dir=runs/G0_Gemma_test_final \
  execution.model_name_or_path=google/gemma-2-2b-it \
  evaluation.split=test \
  execution.test_lock_dir=runs/_test_locks

python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G0-Qwen \
  execution.output_dir=runs/G0_Qwen_test_final \
  execution.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
  evaluation.split=test \
  execution.test_lock_dir=runs/_test_locks
```

### 11.3 G1/G2/G3 adapter final test

G1/G2/G3는 test split에서 재학습하지 않습니다. dev 단계에서 학습한 adapter를 `lm_eval`로 잠금 평가합니다.

```bash
python -m term_ai.experiment.lm_eval \
  --metadata data/metadata/raw_mcq_v1.jsonl \
  --output-dir runs/G1_Gemma_test_final \
  --model-name-or-path google/gemma-2-2b-it \
  --adapter-path runs/G1_Gemma_dev/final_adapter \
  --eval-split test \
  --min-status raw_gt \
  --experiment-id G1-Gemma \
  --test-lock-dir runs/_test_locks

python -m term_ai.experiment.lm_eval \
  --metadata data/metadata/raw_mcq_v1.jsonl \
  --output-dir runs/G1_Qwen_test_final \
  --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
  --adapter-path runs/G1_Qwen_dev/final_adapter \
  --eval-split test \
  --min-status raw_gt \
  --experiment-id G1-Qwen \
  --test-lock-dir runs/_test_locks

python -m term_ai.experiment.lm_eval \
  --metadata data/metadata/raw_mcq_v1.jsonl \
  --output-dir runs/G2_Gemma_test_final \
  --model-name-or-path google/gemma-2-2b-it \
  --adapter-path runs/G2_Gemma_dev/final_adapter \
  --eval-split test \
  --min-status raw_gt \
  --experiment-id G2-Gemma \
  --test-lock-dir runs/_test_locks

python -m term_ai.experiment.lm_eval \
  --metadata data/metadata/raw_mcq_v1.jsonl \
  --output-dir runs/G2_Qwen_test_final \
  --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
  --adapter-path runs/G2_Qwen_dev/final_adapter \
  --eval-split test \
  --min-status raw_gt \
  --experiment-id G2-Qwen \
  --test-lock-dir runs/_test_locks

python -m term_ai.experiment.lm_eval \
  --metadata data/metadata/raw_mcq_v1.jsonl \
  --output-dir runs/G3_Gemma_test_final \
  --model-name-or-path google/gemma-2-2b-it \
  --adapter-path runs/G3_Gemma_dev/final_adapter \
  --eval-split test \
  --min-status raw_gt \
  --experiment-id G3-Gemma \
  --test-lock-dir runs/_test_locks

python -m term_ai.experiment.lm_eval \
  --metadata data/metadata/raw_mcq_v1.jsonl \
  --output-dir runs/G3_Qwen_test_final \
  --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
  --adapter-path runs/G3_Qwen_dev/final_adapter \
  --eval-split test \
  --min-status raw_gt \
  --experiment-id G3-Qwen \
  --test-lock-dir runs/_test_locks
```

### 11.4 G4 final test

G4 final test는 최종 비교 대상으로 선택한 G3 checkpoint 하나를 기준으로 실행합니다. 현재 G4 runner의 final-test lock은 `G4` family 단위라서 Gemma와 Qwen을 모두 test에서 실행하려면 두 번째 실행에는 `execution.allow_repeat_test=true`가 필요하며, 그 사실을 보고서에 남겨야 합니다.

Gemma G3 checkpoint를 선택한 경우:

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G4-8bit \
  execution.output_dir=runs/G4_Gemma_test_final \
  execution.model_name_or_path=google/gemma-2-2b-it \
  execution.adapter_path=runs/G3_Gemma_dev/final_adapter \
  evaluation.split=test \
  execution.test_lock_dir=runs/_test_locks
```

Qwen G3 checkpoint를 선택한 경우:

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G4-8bit \
  execution.output_dir=runs/G4_Qwen_test_final \
  execution.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
  execution.adapter_path=runs/G3_Qwen_dev/final_adapter \
  evaluation.split=test \
  execution.test_lock_dir=runs/_test_locks
```

### 11.5 E1 final test

E1은 test 시에도 train split으로 KD scorer를 학습하고 test split만 평가합니다. 같은 output directory 재사용은 피합니다.

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=E1 \
  execution.output_dir=runs/E1_test_final \
  execution.kd_metadata=data/metadata/kd_all_view_v1.jsonl \
  execution.min_status=any \
  evaluation.split=test \
  training.kd.require_teacher_scores=true \
  execution.test_lock_dir=runs/_test_locks
```

### 11.6 H1 final test

먼저 dev에서 선택된 H1 threshold를 고정합니다. 아래 예시는 `low=0.4`, `high=0.7`입니다.

H1은 metadata를 직접 재평가하지 않고 test prediction log를 병합합니다. 따라서 실질적인 final-test lock은 upstream B0/B3/B4 test 실행에서 생성됩니다.

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=H1 \
  execution.output_dir=runs/H1_test_final \
  evaluation.split=test \
  execution.primary_predictions=runs/B0_test_final/prediction_log.jsonl \
  execution.cross_encoder_predictions=runs/B3_test_final/prediction_log.jsonl \
  execution.fallback_predictions=runs/B4_test_final/prediction_log.jsonl \
  execution.hybrid.tune_policy=false \
  execution.hybrid.low_confidence_threshold=0.4 \
  execution.hybrid.high_confidence_threshold=0.7 \
  execution.test_lock_dir=runs/_test_locks
```

## 12. 통계 검정과 보고서 생성

대표 비교 예시:

```bash
python -m term_ai.experiment.statistics \
  --predictions-a runs/B3_test_final/prediction_log.jsonl \
  --predictions-b runs/G3_Gemma_test_final/prediction_log.jsonl \
  --output reports/B3_vs_G3_Gemma_test_statistics.json \
  --samples 1000

python -m term_ai.experiment.statistics \
  --predictions-a runs/B0_test_final/prediction_log.jsonl \
  --predictions-b runs/H1_test_final/prediction_log.jsonl \
  --output reports/B0_vs_H1_test_statistics.json \
  --samples 1000
```

최종 report skeleton:

```bash
python -m term_ai.experiment.reporting \
  --runs-dir runs \
  --output-dir reports
```

## 13. Resume와 재실행

대부분의 Hydra 실험은 기본적으로 resume을 켭니다. 같은 `execution.output_dir`로 다시 실행하면 가능한 경우 partial prediction, checkpoint, adapter를 이어 씁니다.

새로 시작해야 할 때:

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=B0 \
  execution.output_dir=runs/B0_dev_clean \
  evaluation.split=dev \
  execution.resume=false
```

LoRA Trainer checkpoint load만 끄려면:

```bash
python -m term_ai.experiment.hydra_app \
  execution.run=true \
  model.experiment_id=G2-Gemma \
  execution.output_dir=runs/G2_Gemma_dev_clean \
  execution.model_name_or_path=google/gemma-2-2b-it \
  evaluation.split=dev \
  training.resume=false
```

## 14. 빠른 점검 checklist

- `no train items` 또는 `no dev rows`가 나오면 `execution.min_status`, split별 KD view, metadata path를 확인합니다.
- G3/E1에서 raw row까지 포함해야 하면 `execution.min_status=any`를 사용합니다.
- `kd_dev_view_v1.jsonl`이 없으면 2장의 split별 KD view 생성 명령을 먼저 실행합니다.
- B4/H1은 입력 prediction log 경로가 실제 존재해야 합니다.
- G4 manifest 오류가 나오면 `execution.adapter_path`가 G3 KD `final_adapter`인지 확인합니다.
- `evaluation.split=test`는 final test에서만 사용하고, dev tuning이 끝나기 전에는 실행하지 않습니다.
