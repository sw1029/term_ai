# Term AI Experiment Guide

이 저장소는 TOEIC business vocabulary 데이터를 word-level split으로 고정한 뒤,
데이터 증강, baseline, Small LM LoRA SFT/KD, quantization, hybrid fallback을
비교하기 위한 실험 코드입니다. 세부 설계 근거는 `plan.md`, `plan_raw_v1.md`,
`data_plan.md`, `TODO.md`를 기준으로 합니다.

## 원칙

- split은 augmentation보다 먼저 수행합니다. 같은 `word_id`가 train/dev/test에
  동시에 들어가면 안 됩니다.
- final test는 마지막 비교에만 사용합니다. threshold, fallback policy,
  early stopping, prompt selection은 dev에서만 결정합니다.
- Human spot check는 현재 실행 파이프라인에서 제외합니다. 따라서
  `aug_judge_pass`는 "strict judge validated"이고, `aug_human_pass` 또는
  "approved/human approved"로 부르지 않습니다.
- Teacher 생성 데이터는 최소 `aug_judge_pass`를 만족해야 학습 view에 넣습니다.
  `aug_candidate` 또는 `aug_auto_pass`만으로 학습하지 않습니다.
- G4 quantization은 반드시 동일 G3 KD adapter checkpoint에서 FP16/8bit/4bit만
  바꿔 비교합니다.

## 환경 준비

PowerShell 기준입니다.

```powershell
python -m venv venv
.\venv\Scripts\python -m pip install --upgrade pip
```

GPU torch는 로컬 CUDA 버전에 맞는 wheel을 먼저 설치합니다. 예를 들어 CUDA 12.1
환경이면 다음처럼 설치한 뒤 project extras를 설치합니다.

```powershell
.\venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.\venv\Scripts\python -m pip install -e ".[dev,llm,baseline,metrics,train]"
```

OpenAI 또는 compatible API 호출이 필요한 단계는 `.env`에 key를 둡니다.

```text
OPENAI_API_KEY=...
QWEN_API_KEY=...
DASHSCOPE_API_KEY=...
```

기본 검증:

```powershell
.\venv\Scripts\python -m pytest
```

## Phase 0~1: raw artifact 생성

원본 입력은 기본적으로 `pharaprased_voca.jsonl`입니다.

```powershell
.\venv\Scripts\python -m term_ai.augmentation.pipeline prepare `
  --input pharaprased_voca.jsonl `
  --output-dir data `
  --seed 42

.\venv\Scripts\python -m term_ai.augmentation.dataset_builder raw-sft `
  --anchors data/processed/anchors_v1.jsonl `
  --output-dir data/sft `
  --seed 42

.\venv\Scripts\python -m term_ai.augmentation.dataset_builder raw-mcq `
  --anchors data/processed/anchors_v1.jsonl `
  --output-dir data/metadata `
  --seed 42
```

주요 산출물:

- `data/processed/anchors_v1.jsonl`
- `data/splits/word_split_seed42.json`
- `data/sft/raw_train_sft_v1.jsonl`
- `data/sft/raw_dev_sft_v1.jsonl`
- `data/sft/raw_test_sft_v1.jsonl`
- `data/metadata/raw_mcq_v1.jsonl`

## Phase 2: 증강 생성, 필터, judge validation

요청 제한은 기본적으로 초당 1회입니다.

```powershell
.\venv\Scripts\python -m term_ai.augmentation.orchestrator `
  --split train `
  --total 1000 `
  --requests-per-second 1 `
  --model gpt-5.4-mini

.\venv\Scripts\python -m term_ai.augmentation.orchestrator `
  --split dev `
  --total 200 `
  --requests-per-second 1 `
  --model gpt-5.4-mini

.\venv\Scripts\python -m term_ai.augmentation.orchestrator `
  --split test `
  --total 200 `
  --requests-per-second 1 `
  --model gpt-5.4-mini
```

각 split 후보를 자동 필터링합니다.

```powershell
.\venv\Scripts\python -m term_ai.augmentation.pipeline auto-filter `
  --metadata data/aug/train_aug_candidate_v1.jsonl `
  --output data/aug/train_aug_auto_pass_v1.jsonl
```

Judge model은 generator model과 다르게 지정합니다.

```powershell
.\venv\Scripts\python -m term_ai.augmentation.judge_llm `
  --metadata data/aug/train_aug_auto_pass_v1.jsonl `
  --output data/judge/train_judge_v1.jsonl `
  --model <judge_model> `
  --generator-model gpt-5.4-mini `
  --requests-per-second 1

.\venv\Scripts\python -m term_ai.augmentation.pipeline apply-judge `
  --metadata data/aug/train_aug_auto_pass_v1.jsonl `
  --judge data/judge/train_judge_v1.jsonl `
  --output data/metadata/train_aug_judge_pass_v1.jsonl
```

dev/test도 같은 방식으로 `dev_aug_judge_pass_v1.jsonl`,
`test_aug_judge_pass_v1.jsonl`을 만듭니다. 이후 하나의 strict judge metadata로
합칩니다.

```powershell
Get-Content -Path @(
  "data/metadata/train_aug_judge_pass_v1.jsonl",
  "data/metadata/dev_aug_judge_pass_v1.jsonl",
  "data/metadata/test_aug_judge_pass_v1.jsonl"
) |
  Set-Content -Encoding utf8 data/metadata/aug_judge_pass_v1.jsonl
```

## Phase 2 이후 dataset view 생성

학습용 SFT, raw+aug SFT, strict eval set을 만듭니다.

```powershell
.\venv\Scripts\python -m term_ai.augmentation.dataset_builder validated-sft `
  --metadata data/metadata/aug_judge_pass_v1.jsonl `
  --output-dir data/sft `
  --min-status aug_judge_pass

.\venv\Scripts\python -m term_ai.augmentation.dataset_builder raw-aug-sft `
  --raw-sft-dir data/sft `
  --metadata data/metadata/aug_judge_pass_v1.jsonl `
  --output-dir data/sft `
  --min-status aug_judge_pass `
  --output-prefix raw_judge_aug

.\venv\Scripts\python -m term_ai.augmentation.dataset_builder eval-sets `
  --raw-metadata data/metadata/raw_mcq_v1.jsonl `
  --validated-metadata data/metadata/aug_judge_pass_v1.jsonl `
  --output-dir data/eval `
  --min-status aug_judge_pass
```

KD view는 raw GT에도 teacher score가 있어야 raw+aug+teacher score 계약을
만족합니다.

```powershell
.\venv\Scripts\python -m term_ai.augmentation.dataset_builder kd-metadata `
  --raw-metadata data/metadata/raw_mcq_v1.jsonl `
  --generated-metadata data/metadata/aug_judge_pass_v1.jsonl `
  --output data/metadata/kd_train_view_v1.jsonl `
  --min-status aug_judge_pass `
  --raw-teacher-scores data/metadata/raw_teacher_scores_v1.jsonl
```

raw teacher score가 아직 없으면 `--exclude-raw`로 generated-only KD view를
만들 수 있지만, 이 경우 문서상의 E1/G3 "raw + teacher score" 실험과는 다릅니다.

## Hydra 단일 실험 실행

모든 실험은 dev split에서 먼저 실행합니다.

```powershell
.\venv\Scripts\python -m term_ai.experiment.hydra_app `
  execution.run=true `
  model.experiment_id=B0 `
  execution.output_dir=runs/B0_dev `
  evaluation.split=dev
```

주요 experiment id:

- `B0`: mxbai threshold baseline
- `B1`: mxbai + logistic
- `B2`: mxbai + MLP
- `B3`: reranker/cross-encoder
- `B4`: API recheck fallback
- `G0-Gemma`, `G0-Qwen`: zero-shot Small LM
- `G1-Gemma`, `G1-Qwen`: raw train LoRA SFT
- `G2-Gemma`, `G2-Qwen`: raw + judge-validated augmentation LoRA SFT
- `G3-Gemma`, `G3-Qwen`: raw + aug + teacher score LoRA KD
- `G4-8bit`: same G3 checkpoint quantization comparison wrapper
- `E1`: embedding scorer KD
- `H1`: hybrid scorer + fallback policy

G0/G4처럼 모델 경로가 필요한 경우:

```powershell
.\venv\Scripts\python -m term_ai.experiment.hydra_app `
  execution.run=true `
  model.experiment_id=G0-Qwen `
  execution.output_dir=runs/G0_Qwen_dev `
  execution.model_name_or_path=Qwen/Qwen2.5-3B-Instruct `
  evaluation.split=dev `
  execution.local_cost_per_hour_usd=1.2
```

G4는 G3 KD adapter manifest를 기본적으로 검증합니다.

```powershell
.\venv\Scripts\python -m term_ai.experiment.hydra_app `
  execution.run=true `
  model.experiment_id=G4-8bit `
  execution.output_dir=runs/G4_Gemma_dev `
  execution.model_name_or_path=google/gemma-2-2b-it `
  execution.adapter_path=runs/G3_Gemma_dev/final_adapter `
  evaluation.split=dev
```

## Master workflow

`phase_jobs`를 직접 쓰지 않아도 기본 matrix를 자동 생성할 수 있습니다. 이 명령은
API 호출과 학습을 포함할 수 있으므로, 실행 전 config override를 확인합니다.

```powershell
.\venv\Scripts\python -m term_ai.experiment.hydra_app `
  workflow.master_enabled=true `
  workflow.execute_expensive_steps=true `
  workflow.config.strict_judge_metadata=data/metadata/aug_judge_pass_v1.jsonl `
  workflow.config.raw_teacher_scores=data/metadata/raw_teacher_scores_v1.jsonl
```

master workflow에서 증강 생성까지 같이 수행하려면 split별 total과 judge model을
명시합니다.

```powershell
.\venv\Scripts\python -m term_ai.experiment.hydra_app `
  workflow.master_enabled=true `
  workflow.execute_expensive_steps=true `
  workflow.config.augmentation.enabled=true `
  workflow.config.augmentation.split_totals.train=1000 `
  workflow.config.augmentation.split_totals.dev=200 `
  workflow.config.augmentation.split_totals.test=200 `
  workflow.config.augmentation.requests_per_second=1 `
  workflow.config.augmentation.judge.enabled=true `
  workflow.config.augmentation.judge.model=<judge_model>
```

## Prompt variation, KD ablation, explanation judge

Prompt template sweep:

```powershell
.\venv\Scripts\python -m term_ai.experiment.prompt_variation_sweep `
  --train-jsonl data/sft/raw_judge_aug_train_sft_v1.jsonl `
  --dev-jsonl data/sft/raw_judge_aug_dev_sft_v1.jsonl `
  --output-dir runs/prompt_variation `
  --model-name-or-path google/gemma-2-2b-it `
  --execute-training `
  --eval-metadata data/metadata/raw_mcq_v1.jsonl
```

G3 KD ablation sweep:

```powershell
.\venv\Scripts\python -m term_ai.experiment.kd_sweep `
  --model-name-or-path google/gemma-2-2b-it `
  --metadata-jsonl data/metadata/kd_train_view_v1.jsonl `
  --dev-metadata-jsonl data/metadata/kd_dev_view_v1.jsonl `
  --output-dir runs/G3_kd_ablation `
  --execute-training `
  --eval-metadata data/metadata/raw_mcq_v1.jsonl
```

Explanation judge:

```powershell
.\venv\Scripts\python -m term_ai.experiment.explanation_judge judge `
  --predictions runs/G0_Qwen_dev/prediction_log.jsonl `
  --output runs/explanation_judge/G0_Qwen_explanation_judgments.jsonl `
  --judge-model <judge_model> `
  --generator-model Qwen/Qwen2.5-3B-Instruct `
  --requests-per-second 1

.\venv\Scripts\python -m term_ai.experiment.explanation_judge summarize `
  --judgments runs/explanation_judge/G0_Qwen_explanation_judgments.jsonl `
  --output runs/explanation_judge/G0_Qwen_explanation_judgment_summary.json
```

## 통계 검정과 최종 report

같은 item에 대한 두 prediction log를 비교합니다.

```powershell
.\venv\Scripts\python -m term_ai.experiment.statistics `
  --predictions-a runs/B3_dev/prediction_log.jsonl `
  --predictions-b runs/G3_Gemma_dev/prediction_log.jsonl `
  --output reports/B3_vs_G3_statistics.json `
  --samples 1000
```

최종 report 입력과 markdown skeleton을 생성합니다.

```powershell
.\venv\Scripts\python -m term_ai.experiment.reporting `
  --runs-dir runs `
  --output-dir reports
```

주요 산출물:

- `reports/final_experiment_report_input.json`
- `reports/error_analysis_input.json`
- `reports/explanation_judge_report_input.json`
- `reports/deployment_recommendation_input.json`
- `reports/final_experiment_report.md`
- `reports/deployment_recommendation.md`

## Final test 실행 규칙

dev에서 모든 threshold, fallback, prompt, early stopping을 결정한 뒤 마지막에만
`evaluation.split=test`를 사용합니다. 기본 test lock은 `runs/_test_locks`에
생성됩니다.

```powershell
.\venv\Scripts\python -m term_ai.experiment.hydra_app `
  execution.run=true `
  model.experiment_id=B0 `
  execution.output_dir=runs/B0_test_final `
  evaluation.split=test `
  execution.test_lock_dir=runs/_test_locks
```

반복 test가 필요할 때만 `execution.allow_repeat_test=true`를 쓰며, 보고서에는 그
사실을 명시해야 합니다.

## 문제 확인 checklist

- `no train items`가 나오면 `min_status`, split, metadata path를 확인합니다.
- `judge model must differ`가 나오면 generator와 judge model을 다르게 지정합니다.
- G4에서 manifest 오류가 나오면 adapter가 `train_lora_sft_kd` 또는
  `kd_sweep` 산출물인지 확인합니다.
- generated cloze 성능과 raw test 성능은 같은 표에서 섞지 말고 별도 subset으로
  해석합니다.
- 비용 지표는 API token usage 또는 `execution.local_cost_per_hour_usd` 중 하나가
  있어야 자동 집계됩니다.
