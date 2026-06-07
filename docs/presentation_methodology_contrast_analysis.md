# 발표 관점 방법론 대비 분석

이 문서는 `raw_plan.pptx`의 원래 실험 의도를 기준으로, 현재 `runs/` 산출물에서 발표용으로 가장 극적인 방법론 대비를 만들 수 있는 지점과 부족한 지점을 정리한다. 목적은 단순히 최고 accuracy 모델을 고르는 것이 아니라, 발표에서 설득력 있게 보일 수 있는 "방법론이 왜 필요한가"의 흐름을 만드는 것이다.

## 1. 한 줄 결론

현재 결과로 가장 강한 메시지는 다음이다.

> 단순 embedding threshold는 빠르지만 TOEIC vocabulary judging에서 불안정하고, cross-encoder와 Small LM은 품질을 끌어올린다. 다만 3B급 Small LM은 운영 비용이 크므로, 실제 발표의 climax는 0.5B/1.5B 압축 LM이 3B급 품질을 훨씬 낮은 latency/VRAM으로 근접 재현하는 지점이다.

반대로, 현재 결과만으로는 다음 메시지는 약하다.

> SFT/KD가 zero-shot 대비 accuracy를 극적으로 끌어올렸다.

이 주장은 raw test가 너무 쉬워 대부분 LM 계열이 이미 `0.99+` accuracy를 보이기 때문에 발표 중심축으로 삼기 어렵다. 대신 SFT/KD의 의미는 accuracy 상승보다 `출력 계약`, `calibration`, `confidence 안정성`, `배포 가능한 small model로의 압축`으로 잡아야 한다.

## 2. PPTX의 원래 실험 의도

`raw_plan.pptx`의 핵심은 "대형 LLM을 small/local/cheap한 방법론으로 대체할 수 있는 경계"를 찾는 것이다.

발표 흐름상 원래 의도는 다음 네 축이다.

| 축 | PPTX 의도 | 발표 질문 |
|---|---|---|
| Discriminative baseline | mxbai threshold, MLP, cross-encoder | embedding만으로 충분한가, 어디서 깨지는가 |
| Small LM path | zero-shot, SFT, augmentation SFT, KD | small generative LM이 bounded vocabulary task에서 먹히는가 |
| Compression path | G5 0.5B/1.5B, G4 quantization | 품질을 유지하면서 latency/VRAM을 줄일 수 있는가 |
| Hybrid fallback | embedding + cross-encoder/API/LM fallback | 모든 문항에 비싼 모델을 쓰지 않고 운영 가능한가 |

따라서 발표에서 보여줘야 할 것은 "최고 모델은 무엇인가"가 아니라 다음 구조다.

1. 기존 앱의 embedding threshold가 실패한다.
2. 같은 문제를 cross-encoder와 Small LM이 거의 해결한다.
3. 하지만 3B LM은 느리고 무겁다.
4. 압축/양자화/하이브리드가 품질-효율 경계를 다시 바꾼다.

## 3. 현재 결과가 말하는 실제 흐름

현재 final test 결과 대부분은 `Raw Meaning Selection` 기준이다. 이 점이 중요하다. PPTX는 Synonym, Antonym, Context Cloze, Explanation, Distractor Elimination까지 상정했지만, 현재 최종 비교의 대부분은 raw Korean meaning selection에 집중되어 있다.

### 3.1 핵심 수치 요약

| 계열 | Run | Accuracy | Macro-F1 | ECE | p50 latency | Peak VRAM/RAM | 발표 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| Embedding threshold | `B0_test_final` | 0.607 | 0.606 | 0.198 | 21ms | 676MB | 빠르지만 품질 한계가 큼 |
| Embedding + logistic | `B1_test_final` | 0.599 | 0.598 | 0.124 | 21ms | 676MB | 단순 학습 scorer는 해결책 아님 |
| Embedding + MLP | `B2_test_final` | 0.622 | 0.620 | 0.120 | 21ms | 676MB | B0 대비 소폭 개선 |
| Cross-encoder | `B3_test_final` | 1.000 | 1.000 | 0.315 | 24ms | 2206MB | 가장 극적인 discriminative 대비 |
| API recheck subset | `B4_test_final` | 0.998 | 0.998 | 0.009 | 1055ms | 563MB RAM | 품질은 높지만 외부 API latency 부담 |
| 3B Qwen KD | `G3_Qwen_test_final` | 0.999 | 0.999 | 0.023 | 2620ms | 5950MB | 품질은 높지만 무겁고 느림 |
| Qwen 0.5B SFT | `G5_Qwen0p5_G1_test_final` | 0.998 | 0.998 | 0.003 | 294ms | 987MB | 3B 대비 훨씬 배포 친화적 |
| Qwen 1.5B SFT | `G5_Qwen1p5_G1_test_final` | 1.000 | 1.000 | 0.000 | 355ms | 3001MB | LM 계열 climax 후보 |
| Qwen G4 4bit | `G4_Qwen_test_final/4bit` | 0.999 | 0.999 | 0.164 | 1863ms | 5950MB | accuracy 유지, latency 개선 |
| Hybrid | `H1_test_final` | 0.861 | 0.860 | 0.156 | 22ms | 2206MB | B0 대비 개선, 정책 보강 필요 |

`B4_test_final`은 `n=579`로 전체 raw test `n=1151`과 직접 같은 평가 범위가 아니다. 발표에서는 "API가 어려운 subset에서 매우 강하지만 느리다" 정도로 쓰는 것이 안전하다.

## 4. 발표용으로 가장 강한 대비

### 4.1 첫 번째 극적 대비: embedding threshold의 한계

가장 선명한 시작점은 `B0/B1/B2`와 `B3` 비교다.

| 비교 | Accuracy | p50 latency | 메시지 |
|---|---:|---:|---|
| `B0` mxbai threshold | 0.607 | 21ms | 기존 앱형 semantic similarity는 빠르지만 의미 판별력이 부족 |
| `B2` MLP scorer | 0.622 | 21ms | embedding feature에 classifier를 얹어도 한계가 큼 |
| `B3` cross-encoder | 1.000 | 24ms | query-option interaction을 직접 보면 거의 해결 |

발표 메시지:

> 단순 embedding similarity는 "비슷한 문장 벡터"를 보는 방식이라 선택지 간 의미 구분이 약하다. Cross-encoder는 question과 option을 함께 읽기 때문에 같은 latency 규모에서 accuracy가 크게 오른다.

이 대비는 PPTX의 "embedding space collision", "짧은 답/다의어에서 오판 가능성"과 가장 잘 연결된다.

주의할 점:

- `B3=1.000`은 발표에 매우 강하지만, 너무 완벽하므로 "문항이 쉬운 raw meaning selection에 한정"이라는 단서를 달아야 한다.
- 코드상 reranker는 `query_text`와 `option` pair를 scoring하고 argmax를 고르는 구조라 직접 label leakage는 보이지 않는다.
- 다만 dataset 난이도 자체가 cross-encoder에 유리할 수 있어 hard eval 추가가 필요하다.

### 4.2 두 번째 극적 대비: 3B LM은 좋지만 너무 무겁다

`G3_Qwen`은 거의 완벽한 품질을 보인다.

| Run | Accuracy | ECE | p50 latency | tokens/sec | Peak VRAM/RAM |
|---|---:|---:|---:|---:|---:|
| `G0_Qwen_test_final` | 0.990 | 0.976 | 2502ms | 25.5 | 5943MB |
| `G1_Qwen_test_final` | 0.998 | 0.111 | 2624ms | 24.3 | 5950MB |
| `G2_Qwen_test_final` | 0.997 | 0.449 | 2626ms | 24.3 | 5950MB |
| `G3_Qwen_test_final` | 0.999 | 0.023 | 2620ms | 24.4 | 5950MB |

여기서 accuracy 차이만 보면 dramatic하지 않다. `0.990 -> 0.999`는 좋은 개선이지만 발표의 중심축으로는 약하다.

대신 극적인 지점은 calibration이다.

- `G0_Qwen`: ECE `0.976`
- `G3_Qwen`: ECE `0.023`

발표 메시지:

> Zero-shot LM은 정답을 많이 맞히지만 confidence와 출력 계약이 앱 scorer로 쓰기에는 불안정하다. KD/SFT의 가치는 accuracy 자체보다 모델 출력을 scoring pipeline에 넣을 수 있게 안정화하는 데 있다.

이 메시지는 "LLM이 맞히는가"가 아니라 "제품 채점기로 쓸 수 있는가"로 논점을 바꿔준다.

### 4.3 세 번째 극적 대비: 압축 LM의 배포 가능성

LM 계열 안에서 가장 좋은 climax는 `G3_Qwen 3B`와 `G5` compressed student 비교다.

| Run | Accuracy | ECE | p50 latency | tokens/sec | Peak VRAM/RAM |
|---|---:|---:|---:|---:|---:|
| `G3_Qwen_test_final` | 0.999 | 0.023 | 2620ms | 24.4 | 5950MB |
| `G5_Qwen0p5_G1_test_final` | 0.998 | 0.003 | 294ms | 88.3 | 987MB |
| `G5_Qwen0p5_G2_test_final` | 0.999 | 0.161 | 516ms | 88.4 | 987MB |
| `G5_Qwen1p5_G1_test_final` | 1.000 | 0.000 | 355ms | 42.2 | 3001MB |
| `G5_Qwen1p5_G2_test_final` | 0.999 | 0.001 | 352ms | 42.5 | 3001MB |

발표 메시지:

> 3B KD 모델은 품질이 좋지만 inference cost가 크다. 그런데 0.5B/1.5B student는 거의 같은 accuracy를 훨씬 낮은 latency와 VRAM으로 달성한다. 이 지점이 "Small LM can replace large fallback for bounded task"의 실질적 근거다.

강한 숫자:

- `G5_Qwen0p5_G1`은 `G3_Qwen` 대비 p50 latency가 `2620ms -> 294ms`로 약 89% 감소한다.
- Peak VRAM/RAM은 `5950MB -> 987MB`로 약 83% 감소한다.
- Accuracy는 `0.999 -> 0.998`로 거의 유지된다.

이 비교는 발표에서 bar chart보다 2-axis plot이 좋다.

- x축: p50 latency
- y축: accuracy
- marker size: peak VRAM/RAM
- highlight: `G3_Qwen`, `G5_Qwen0p5_G1`, `G5_Qwen1p5_G1`

### 4.4 네 번째 대비: quantization은 품질 손실보다 latency story

Qwen G4 결과:

| Mode | Accuracy | ECE | p50 latency | p95 latency | tokens/sec |
|---|---:|---:|---:|---:|---:|
| fp16 | 0.999 | 0.023 | 2628ms | 2663ms | 24.3 |
| 8bit | 0.998 | 0.000 | 5059ms | 5527ms | 12.5 |
| 4bit | 0.999 | 0.164 | 1863ms | 1972ms | 28.9 |

발표 메시지:

> 동일 G3 checkpoint에서 4bit는 accuracy를 거의 유지하면서 latency를 낮춘다. 다만 calibration은 악화될 수 있으므로 quantization 후 confidence calibration은 별도 과제다.

주의할 점:

- 현재 `peak_VRAM_or_RAM`은 fp16/8bit/4bit에서 거의 같게 찍혀 있다.
- 따라서 "4bit로 VRAM이 줄었다"는 주장은 현재 로그만으로는 하지 않는 편이 안전하다.
- 대신 "latency와 throughput 측면에서 4bit가 유리했다"로 제한하는 것이 좋다.

## 5. LM 계열 안에서만 발표해야 할 경우

사용자가 LM 계열만으로 대비를 보여주고 싶다면, 다음 흐름이 가장 안전하다.

### 5.1 LM-only 발표 제목 후보

> Fine-tuning improves reliability, compression improves deployability.

또는 한국어 제목:

> 정답률보다 중요한 것은 신뢰도와 배포성: Small LM 압축 실험

### 5.2 LM-only 스토리라인

1. Zero-shot LM은 이미 raw meaning task에서 많이 맞힌다.
2. 하지만 zero-shot Qwen은 confidence/calibration이 망가져 앱 scorer로 쓰기 어렵다.
3. SFT/KD는 accuracy 폭발보다 출력 계약과 calibration 안정화에 의미가 있다.
4. 3B KD는 품질은 높지만 latency/VRAM이 크다.
5. 0.5B/1.5B student는 품질을 유지하면서 운영 비용을 크게 낮춘다.
6. Quantization은 추가 latency 최적화 수단이지만 calibration 재점검이 필요하다.

### 5.3 LM-only 비교 표

| Stage | Run | Accuracy | ECE | p50 latency | 발표 포인트 |
|---|---|---:|---:|---:|---|
| Zero-shot | `G0_Qwen` | 0.990 | 0.976 | 2502ms | 맞히지만 confidence가 scorer로 부적합 |
| Raw SFT | `G1_Qwen` | 0.998 | 0.111 | 2624ms | 출력 안정화/정답률 개선 |
| KD | `G3_Qwen` | 0.999 | 0.023 | 2620ms | calibration까지 안정화 |
| 0.5B student | `G5_Qwen0p5_G1` | 0.998 | 0.003 | 294ms | 배포 가능성 급상승 |
| 1.5B student | `G5_Qwen1p5_G1` | 1.000 | 0.000 | 355ms | 품질-효율 균형 최강 후보 |
| 4bit | `G4_Qwen/4bit` | 0.999 | 0.164 | 1863ms | latency 개선, calibration 주의 |

### 5.4 LM-only 발표에서 피해야 할 주장

피해야 할 주장:

> KD가 zero-shot 대비 accuracy를 극적으로 올렸다.

이유:

- `G0_Qwen`도 이미 `0.990`이다.
- `G0_Gemma`도 `0.995`다.
- `G5_Qwen1p5_ZS`도 `0.997`이다.

대신 쓸 주장:

> Raw meaning selection은 zero-shot도 높지만, 제품 채점기로 쓰려면 calibration, output contract, latency, VRAM까지 봐야 한다. 그 관점에서 SFT/KD/Compression의 의미가 드러난다.

## 6. 현재 실험에서 부족한 지점

### 6.1 final test가 raw meaning selection에 치우침

현재 대부분의 final test metrics는 `Raw Meaning Selection`만 포함한다.

예시:

- `B0_test_final`: `Raw Meaning Selection = 1151`
- `B3_test_final`: `Raw Meaning Selection = 1151`
- `G3_Qwen_test_final`: `Raw Meaning Selection = 1151`
- `G5_*_test_final`: 대부분 `Raw Meaning Selection = 1151`

PPTX는 다음 task도 포함했다.

- Synonym Selection
- Antonym Selection
- Context Cloze
- Meaning Explanation
- Distractor Elimination

따라서 현재 결과는 PPTX 전체 의도 중 "raw meaning 선택형 채점"에는 잘 맞지만, "다양한 vocabulary judging task에서 방법론 차이가 드러나는가"에는 아직 부족하다.

### 6.2 Context Cloze 데이터의 answer position bias

증강 데이터는 바로 발표용 평가로 쓰기 위험하다.

| 데이터 | 총 개수 | answer_idx 분포 |
|---|---:|---|
| `data/metadata/aug_judge_pass_v1.jsonl` | 1450 | A=1269, B=98, C=64, D=19 |
| `data/eval/test_cloze_validated_v1.jsonl` | 36 | A=34, B=2 |
| `data/metadata/raw_mcq_v1.jsonl` | 7743 | A=1943, B=1973, C=1931, D=1896 |

raw MCQ는 정답 위치가 균형적이지만, generated/cloze 데이터는 A 편향이 매우 강하다. 이 상태에서 Context Cloze 성능을 발표하면 모델 성능이 아니라 position bias를 측정할 수 있다.

### 6.3 B3가 너무 완벽해서 hard set 필요

`B3_test_final`은 accuracy `1.000`이다. 발표에는 강하지만, 학술적/실험적 설득력에는 리스크가 있다.

확인된 점:

- B3 구현은 각 item에 대해 `(query_text, option)` pair를 넣고 score argmax를 선택한다.
- 직접 label을 예측에 넣는 코드상 leakage는 보이지 않는다.

남는 리스크:

- raw meaning task 자체가 cross-encoder에 너무 쉽다.
- distractor가 semantic hard negative가 아닐 수 있다.
- Context Cloze와 Synonym/Antonym으로 확장하면 결과가 달라질 수 있다.

## 7. 발표를 극적으로 만들기 위한 보강 실험 2개

현재 결과만으로도 발표는 가능하지만, PPTX의 원래 방법론 의도와 더 강하게 맞추려면 아래 두 단계가 가장 효율적이다.

### 7.1 보강 실험 1: Balanced Hard Evaluation Set

목적:

> B0/B2가 왜 약하고, B3/Small LM이 왜 필요한지 raw meaning보다 더 어려운 조건에서 보여준다.

해야 할 일:

1. `aug_judge_pass`와 `test_cloze_validated`의 options를 seed 고정으로 재셔플한다.
2. answer_idx가 A/B/C/D에 균등하게 분포하도록 만든다.
3. task별 최소 평가 개수를 맞춘다.
4. hard slice를 만든다.

추천 hard slice:

| Slice | 의도 |
|---|---|
| short Korean meaning | 짧은 답에서 embedding collision 확인 |
| polysemy word | 다의어에서 context/part-of-speech 필요성 확인 |
| near-synonym distractor | 단순 similarity의 한계 확인 |
| Context Cloze | 문맥을 읽는 모델과 option scorer의 차이 확인 |
| Synonym/Antonym mixed | 의미 방향성을 구분하는지 확인 |

추천 비교 모델:

- `B0`
- `B2`
- `B3`
- `G3_Qwen`
- `G5_Qwen0p5_G1`
- `G5_Qwen1p5_G1`

예상 발표 효과:

> Raw meaning에서는 대부분 LM이 잘하지만, hard set에서는 방법론별 차이가 더 선명해진다. 특히 embedding threshold는 어려운 의미 구분에서 무너지고, cross-encoder/Small LM은 문맥을 활용한다.

### 7.2 보강 실험 2: Calibrated Hybrid / Cost Frontier

목적:

> 모든 문항에 비싼 모델을 쓰지 않고도 accuracy-latency-cost 균형을 잡는 운영 정책을 보여준다.

현재 H1의 문제:

- B0 confidence가 높아도 실제 accuracy가 낮다.
- `H1_test_final`에서 high confidence로 primary를 통과한 687개 item의 accuracy는 약 `0.767`이다.
- cross-encoder를 사용한 464개 item은 accuracy `1.000`이다.

즉, H1은 "fallback이 의미 있다"는 근거는 있지만, threshold policy가 아직 보수적으로 충분히 튜닝되지 않았다.

해야 할 일:

1. H1 high threshold grid를 더 넓힌다.
   - 현재: `0.3~0.8`
   - 추천: `0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.01`
2. fallback 후보를 3종으로 비교한다.
   - B3 cross-encoder
   - G5 0.5B
   - API recheck
3. 실제 cost estimate를 넣는다.
4. Pareto frontier를 그린다.

추천 발표 그래프:

- x축: p50 latency 또는 cost per 1000 questions
- y축: accuracy
- 선/점: B0-only, B0+B3, B0+G5, B0+API
- marker label: fallback rate

예상 발표 효과:

> 기존 앱 구조를 버리는 것이 아니라, confidence와 stress signal을 기준으로 필요한 문항에만 강한 모델을 붙이면 운영 비용을 통제하면서 품질을 끌어올릴 수 있다.

## 8. 추천 발표 구성

### Slide 1: Problem Reframing

제목:

> From Semantic Similarity to Deployable Vocabulary Judging

핵심 메시지:

- 기존 앱은 embedding threshold 기반이다.
- 빠르지만 짧은 답/다의어/유사 의미에서 불안정하다.
- 연구 질문은 "대형 LLM을 small/local 방법으로 대체 가능한가"이다.

### Slide 2: Methodology Ladder

시각화:

```text
B0 mxbai threshold
  -> B1/B2 learned embedding scorer
  -> B3 cross-encoder
  -> G0/G1/G3 Small LM
  -> G5 compressed Small LM
  -> G4 quantized LM
  -> H1 hybrid fallback
```

핵심 메시지:

- 각 단계는 단순 모델 나열이 아니라 문제 해결 방식의 변화다.
- similarity -> pairwise reasoning -> generative judging -> compressed deployment -> hybrid operation.

### Slide 3: Embedding Baseline Breaks

그래프:

- bar chart: `B0`, `B1`, `B2`, `B3`
- y축: accuracy
- annotation: p50 latency

핵심 숫자:

- `B0`: 0.607
- `B2`: 0.622
- `B3`: 1.000

말할 내용:

> embedding feature를 조금 학습해도 근본적인 한계는 남는다. query와 option을 함께 읽는 cross-encoder가 task 구조에 더 맞다.

### Slide 4: Small LM Is Accurate But Heavy

그래프:

- `G0_Qwen`, `G1_Qwen`, `G3_Qwen`
- accuracy와 ECE를 같이 보여준다.

핵심 숫자:

- `G0_Qwen`: acc 0.990, ECE 0.976
- `G3_Qwen`: acc 0.999, ECE 0.023

말할 내용:

> zero-shot도 맞히지만 confidence가 망가진다. SFT/KD는 raw accuracy보다 scoring system으로 쓸 수 있는 출력을 만드는 데 의미가 있다.

### Slide 5: Compression Is the Real Replacement Boundary

그래프:

- scatter plot
- x축: p50 latency
- y축: accuracy
- size: VRAM/RAM

표시할 점:

- `G3_Qwen`
- `G5_Qwen0p5_G1`
- `G5_Qwen1p5_G1`

핵심 숫자:

- `G3_Qwen`: acc 0.999, p50 2620ms, peak 5950MB
- `G5_Qwen0p5_G1`: acc 0.998, p50 294ms, peak 987MB
- `G5_Qwen1p5_G1`: acc 1.000, p50 355ms, peak 3001MB

말할 내용:

> 3B가 잘한다는 것은 충분하지 않다. 제품 대체 가능성은 0.5B/1.5B에서 같은 품질을 훨씬 싸게 재현할 때 생긴다.

### Slide 6: Quantization as Latency Optimization

그래프:

- `G4_Qwen fp16`, `8bit`, `4bit`
- latency bar + accuracy line

핵심 숫자:

- fp16: acc 0.999, p50 2628ms
- 4bit: acc 0.999, p50 1863ms

주의:

- VRAM 감소는 현재 로그로 강하게 말하지 않는다.
- calibration은 4bit에서 악화될 수 있다.

### Slide 7: Hybrid Policy, Current and Next

현재 결과:

- `B0`: acc 0.607, p50 21ms
- `H1`: acc 0.861, p50 22ms
- cross-encoder 사용 item accuracy: 1.000
- high-confidence primary item accuracy: 약 0.767

핵심 메시지:

> hybrid는 방향성이 맞지만, current threshold는 아직 완성형이 아니다. confidence calibration과 threshold expansion이 다음 단계다.

### Slide 8: What We Need to Add

두 보강 실험:

1. Balanced Hard Evaluation Set
2. Calibrated Hybrid / Cost Frontier

핵심 메시지:

> 현재 결과는 방법론의 가능성을 보여준다. 추가 실험 1~2개를 통해 PPTX의 원래 의도인 "대체 경계"를 더 명확히 만들 수 있다.

## 9. 발표에서 사용할 수 있는 문장

### 9.1 Opening

> 처음에는 embedding similarity threshold만으로 충분할 것처럼 보였지만, 실제 선택형 의미 판별에서는 빠른 대신 불안정했다. 이 실험은 그 실패 지점에서 시작해 cross-encoder, Small LM, KD, compression, hybrid fallback으로 점진적으로 대체 경계를 찾는 과정이다.

### 9.2 B0 vs B3

> B0는 21ms로 빠르지만 accuracy가 0.607에 머물렀다. 반면 B3 cross-encoder는 비슷한 single-item latency 범위에서 1.000까지 올라갔다. 즉 이 task에서는 단순 embedding similarity보다 query-option interaction을 직접 보는 방식이 훨씬 적합했다.

### 9.3 SFT/KD

> SFT/KD는 raw test에서 accuracy를 폭발적으로 올리는 역할보다는, zero-shot LM의 불안정한 confidence와 출력 계약을 scoring pipeline에 넣을 수 있게 안정화하는 역할이 더 크다.

### 9.4 Compression

> 3B KD 모델은 품질이 높지만 p50 latency가 약 2.6초였다. 반면 0.5B student는 비슷한 accuracy를 약 0.3초와 1GB 미만 peak footprint로 달성했다. 이 지점이 실제 replacement boundary다.

### 9.5 Caveat

> 다만 현재 final test는 Raw Meaning Selection 중심이므로, Context Cloze와 hard negative가 포함된 balanced hard set을 추가해야 PPTX의 전체 task 설계를 더 엄밀하게 입증할 수 있다.

## 10. 최종 권장 메시지

발표의 큰 제목은 다음 방향이 가장 결과와 잘 맞는다.

> Small LM이 대형 LLM을 대체할 수 있는가?

보다 정확한 결론은 다음이다.

> bounded TOEIC vocabulary judging에서는 단순 embedding threshold보다 cross-encoder와 Small LM이 훨씬 강하다. 그러나 3B Small LM 자체는 운영 부담이 크므로, 실제 배포 가능성은 0.5B/1.5B 압축 모델과 calibrated hybrid policy에서 나온다.

현재 결과만으로 강하게 말할 수 있는 것:

- embedding threshold baseline은 raw meaning selection에서 명확한 한계를 보였다.
- cross-encoder는 같은 문제에서 매우 강한 discriminative alternative다.
- zero-shot LM은 accuracy가 높아도 calibration/output contract가 불안정할 수 있다.
- 0.5B/1.5B student는 3B급 품질을 훨씬 낮은 latency/VRAM으로 근접 재현했다.
- quantization은 accuracy를 유지하면서 latency를 개선할 수 있지만 calibration과 VRAM 계측은 추가 확인이 필요하다.

추가 검증 없이는 조심해야 할 것:

- "SFT/KD가 accuracy를 극적으로 올렸다."
- "Context Cloze에서도 같은 결론이 난다."
- "4bit가 VRAM을 줄였다."
- "`B3=1.000`이 모든 task에서 일반화된다."

따라서 발표 전략은 다음이 가장 안전하다.

1. B0 실패로 문제의식을 만든다.
2. B3와 LM으로 방법론 전환의 효과를 보여준다.
3. G3의 무거움을 보여준다.
4. G5 압축 모델로 배포 가능성을 보여준다.
5. 남은 gap은 balanced hard set과 calibrated hybrid 실험으로 제시한다.
