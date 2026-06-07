from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape

from lxml import etree
from PIL import Image


EMU_PER_INCH = 914400
SLIDE_W = 12192000
SLIDE_H = 6858000

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

REL_NS = NS["rel"]
P_NS = NS["p"]
R_NS = NS["r"]

COLORS = {
    "navy": "0B1F33",
    "teal": "1B8A8F",
    "green": "3A9D5D",
    "orange": "E79A32",
    "red": "C84A3A",
    "purple": "7F63B8",
    "gray": "6F747D",
    "light_gray": "F3F5F7",
    "line": "D9DEE5",
    "black": "111827",
    "muted": "5F6B7A",
    "white": "FFFFFF",
}


def emu(inches: float) -> int:
    return int(round(inches * EMU_PER_INCH))


def xml_text(value: str) -> str:
    return escape(str(value), {"\n": "&#10;"})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def pp(value: float, digits: int = 1) -> str:
    return f"{value * 100:+.{digits}f}pp"


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}ms" if value < 1000 else f"{value/1000:.2f}s"


def fmt_mb(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}MB" if value < 1024 else f"{value/1024:.1f}GB"


def label_from_idx(value: Any) -> str:
    if isinstance(value, int) and 0 <= value <= 3:
        return "ABCD"[value]
    return "?"


def clean_shape_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_ -]", "", name)[:64] or "shape"


def line_spacing_pct(value: int) -> int:
    """Return DrawingML spacing in 1/1000 percent units.

    Earlier calls used values like 900 to mean "dense but readable". Google
    Slides imports that literally as 0.9, which stacks paragraphs on top of
    each other. Clamp legacy compact values to normal 100% spacing.
    """
    if value < 10_000:
        return max(100_000, value * 100)
    return value


TEXT_REPLACEMENTS = {
    "TOEIC Business Vocabulary Judging | No-API / No-Retrain Result Report": "TOEIC 비즈니스 어휘 판정 | No-API / No-Retrain 결과 보고",
    "Can Small / Local AI Replace Large LLM or API for TOEIC Vocabulary Judging?": "작은 로컬 AI의 대형 LLM/API 대체 가능성",
    "No-API / No-Retrain 확인 결과 + 원본 실험 설계 의도 기반 방법론 대비": "No-API / No-Retrain 확인 결과와 원본 실험 설계 의도 기반 방법론 대비",
    "white background result deck | raw_plan.pptx design density preserved": "흰 바탕 결과 발표안 | raw_plan.pptx의 내용 밀도 유지",
    "B/G/H Experiment Stages and Intent": "B/G/H 실험 단계와 의도",
    "Original PPTX Intent → Result Reading Frame": "원본 PPTX 의도 → 결과 해석 프레임",
    "Raw Final Test vs Check-500": "Raw final test와 Check-500 비교",
    "Embedding-only Baseline Breaks": "Embedding-only 기준선의 한계",
    "Methodological Stage Ladder": "방법론 단계 사다리",
    "Average Accuracy Hides Hard Slices": "평균 정확도 뒤의 어려운 구간",
    "Antonym Slice Is the Real Stress Test": "Antonym 구간이 실제 stress test",
    "Accuracy vs Latency vs Memory Frontier": "정확도-지연시간-메모리 경계",
    "Runtime Cost Makes the Deployment Story": "실행 시간이 배포 판단을 만든다",
    "Reliability Is Not the Same as Accuracy": "신뢰성은 정확도와 다르다",
    "Hybrid Routing Is a Policy Experiment": "Hybrid routing은 운영 정책 실험",
    "Paired Statistics: Which Gains Are Real?": "Paired statistics: 실제 차이가 있는 개선",
    "Original Final Test Result Landscape": "Original final test 결과 지형",
    "LM Stages: Accuracy Saturates, Reliability Matters": "LM 단계: 정확도 포화 이후 신뢰성",
    "Compression Boundary": "압축 경계",
    "Experimental Design Appendix Map": "실험 설계 부록 지도",
    "Data and Augmentation Protocol": "데이터와 증강 절차",
    "Model Family Architecture Map": "모델군 구조 지도",
    "Objective, Inference, and Metric Protocol": "목표 함수·추론·지표 절차",
    "Visual Evidence: Evaluation and Method Gap": "시각 근거: 평가 난이도와 방법론 격차",
    "Visual Evidence: Hard Slice and Deployment": "시각 근거: 어려운 구간과 배포성",
    "Evaluation Set Evidence": "평가 세트 근거",
    "Evaluation Subset Visual Audit": "평가 부분집합 시각 점검",
    "Model Structures: B0 / B2 / B3": "모델 구조: B0 / B2 / B3",
    "Model Structures: G3 / G4": "모델 구조: G3 / G4",
    "Model Structures: G5 Small Local LM": "모델 구조: G5 소형 로컬 LM",
    "No-Retrain Evaluation Scope": "재학습 없는 평가 범위",
    "Model Structure: H1 Local Hybrid": "모델 구조: H1 로컬 Hybrid",
    "Metric Evidence: Reliability": "지표 근거: 신뢰성",
    "Metric Evidence: Paired Statistics": "지표 근거: paired statistics",
    "Metric Evidence: Runtime Cost": "지표 근거: 실행 시간",
    "Aggregate Metric Evidence Matrix": "집계 지표 근거표",
    "EXPERIMENT RESULTS": "실험 결과",
    "EXPERIMENT DESIGN": "실험 설계",
    "ORIGINAL PLAN": "원본 설계",
    "EVALUATION FRAME": "평가 프레임",
    "RESULT 1": "결과 1",
    "RESULT 2": "결과 2",
    "RESULT 3": "결과 3",
    "RESULT 4": "결과 4",
    "RESULT 5": "결과 5",
    "STATISTICS": "통계",
    "ORIGINAL FINAL": "원본 final",
    "LM PATH": "LM 경로",
    "COMPRESSION": "압축",
    "APPENDIX": "부록",
    "Appendix": "부록",
    "appendix": "부록",
    "Best accuracy": "최고 정확도",
    "Deployable local": "배포 후보 로컬",
    "Baseline gap": "기준선 격차",
    "Hard slice": "어려운 구간",
    "Embedding baseline": "Embedding 기준선",
    "Embedding similarity": "Embedding 유사도",
    "Embedding-only": "Embedding-only",
    "MLP scorer reuse": "MLP scorer 재사용",
    "Cross-encoder": "Cross-encoder",
    "3B local LM": "3B 로컬 LM",
    "4bit compressed": "4bit 압축",
    "0.5B / 1.5B local LM": "0.5B / 1.5B 로컬 LM",
    "Local hybrid": "로컬 Hybrid",
    "Discriminative baseline": "판별형 기준선",
    "Small LM path": "소형 LM 경로",
    "Compression path": "압축 경로",
    "Hybrid fallback": "Hybrid fallback",
    "Similarity → Pairwise reading → Local LM → Compression": "유사도 → pairwise 판독 → 로컬 LM → 압축",
    "Similarity": "유사도",
    "Pairwise reading": "pairwise 판독",
    "Local LM": "로컬 LM",
    "Compression": "압축",
    "Raw final 포화": "Raw final 포화",
    "Check-500 mixed/hard slice": "Check-500 혼합/어려운 구간",
    "Embedding similarity 한계": "Embedding 유사도 한계",
    "Cross-encoder 점프": "Cross-encoder 점프",
    "Antonym hard slice": "Antonym 어려운 구간",
    "Antonym stress test": "Antonym stress test",
    "Hard slice 격차": "어려운 구간 격차",
    "Local runtime 차이": "로컬 실행 시간 차이",
    "Accuracy | Calibration | Output contract": "정확도 | Calibration | 출력 형식",
    "Confidence routing": "신뢰도 routing",
    "Local fallback": "로컬 fallback",
    "Paired comparison": "paired 비교",
    "Method contrast 제한": "방법론 대비 제한",
    "LM accuracy saturation": "LM 정확도 포화",
    "Confidence reliability": "신뢰도 안정성",
    "Output stability": "출력 안정성",
    "Small local LM": "소형 로컬 LM",
    "4bit runtime": "4bit 실행 시간",
    "data construction": "데이터 구성",
    "model families": "모델군",
    "evaluation protocol": "평가 절차",
    "evidence boundary": "근거 범위",
    "raw anchors": "원본 anchor",
    "generated task types": "생성 task 유형",
    "validation gates": "검증 gate",
    "teacher signals": "teacher signal",
    "teacher signal": "teacher 신호",
    "component state": "구성 요소 상태",
    "evaluation mode": "평가 모드",
    "training objective": "학습 목표",
    "answer contract": "답안 형식",
    "deterministic inference": "결정적 추론",
    "aggregate metrics": "집계 지표",
    "Raw saturation": "Raw 포화",
    "Check-500 contrast": "Check-500 대비",
    "Method gap": "방법론 격차",
    "Antonym stress": "Antonym stress",
    "Latency-memory frontier": "지연시간-메모리 경계",
    "500 test items": "500개 test 항목",
    "raw/generated mix": "원본/생성 혼합",
    "task slice composition": "task 구간 구성",
    "controlled subset": "통제 부분집합",
    "answer distribution": "정답 label 분포",
    "generated-slice audit": "생성 구간 점검",
    "Discriminative family": "판별형 계열",
    "per-model flow": "모델별 흐름",
    "no adapter": "adapter 없음",
    "LoRA adapter": "LoRA adapter",
    "4bit loading": "4bit 로딩",
    "deployable candidates": "배포 후보",
    "adapter loading": "adapter 로딩",
    "quantized loading": "양자화 로딩",
    "routing": "routing",
    "confidence gate": "신뢰도 gate",
    "B3 route": "B3 route",
    "G5 fallback": "G5 fallback",
    "output contract": "출력 형식",
    "aggregate only": "집계만 사용",
    "common-item comparison": "공통 문항 비교",
    "bootstrap CI": "bootstrap CI",
    "latency": "지연시간",
    "memory": "메모리",
    "local runtime": "로컬 실행 시간",
    "task-level accuracy": "task별 정확도",
    "reliability": "신뢰성",
    "runtime": "실행 시간",
    "paired evidence": "paired 근거",
    "Raw GT": "원본 GT",
    "Raw": "원본",
    "word / meaning": "단어 / 뜻",
    "word / POS": "단어 / 품사",
    "4-option MCQ": "4지선다 MCQ",
    "Augmentation": "증강",
    "synonym / sense": "동의어 / 의미 구분",
    "antonym / cloze": "반의어 / 문맥 빈칸",
    "Validation": "검증",
    "auto filter": "자동 필터",
    "judge validation": "judge 검증",
    "Check subset": "확인용 부분집합",
    "test items": "test 항목",
    "raw + generated": "원본 + 생성",
    "Local runs": "로컬 실행",
    "no API check": "API 없는 확인",
    "B family": "B 계열",
    "G family": "G 계열",
    "H family": "H 계열",
    "Discriminative scoring": "판별형 점수화",
    "Local LM and adapters": "로컬 LM과 adapter",
    "Routing policy": "routing 정책",
    "Metrics": "지표",
    "Aggregate evidence only": "집계 근거만 사용",
    "accuracy": "정확도",
    "task slice": "task 구간",
    "strict parse": "엄격 parsing",
    "memory": "메모리",
    "paired stats": "paired 통계",
    "design evidence": "설계 근거",
    "model structure": "모델 구조",
    "visible limits": "표시된 한계",
    "Raw anchors": "원본 anchor",
    "Task generation": "task 생성",
    "Auto filter": "자동 필터",
    "Judge gate": "judge gate",
    "semantic": "의미",
    "leakage": "누수",
    "Source mix": "Source mix",
    "Generated": "생성",
    "check subset only": "확인용 부분집합만",
    "Generated task coverage": "생성 task 구성",
    "Synonym": "동의어",
    "Sense": "의미 구분",
    "Antonym": "반의어",
    "Cloze": "문맥 빈칸",
    "stress coverage": "stress 구성",
    "slice view": "구간 보기",
    "Teacher signal placement": "teacher signal 위치",
    "Hard label": "hard label",
    "Soft": "soft 점수",
    "kept as target": "target 유지",
    "4-way": "4지선다",
    "scores": "점수",
    "KD soft scores": "KD soft 점수",
    "G5 hard-label check": "G5 hard-label 확인",
    "B family: discriminative scoring": "B 계열: 판별형 점수화",
    "G family: local LM + adapter": "G 계열: 로컬 LM + adapter",
    "H family: routing policy": "H 계열: routing 정책",
    "embed": "embed",
    "reuse": "재사용",
    "same 3B": "같은 3B",
    "4bit load": "4bit 로딩",
    "Accuracy range": "정확도 범위",
    "option scoring": "선택지 점수화",
    "no generation": "생성 없음",
    "G5 SFT adapter": "G5 SFT adapter",
    "KD separate path": "KD 별도 경로",
    "Gate": "Gate",
    "confidence": "신뢰도",
    "routing candidate": "routing 후보",
    "cost sweep next": "cost sweep 후속",
    "Training objective paths": "학습 목표 흐름",
    "assistant": "assistant",
    "completion CE": "completion CE",
    "adapter": "adapter",
    "update": "업데이트",
    "G1 SFT": "G1 SFT",
    "check": "확인",
    "G3 KD objective": "G3 KD 목표",
    "Inference and answer contract": "추론과 답안 형식",
    "Prompt": "Prompt",
    "contract": "형식",
    "deterministic": "결정적",
    "generation": "생성",
    "Parser": "Parser",
    "strict": "엄격",
    "fallback": "fallback",
    "Accuracy metric": "정확도 지표",
    "Strict parse metric": "엄격 parsing 지표",
    "Metric groups": "지표 그룹",
    "overall": "전체",
    "Strict parse": "엄격 parsing",
    "footprint": "사용량",
    "same 500 items": "동일 500문항",
    "Raw final vs Check-500": "Raw final과 Check-500",
    "Methodology Accuracy Gap": "방법론 정확도 격차",
    "B0/B2/B3/G 계열 대비": "B0/B2/B3/G 계열 대비",
    "Antonym Hard Slice": "Antonym 어려운 구간",
    "Accuracy-Latency-Memory": "정확도-지연시간-메모리",
    "Total metadata rows": "전체 metadata 행",
    "train/dev/test retained": "train/dev/test 유지",
    "Check test rows": "Check test 행",
    "presentation subset": "발표용 부분집합",
    "Raw test rows": "원본 test 행",
    "original GT items": "원본 GT 항목",
    "Generated rows": "생성 행",
    "judge-passed items": "judge 통과 항목",
    "No-API runs": "No-API 실행",
    "local-only models": "로컬 전용 모델",
    "Split retained for controlled checking": "통제 확인용 split 유지",
    "train/dev retained": "train/dev 유지",
    "test-only comparison": "test만 비교",
    "B0/B2 guard": "B0/B2 조건 유지",
    "Test task composition": "test task 구성",
    "Raw Meaning Selection": "원본 뜻 선택",
    "Sense Disambiguation": "의미 구분",
    "Context Cloze": "문맥 빈칸",
    "Antonym Selection": "반의어 선택",
    "Synonym Selection": "동의어 선택",
    "Evaluation construction": "평가 구성",
    "Retain context": "train/dev 유지",
    "Select test subset": "test 부분집합 선택",
    "Mix sources": "출처 혼합",
    "Evaluate only": "평가만 수행",
    "no API call": "API 호출 없음",
    "no retraining": "재학습 없음",
    "check-only": "확인 전용",
    "fixed seed": "고정 seed",
    "local eval": "로컬 평가",
    "Answer label distribution": "정답 label 분포",
    "raw balance": "원본 균형",
    "generated stress": "생성 stress",
    "aggregate label view": "집계 label 보기",
    "Generated slice audit": "생성 구간 점검",
    "not final": "최종 아님",
    "balance next": "균형 보완",
    "generated slice boundary": "생성 구간 범위",
    "Embedding similarity baseline": "Embedding 유사도 기준선",
    "Embedding feature + reused scorer": "Embedding feature + 재사용 scorer",
    "Question-option cross-encoder": "문제-선택지 cross-encoder",
    "sentence embedding": "문장 embedding",
    "similarity score": "유사도 점수",
    "argmax answer": "argmax 답안",
    "embedding features": "embedding feature",
    "existing MLP scorer": "기존 MLP scorer",
    "question + option": "문제 + 선택지",
    "pairwise score": "pairwise 점수",
    "ranked answer": "순위 답안",
    "Adapter": "Adapter",
    "Train": "Train",
    "none": "없음",
    "Qwen 3B local LM + existing LoRA adapter": "Qwen 3B 로컬 LM + 기존 LoRA adapter",
    "Qwen 3B 4bit loading + same adapter path": "Qwen 3B 4bit 로딩 + 같은 adapter 경로",
    "MCQ prompt": "MCQ prompt",
    "Qwen 3B base": "Qwen 3B base",
    "LoRA final adapter": "LoRA final adapter",
    "structured answer": "구조화 답안",
    "4bit loaded base": "4bit 로딩 base",
    "LoRA adapter merge path": "LoRA adapter 병합 경로",
    "Smallest local student + LoRA adapter": "가장 작은 로컬 student + LoRA adapter",
    "Small local student + LoRA adapter": "소형 로컬 student + LoRA adapter",
    "Qwen 0.5B base": "Qwen 0.5B base",
    "Qwen 1.5B base": "Qwen 1.5B base",
    "G1 LoRA adapter": "G1 LoRA adapter",
    "No-API / No-Retrain confirmation scope": "No-API / No-Retrain 확인 범위",
    "existing MLP scorer reused": "기존 MLP scorer 재사용",
    "no new scorer training": "새 scorer 학습 없음",
    "cross-encoder scoring only": "cross-encoder 점수화만 수행",
    "fine-tune disabled": "fine-tune 비활성화",
    "existing 3B LoRA KD adapter": "기존 3B LoRA KD adapter",
    "evaluation only": "평가만 수행",
    "existing 0.5B/1.5B G1 SFT adapter": "기존 0.5B/1.5B G1 SFT adapter",
    "not KD in this run": "이번 실행은 KD 아님",
    "policy over existing prediction outputs": "기존 예측 출력 기반 정책",
    "routing check only": "routing 확인만 수행",
    "load only": "로딩만 수행",
    "compose outputs": "출력 조합",
    "loaded artifacts only": "로드된 artifact만 평가",
    "no B4/API": "B4/API 없음",
    "no new LoRA/KD": "새 LoRA/KD 없음",
    "evaluation mode": "평가 모드",
    "Input": "입력",
    "MCQ item": "MCQ 문항",
    "with options": "선택지 포함",
    "Primary route": "주 경로",
    "fast local answer": "빠른 로컬 답안",
    "Fallback route": "fallback 경로",
    "harder items": "어려운 문항",
    "Output": "출력",
    "selected answer": "선택 답안",
    "Selected gate": "선택 gate",
    "low": "low",
    "high": "high",
    "Calibration and Output Contract": "Calibration과 출력 형식",
    "model-level aggregate metrics": "모델 단위 집계 지표",
    "Paired Accuracy Delta": "Paired 정확도 차이",
    "Runtime Cost by Method": "방법론별 실행 시간",
    "500-item runtime view": "500문항 실행 시간 보기",
    "Model": "모델",
    "Strict err": "형식 오류",
    "common": "공통",
    "delta": "차이",
    "G3 antonym": "G3 반의어",
    "Task slice": "task 구간",
    "Paired stats": "paired 통계",
    "Hard label": "정답 label",
    "hard label": "정답 label",
    "MLP scorer": "MLP 점수기",
    "Acc:": "정확도:",
    "H1 Acc": "H1 정확도",
    "embedding feature": "embedding 특징값",
    "Hybrid fallback": "Hybrid 대체 경로",
    "Local fallback": "로컬 대체 경로",
    "B3 route": "B3 경로",
    "G5 fallback": "G5 대체 경로",
    "test 500 items, seed fixed": "test 500개, seed 고정",
    "raw 388 + generated 112": "원본 388 + 생성 112",
    "raw 388": "원본 388",
    "gen 112": "생성 112",
    "format": "형식",
    "meaning": "뜻",
    "Judge gate": "judge 검증 gate",
    "judge gate": "judge 검증 gate",
    "teacher score": "teacher 점수",
    "assistant": "assistant 출력",
    "Fallback": "대체 경로",
    "fallback": "대체 경로",
    "MCQ prompt": "MCQ 프롬프트",
    "threshold": "임계값",
    "G5 local LM": "G5 로컬 LM",
    "selected": "선택됨",
    "Fallback route": "대체 경로",
    "aggregate only": "집계만 사용",
    "errors": "오류",
    "leap": "상승",
    "plot": "그림",
}

REGEX_TEXT_REPLACEMENTS = [
    (r"\bNo-API\s*/\s*No-Retrain\b", "API 없음 / 재학습 없음"),
    (r"\bNo-API\b", "API 없음"),
    (r"\bNo-Retrain\b", "재학습 없음"),
    (r"\bOriginal final test\b", "원본 최종 평가"),
    (r"\bOriginal final\b", "원본 최종 평가"),
    (r"\bRaw final test\b", "원본 최종 평가"),
    (r"\bRaw final\b", "원본 최종 평가"),
    (r"\bfinal raw test\b", "최종 원본 평가"),
    (r"\bfinal test\b", "최종 평가"),
    (r"\btest set\b", "평가 세트"),
    (r"\btest items\b", "평가 항목"),
    (r"\btest rows\b", "평가 행"),
    (r"\btest row\b", "평가 행"),
    (r"\btest-only\b", "평가만"),
    (r"\btest\b", "평가"),
    (r"\btrain/dev/test\b", "학습/검증/평가"),
    (r"\btrain/dev\b", "학습/검증"),
    (r"\btrain\b", "학습"),
    (r"\bdev\b", "검증"),
    (r"\braw/generated\b", "원본/생성"),
    (r"\bgenerated-slice\b", "생성 구간"),
    (r"\bgenerated slice\b", "생성 구간"),
    (r"\bgenerated\b", "생성"),
    (r"\bgeneration\b", "생성"),
    (r"\bRaw\b", "원본"),
    (r"\braw\b", "원본"),
    (r"\bSynonym\b", "동의어"),
    (r"\bsynonym\b", "동의어"),
    (r"\bSense\b", "의미 구분"),
    (r"\bsense\b", "의미 구분"),
    (r"\bAntonym\b", "반의어"),
    (r"\bantonym\b", "반의어"),
    (r"\bCloze\b", "문맥 빈칸"),
    (r"\bcloze\b", "문맥 빈칸"),
    (r"\bsyn\b", "동의어"),
    (r"\bant\b", "반의어"),
    (r"\bContext\b", "문맥"),
    (r"\bcontext\b", "문맥"),
    (r"\bEmbedding-only\b", "임베딩 단독"),
    (r"\bEmbedding\b", "임베딩"),
    (r"\bembedding\b", "임베딩"),
    (r"\bembed\b", "임베딩"),
    (r"\bCross-encoder\b", "교차 인코더"),
    (r"\bcross-encoder\b", "교차 인코더"),
    (r"\bPairwise\b", "쌍별"),
    (r"\bpairwise\b", "쌍별"),
    (r"\bSimilarity\b", "유사도"),
    (r"\bsimilarity\b", "유사도"),
    (r"\bHybrid\b", "하이브리드"),
    (r"\bhybrid\b", "하이브리드"),
    (r"\brouting\b", "라우팅"),
    (r"\broute\b", "경로"),
    (r"\bFallback\b", "대체 경로"),
    (r"\bfallback\b", "대체 경로"),
    (r"\bPrimary\b", "primary"),
    (r"\bprimary\b", "primary"),
    (r"\bGate\b", "게이트"),
    (r"\bgate\b", "게이트"),
    (r"\bJudge\b", "검증자"),
    (r"\bjudge\b", "검증자"),
    (r"\bTeacher\b", "교사"),
    (r"\bteacher\b", "교사"),
    (r"\bStudent\b", "학생 모델"),
    (r"\bstudent\b", "학생 모델"),
    (r"\bAnchor\b", "기준어"),
    (r"\banchor\b", "기준어"),
    (r"\banchors\b", "기준어"),
    (r"\bTask slice\b", "과제 구간"),
    (r"\btask slice\b", "과제 구간"),
    (r"\bTask\b", "과제"),
    (r"\btask\b", "과제"),
    (r"\bHard label\b", "정답 라벨"),
    (r"\bhard label\b", "정답 라벨"),
    (r"\bhard-label\b", "정답 라벨"),
    (r"\bHard\b", "어려운"),
    (r"\bhard\b", "어려운"),
    (r"\bSoft\b", "소프트"),
    (r"\bsoft\b", "소프트"),
    (r"\bLabel\b", "라벨"),
    (r"\blabel\b", "라벨"),
    (r"\bTarget\b", "목표"),
    (r"\btarget\b", "목표"),
    (r"\bScore\b", "점수"),
    (r"\bscore\b", "점수"),
    (r"\bScores\b", "점수"),
    (r"\bscores\b", "점수"),
    (r"\bScorer\b", "점수기"),
    (r"\bscorer\b", "점수기"),
    (r"\bAccuracy\b", "정확도"),
    (r"\baccuracy\b", "정확도"),
    (r"\bAcc\b", "정확도"),
    (r"\bCalibration\b", "보정"),
    (r"\bcalibration\b", "보정"),
    (r"\bOutput\b", "출력"),
    (r"\boutput\b", "출력"),
    (r"\bContract\b", "형식"),
    (r"\bcontract\b", "형식"),
    (r"\bStrict parse\b", "엄격 파싱"),
    (r"\bstrict parse\b", "엄격 파싱"),
    (r"\bStrict parsing\b", "엄격 파싱"),
    (r"\bstrict parsing\b", "엄격 파싱"),
    (r"\bStrict\b", "엄격"),
    (r"\bstrict\b", "엄격"),
    (r"\bParsing\b", "파싱"),
    (r"\bparsing\b", "파싱"),
    (r"\bParser\b", "파서"),
    (r"\bparser\b", "파서"),
    (r"\bPrompt\b", "프롬프트"),
    (r"\bprompt\b", "프롬프트"),
    (r"\bassistant output\b", "assistant 출력"),
    (r"\bassistant\b", "assistant"),
    (r"\bcompletion CE\b", "완성 CE"),
    (r"\bAdapter\b", "어댑터"),
    (r"\badapter\b", "어댑터"),
    (r"\bArtifacts\b", "산출물"),
    (r"\bartifacts\b", "산출물"),
    (r"\bArtifact\b", "산출물"),
    (r"\bartifact\b", "산출물"),
    (r"\bBase\b", "기반 모델"),
    (r"\bbase\b", "기반 모델"),
    (r"\bLoading\b", "로딩"),
    (r"\bloading\b", "로딩"),
    (r"\bloaded\b", "로드된"),
    (r"\bload\b", "로딩"),
    (r"\bTrain\b", "학습"),
    (r"\btraining\b", "학습"),
    (r"\bFine-tune\b", "파인튜닝"),
    (r"\bfine-tune\b", "파인튜닝"),
    (r"\bFT off\b", "FT 비활성"),
    (r"\boff\b", "비활성"),
    (r"\bStress test\b", "고난도 평가"),
    (r"\bstress test\b", "고난도 평가"),
    (r"\bStress\b", "고난도"),
    (r"\bstress\b", "고난도"),
    (r"\bLocal\b", "로컬"),
    (r"\blocal\b", "로컬"),
    (r"\bRuntime\b", "실행 시간"),
    (r"\bruntime\b", "실행 시간"),
    (r"\bLatency\b", "지연시간"),
    (r"\blatency\b", "지연시간"),
    (r"\bMemory\b", "메모리"),
    (r"\bmemory\b", "메모리"),
    (r"\bReliability\b", "신뢰성"),
    (r"\breliability\b", "신뢰성"),
    (r"\bConfidence\b", "신뢰도"),
    (r"\bconfidence\b", "신뢰도"),
    (r"\bMetric groups\b", "지표 그룹"),
    (r"\bMetric\b", "지표"),
    (r"\bmetric\b", "지표"),
    (r"\bMetrics\b", "지표"),
    (r"\bmetrics\b", "지표"),
    (r"\bPaired stats\b", "쌍대 통계"),
    (r"\bpaired stats\b", "쌍대 통계"),
    (r"\bPaired statistics\b", "쌍대 통계"),
    (r"\bpaired statistics\b", "쌍대 통계"),
    (r"\bPaired\b", "쌍대"),
    (r"\bpaired\b", "쌍대"),
    (r"\bcommon-item\b", "공통 문항"),
    (r"\bbootstrap CI\b", "부트스트랩 신뢰구간"),
    (r"\bcommon\b", "공통"),
    (r"\bdelta\b", "차이"),
    (r"\bModel\b", "모델"),
    (r"\bmodel\b", "모델"),
    (r"\bfamilies\b", "계열"),
    (r"\bfamily\b", "계열"),
    (r"\bFeature\b", "특징값"),
    (r"\bfeature\b", "특징값"),
    (r"\bfeatures\b", "특징값"),
    (r"\bOption\b", "선택지"),
    (r"\boption\b", "선택지"),
    (r"\boptions\b", "선택지"),
    (r"\bAnswer\b", "답안"),
    (r"\banswer\b", "답안"),
    (r"\bargmax\b", "최댓값 선택"),
    (r"\branked\b", "순위화된"),
    (r"\bselected\b", "선택됨"),
    (r"\bwith\b", "포함"),
    (r"\bquality ceiling\b", "품질 상한"),
    (r"\bquality\b", "품질"),
    (r"\bceiling\b", "상한"),
    (r"\bZero-shot\b", "제로샷"),
    (r"\bzero-shot\b", "제로샷"),
    (r"\bcost sweep\b", "비용 탐색"),
    (r"\bCost\b", "비용"),
    (r"\bcost\b", "비용"),
    (r"\bview\b", "보기"),
    (r"\bboundary\b", "범위"),
    (r"\bbalance\b", "균형"),
    (r"\bnext\b", "후속"),
    (r"\bonly\b", "만"),
    (r"\bnew\b", "새"),
    (r"\bexisting\b", "기존"),
    (r"\breused\b", "재사용"),
    (r"\breuse\b", "재사용"),
    (r"\bdisabled\b", "비활성화"),
    (r"\bseparate path\b", "별도 경로"),
    (r"\bsame\b", "같은"),
    (r"\bsmall\b", "소형"),
    (r"\bSmallest\b", "가장 작은"),
    (r"\bRetain\b", "유지"),
    (r"\bretained\b", "유지"),
    (r"\bSelect\b", "선택"),
    (r"\bMix\b", "혼합"),
    (r"\bEvaluate\b", "평가"),
    (r"\bEvaluate only\b", "평가만 수행"),
    (r"\bcheck-only\b", "확인 전용"),
    (r"\bfixed seed\b", "고정 seed"),
    (r"\bnot final\b", "최종 아님"),
    (r"\bnot KD in this run\b", "이번 실행은 KD 아님"),
    (r"\bMetadata\b", "메타데이터"),
    (r"\bmetadata\b", "메타데이터"),
]

ASCII_TOKEN_REPLACEMENTS = {
    "Discriminative": "판별형",
    "baseline": "기준선",
    "Baseline": "기준선",
    "compression": "압축",
    "Compression": "압축",
    "reading": "판독",
    "Reading": "판독",
    "comparison": "비교",
    "Comparison": "비교",
    "similarity": "유사도",
    "Similarity": "유사도",
    "metric": "지표",
    "Metric": "지표",
    "contract": "형식",
    "Contract": "형식",
    "cross-encoder": "교차 인코더",
    "Cross-encoder": "교차 인코더",
    "cloze": "문맥 빈칸",
    "Cloze": "문맥 빈칸",
    "judging": "판정",
    "Judging": "판정",
    "payload": "데이터 본문",
    "Payload": "데이터 본문",
    "anchor": "기준어",
    "Anchor": "기준어",
    "label": "라벨",
    "Label": "라벨",
    "sweep": "탐색",
    "Sweep": "탐색",
    "aug": "증강",
    "mixed": "혼합",
    "Mixed": "혼합",
    "slice": "구간",
    "Slice": "구간",
    "rows": "행",
    "row": "행",
    "seed": "시드",
    "Seed": "시드",
    "backbone": "중심축",
    "classifier": "분류기",
    "reranking": "재순위화",
    "compressed": "압축된",
    "case": "사례",
    "cases": "사례",
    "heatmap": "히트맵",
    "selection": "선택",
    "Selection": "선택",
    "Hybrid": "하이브리드",
    "hybrid": "하이브리드",
    "grid": "탐색 격자",
    "frontier": "경계선",
    "fine-tuning": "파인튜닝",
    "improves": "개선한다",
    "deployability": "배포성",
    "climax": "핵심 장면",
    "bounded": "제한된",
    "statistics": "통계",
    "Statistics": "통계",
    "vocabulary": "어휘",
    "Vocabulary": "어휘",
    "view": "보기",
    "View": "보기",
    "loss": "손실",
    "Loss": "손실",
    "distribution": "분포",
    "Distribution": "분포",
    "run": "실행",
    "Run": "실행",
    "outputs": "출력",
    "output": "출력",
    "prompt": "프롬프트",
    "Prompt": "프롬프트",
    "parser": "파서",
    "Parser": "파서",
    "item": "문항",
    "items": "문항",
    "inference": "추론",
    "Inference": "추론",
    "peak": "최대",
    "validated": "검증된",
    "Validated": "검증된",
    "augmentation": "증강",
    "Augmentation": "증강",
    "controlled": "통제된",
    "confirmation": "확인",
    "unbiased": "비편향",
    "benchmark": "벤치마크",
    "question": "문제",
    "Question": "문제",
    "pair": "쌍",
    "base": "기반 모델",
    "logits": "로짓",
    "logit": "로짓",
    "causal": "인과",
    "letter": "문자",
    "sequence": "시퀀스",
    "distillation": "증류",
    "loading": "로딩",
    "Loading": "로딩",
    "deployment": "배포",
    "candidate": "후보",
    "cross entropy": "교차 엔트로피",
    "token": "토큰",
    "policy": "정책",
    "Policy": "정책",
    "saving": "절감",
    "aggregate": "집계",
    "Aggregate": "집계",
    "level": "단위",
    "signal": "신호",
    "Signal": "신호",
    "bootstrap": "부트스트랩",
    "business": "비즈니스",
    "human": "사람",
    "approval": "승인",
    "completion": "완성",
    "lambda": "람다",
    "soft": "소프트",
    "Soft": "소프트",
    "gain": "개선폭",
    "wall-clock": "실제 경과시간",
    "trade-off": "절충",
    "trade-비활성": "절충",
    "gate": "게이트",
    "Gate": "게이트",
    "non": "비",
    "Embedding": "임베딩",
    "embedding": "임베딩",
    "scorer": "점수기",
    "scorers": "점수기",
    "Context": "문맥",
    "context": "문맥",
    "eval": "평가",
    "routing": "라우팅",
    "adapter": "어댑터",
    "adapters": "어댑터",
    "artifact": "산출물",
    "artifacts": "산출물",
    "prediction": "예측값",
    "predictions": "예측값",
    "over": "기반",
    "split": "분할",
    "Zero-shot": "제로샷",
    "zero-shot": "제로샷",
    "error": "오류",
    "errors": "오류",
    "Calibration": "보정",
    "calibration": "보정",
    "task": "과제",
    "tasks": "과제",
    "final": "최종",
    "test": "평가",
    "set": "세트",
    "low": "낮음",
    "high": "높음",
    "vs": "대비",
    "assistant": "어시스턴트",
}

TECHNICAL_TERM_RESTORES = {
    "API 없음 / 재학습 없음": "No-API / No-Retrain",
    "API 없는": "No-API",
    "API 없음": "No-API",
    "재학습 없음": "No-Retrain",
    "제로샷": "zero-shot",
    "임베딩": "Embedding",
    "교차 인코더": "Cross-encoder",
    "쌍별": "pairwise",
    "쌍대 통계": "paired statistics",
    "쌍대 비교": "paired comparison",
    "쌍대 근거": "paired evidence",
    "쌍대 bootstrap": "paired bootstrap",
    "부트스트랩 신뢰구간": "bootstrap CI",
    "하이브리드": "Hybrid",
    "로컬 Hybrid": "Local Hybrid",
    "로컬": "local",
    "라운팅": "routing",
    "라우팅": "routing",
    "게이트": "gate",
    "신뢰도 gate": "confidence gate",
    "점수기": "scorer",
    "어댑터": "adapter",
    "프롬프트": "prompt",
    "파서": "parser",
    "어시스턴트": "assistant",
    "교사 로짓": "teacher logits",
    "교사 점수": "teacher score",
    "교사 신호": "teacher signal",
    "교사": "teacher",
    "학생 모델": "student",
    "기준어": "anchor",
    "검증자 검증": "judge validation",
    "검증자": "judge",
    "검증 게이트": "validation gate",
    "검증 gate": "validation gate",
    "자동 필터": "auto filter",
    "소프트 점수": "soft score",
    "소프트": "soft",
    "로짓": "logits",
    "손실": "loss",
    "분포": "distribution",
    "완성 토큰 CE": "completion CE",
    "완성 토큰 영역": "completion token span",
    "교차 엔트로피": "cross entropy",
    "인과-LM": "causal LM",
    "시퀀스 증류": "sequence distillation",
    "비편향 벤치마크": "unbiased benchmark",
    "고난도 평가": "stress test",
    "고난도 구간": "stress slice",
    "어려운 구간": "hard slice",
    "생성 구간": "generated slice",
    "과제 구간": "task slice",
    "과제": "task",
    "출력 형식": "output contract",
    "답안 형식": "answer contract",
    "형식 오류": "parse error",
    "엄격 파싱": "strict parsing",
    "엄격 형식": "strict contract",
    "보정": "Calibration",
    "신뢰도": "confidence",
    "신뢰성": "reliability",
    "정확도": "Accuracy",
    "지연시간": "latency",
    "실행 시간": "runtime",
    "메모리": "memory",
    "생성 행": "generated rows",
    "생성 task": "generated task",
    "생성 과제": "generated task",
    "원본/생성": "raw/generated",
    "원본 + 생성": "raw + generated",
    "원본 388": "raw 388",
    "생성 112": "generated 112",
    "원본 포화": "raw saturation",
    "원본 최종": "raw final",
    "원본 뜻 선택": "Raw Meaning Selection",
    "동의어 선택": "Synonym Selection",
    "의미 구분": "Sense Disambiguation",
    "반의어 선택": "Antonym Selection",
    "문맥 빈칸": "Context Cloze",
    "문맥": "Context",
    "동의어": "Synonym",
    "반의어": "Antonym",
    "출처 구성": "source mix",
    "정답 라벨": "hard label",
    "정답 label": "hard label",
    "정답 문자": "answer letter",
    "선택지": "option",
    "최댓값 선택": "argmax",
    "순위 답안": "ranked answer",
    "선택 답안": "selected answer",
    "선택됨": "selected",
    "평가 모드": "evaluation mode",
    "평가만 수행": "evaluation only",
    "평가만": "test-only",
    "평가 항목": "test items",
    "평가 행": "test rows",
    "학습/검증/평가": "train/dev/test",
    "학습/검증": "train/dev",
    "시드": "seed",
    "산출물": "artifact",
    "예측값": "prediction",
    "정책": "policy",
    "비용 탐색": "cost sweep",
    "경계선": "frontier",
    "경계": "boundary",
    "배포성": "deployability",
    "배포 후보": "deployment candidate",
    "장비 부담": "memory footprint",
}

GRAMMAR_REPLACEMENTS = {
    "확인-500": "Check-500",
    "No-API / No-Retrain 확인 결과와 raw final 실험 설계 의도 기반 방법론 대비": "No-API / No-Retrain 확인 결과와 원본 실험 설계 의도 기반 방법론 대비",
    "작은 local AI의 대형 LLM/API 대체 가능성": "작은 local AI로 대형 LLM/API를 대체할 수 있는가",
    "작은 로컬 AI의 대형 LLM/API 대체 가능성": "작은 local AI로 대형 LLM/API를 대체할 수 있는가",
    "최고 Accuracy": "Best Accuracy",
    "deployment candidate local": "Deployable local",
    "deployment candidate boundary": "deployment boundary",
    "배포 boundary": "deployment boundary",
    "No-API local 대체 가능성": "No-API local 대체 가능성",
    "로컬": "local",
    "행를": "행을",
    "행는": "행은",
    "세트은": "세트는",
    "세트을": "세트를",
    "세트으로": "세트로",
    "비교으로": "비교로",
    "형식가": "형식이",
    "형식를": "형식을",
    "option를": "option을",
    "contract을": "contract를",
    "answer contract을": "answer contract를",
    "reliability은": "Reliability는",
    "reliability이": "reliability가",
    "policy이": "policy가",
    "임계값와": "임계값과",
    "경계선를": "frontier를",
    "지표과": "지표와",
    "구간와": "slice와",
    "구간가": "slice가",
    "특징값를": "features를",
    "쌍를": "pair를",
    "손실를": "loss를",
    "분포이": "distribution이",
    "출력를": "출력을",
    "절충를": "절충을",
    "손실가": "loss가",
    "정책는": "policy는",
    "문맥는": "Context는",
    "반의어은": "Antonym은",
    "raw final 평가 평가": "raw final test",
    "raw final 평가": "raw final test",
    "raw final-평가": "raw final test",
    "원본/Synonym/Sense Disambiguation/Context": "Raw/Synonym/Sense/Context",
    "추가 어려운 평가": "추가 hard evaluation",
    "Accuracy-속도-memory footprint": "Accuracy-latency-memory footprint",
    "답변 시간": "latency",
    "runtime이": "Runtime이",
    "latency이": "latency가",
    "B4/No-API": "No B4/API",
    "API 호출 없음 / No-Retrain": "No-API call / No-Retrain",
    "local 실행": "local run",
    "No-API 확인": "No-API check",
    "KD soft 점수": "KD soft score",
    "teacher-점수": "teacher score",
    "judge-검증된": "judge-validated",
    "judge-validated는": "judge-validated는",
    "쌍대 Accuracy 차이": "paired Accuracy Delta",
    "쌍대 부트스트랩": "paired bootstrap",
    "paired 부트스트랩": "paired bootstrap",
    "hard slice과": "hard slice와",
    "경로 distribution는": "route distribution은",
    "distribution가": "distribution이",
    "완성 CE": "completion CE",
    "완성 영역": "completion span",
    "질의 대응용 근거 자료임을 전환합니다": "질의 대응용 근거 자료로 전환합니다",
    "대체 경로": "fallback",
    "주 경로": "primary route",
    "주 | 3.6%": "primary | 3.6%",
    "선택 gate: 낮음 0.75 / 높음 0.85": "Selected gate: low 0.75 / high 0.85",
    "공통 문항 비교": "common-item comparison",
    "쌍대 비교": "paired comparison",
    "쌍대": "paired",
    "어려운-구간": "hard-slice",
    "검증자가 검증한는": "judge-validated는",
    "검증자가 검증한 증강": "judge-validated augmentation",
    "답안-문자": "answer letter",
    "공통-문항": "common-item",
    "비-원본": "non-raw",
    "문맥 빈칸가": "Context Cloze가",
    "지연시간가": "latency가",
    "지연시간와": "latency와",
    "원본 최종-평가": "raw final test",
    "원본 최종 평가 평가": "raw final test",
    "실제 실제 경과시간": "실제 wall-clock",
    "파인튜닝 개선한다 reliability, 압축 개선한다 deployability": "fine-tuning improves reliability, compression improves deployability",
    "손실, 추론 방식": "loss와 inference 방식",
    "지표 산출 방식": "metric 산출 방식",
    "answer contract를 따르도록": "answer contract를 따르도록",
    "strict contract를 지키지 않아도": "strict contract를 지키지 않아도",
    "hard slice와 deployability": "hard slice와 deployability",
    "대체 경로 경로": "fallback route",
    "선택됨 policy": "selected policy",
    "낮음 임계값": "low threshold",
    "높음 임계값": "high threshold",
    "주 3.6%": "primary 3.6%",
    "어려운-slice": "hard-slice",
    "반의어은 n=14": "Antonym은 n=14",
}


def localize_text(value: str) -> str:
    text = str(value)
    for src, dst in sorted(TEXT_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(src, dst)
    for pattern, repl in REGEX_TEXT_REPLACEMENTS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    for src, dst in sorted(ASCII_TOKEN_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"(?<![A-Za-z]){re.escape(src)}(?![A-Za-z])", dst, text, flags=re.IGNORECASE)
    text = re.sub("claim", "해석", text, flags=re.IGNORECASE)
    text = re.sub("defense", "근거", text, flags=re.IGNORECASE)
    text = text.replace("방어", "설명")
    for src, dst in sorted(TECHNICAL_TERM_RESTORES.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(src, dst)
    for src, dst in sorted(GRAMMAR_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(src, dst)
    for src, dst in sorted(GRAMMAR_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(src, dst)
    return text


def sanitize_note_line(value: str) -> str | None:
    text = value.strip()
    if not text:
        return ""
    if re.match(r"^(질문|Q)\s*:", text, flags=re.IGNORECASE):
        return None
    text = re.sub(r"^(방어|A)\s*:\s*", "", text, flags=re.IGNORECASE)
    return localize_text(text)


@dataclass
class Element:
    xml: str
    rels: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class Slide:
    section: str
    number: str
    title: str
    core: str
    elements: list[Element]
    notes: list[str]
    image_rels: list[tuple[str, Path]] = field(default_factory=list)


class SlideBuilder:
    def __init__(self, slide_no: int, section: str, number: str, title: str, core: str) -> None:
        self.slide_no = slide_no
        self.section = section
        self.number = number
        self.title = title
        self.core = core
        self.notes: list[str] = []
        self.elements: list[Element] = []
        self.shape_id = 2
        self.image_rels: list[tuple[str, Path]] = []

    def next_id(self) -> int:
        current = self.shape_id
        self.shape_id += 1
        return current

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "FFFFFF",
        line: str | None = None,
        radius: bool = False,
        alpha: int | None = None,
    ) -> None:
        sid = self.next_id()
        fill_xml = f'<a:solidFill><a:srgbClr val="{fill}">' + (f'<a:alpha val="{alpha}"/>' if alpha is not None else "") + "</a:srgbClr></a:solidFill>"
        line_xml = '<a:ln w="0"><a:noFill/></a:ln>' if line is None else f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        prst = "roundRect" if radius else "rect"
        self.elements.append(
            Element(
                f"""
<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="rect-{sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
<a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}</p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr/></a:p></p:txBody></p:sp>
""".strip()
            )
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = "D9DEE5", width: int = 12700) -> None:
        sid = self.next_id()
        self.elements.append(
            Element(
                f"""
<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{sid}" name="line-{sid}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
<p:spPr><a:xfrm><a:off x="{emu(min(x1, x2))}" y="{emu(min(y1, y2))}"/><a:ext cx="{abs(emu(x2-x1))}" cy="{abs(emu(y2-y1))}"/></a:xfrm>
<a:prstGeom prst="line"><a:avLst/></a:prstGeom><a:ln w="{width}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln></p:spPr>
</p:cxnSp>
""".strip()
            )
        )

    def textbox(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        paragraphs: list[str],
        size: int = 16,
        color: str = "111827",
        bold: bool = False,
        fill: str | None = None,
        line: str | None = None,
        margin: float = 0.08,
        name: str = "text",
        align: str = "l",
        font: str = "Malgun Gothic",
        bullet: bool = False,
        line_spacing: int | None = None,
        localize: bool = True,
    ) -> None:
        sid = self.next_id()
        if fill is None:
            fill_xml = "<a:noFill/>"
        else:
            fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
        line_xml = '<a:ln w="0"><a:noFill/></a:ln>' if line is None else f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        body = []
        for text in paragraphs:
            if localize:
                text = localize_text(text)
            if text == "":
                body.append("<a:p><a:endParaRPr/></a:p>")
                continue
            lines = text.split("\n")
            for idx, line_text in enumerate(lines):
                display = f"• {line_text}" if bullet and line_text and not line_text.startswith("•") else line_text
                ppr = f'<a:pPr algn="{align}">'
                if line_spacing is not None:
                    ppr += f'<a:lnSpc><a:spcPct val="{line_spacing_pct(line_spacing)}"/></a:lnSpc>'
                ppr += "</a:pPr>"
                body.append(
                    f"""
<a:p>{ppr}<a:r><a:rPr lang="ko-KR" sz="{size * 100}" {'b="1"' if bold else ''}><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="{font}"/><a:ea typeface="{font}"/></a:rPr><a:t>{xml_text(display)}</a:t></a:r></a:p>
""".strip()
                )
                if idx != len(lines) - 1:
                    continue
        body_xml = "".join(body) if body else "<a:p><a:endParaRPr/></a:p>"
        self.elements.append(
            Element(
                f"""
<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{clean_shape_name(name)}-{sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}</p:spPr>
<p:txBody><a:bodyPr wrap="square" anchor="t" lIns="{emu(margin)}" rIns="{emu(margin)}" tIns="{emu(margin)}" bIns="{emu(margin)}"/><a:lstStyle/>{body_xml}</p:txBody></p:sp>
""".strip()
            )
        )

    def image(self, path: Path, x: float, y: float, w: float, h: float, name: str | None = None) -> None:
        sid = self.next_id()
        rel_id = f"rId{len(self.image_rels) + 3}"
        self.image_rels.append((rel_id, path))
        self.elements.append(
            Element(
                f"""
<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="{clean_shape_name(name or path.name)}" descr="{xml_text(path.name)}"/><p:cNvPicPr preferRelativeResize="0"><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
<p:blipFill rotWithShape="1"><a:blip r:embed="{rel_id}"><a:alphaModFix/></a:blip><a:srcRect b="0" l="0" r="0" t="0"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr></p:pic>
""".strip()
            )
        )

    def image_fit(self, path: Path, x: float, y: float, w: float, h: float, name: str | None = None) -> None:
        with Image.open(path) as img:
            iw, ih = img.size
        target_ratio = w / h
        image_ratio = iw / ih
        if image_ratio > target_ratio:
            fitted_w = w
            fitted_h = w / image_ratio
        else:
            fitted_h = h
            fitted_w = h * image_ratio
        self.image(path, x + (w - fitted_w) / 2, y + (h - fitted_h) / 2, fitted_w, fitted_h, name=name)

    def metric_card(self, x: float, y: float, w: float, title: str, value: str, note: str, color: str) -> None:
        self.rect(x, y, w, 1.03, fill="F7F9FB", line="D9DEE5", radius=True)
        self.textbox(x + 0.13, y + 0.09, w - 0.26, 0.22, [title], size=8, color="5F6B7A", bold=True, margin=0.0)
        self.textbox(x + 0.13, y + 0.31, w - 0.26, 0.36, [value], size=17, color=color, bold=True, margin=0.0)
        self.textbox(x + 0.13, y + 0.72, w - 0.26, 0.22, [note], size=8, color="5F6B7A", margin=0.0)

    def build(self, notes: list[str]) -> Slide:
        content_elements = self.elements
        self.elements = []

        # Full-slide background must be behind plots and body content. Google Slides'
        # importer respects XML order strictly, so adding this later hides images.
        self.rect(0.0, 0.0, 13.333, 7.5, fill="FFFFFF", line=None)
        background_elements = self.elements
        self.elements = []

        # Header and footer chrome should stay above body content.
        self.rect(0.0, 0.0, 0.08, 7.5, fill="1B8A8F", line=None)
        self.textbox(0.46, 0.28, 2.0, 0.24, [self.section.upper()], size=8, color="1B8A8F", bold=True, margin=0.0)
        self.textbox(12.34, 0.26, 0.55, 0.24, [self.number], size=8, color="8A8F98", bold=True, margin=0.0, align="r")
        self.textbox(0.46, 0.58, 8.5, 0.38, [self.title], size=20, color="0B1F33", bold=True, margin=0.0)
        self.textbox(0.46, 1.02, 11.8, 0.34, [self.core], size=10, color="5F6B7A", margin=0.0)
        self.line(0.46, 1.30, 12.85, 1.30, color="D9DEE5", width=6350)
        self.textbox(0.46, 7.16, 9.8, 0.18, ["TOEIC Business Vocabulary Judging | No-API / No-Retrain Result Report"], size=7, color="8A8F98", margin=0.0)
        self.textbox(12.25, 7.16, 0.6, 0.18, [self.number], size=7, color="8A8F98", margin=0.0, align="r")
        chrome_elements = self.elements
        self.elements = background_elements + content_elements + chrome_elements
        return Slide(self.section, self.number, self.title, self.core, self.elements, notes, list(self.image_rels))

    def build_plain(self, notes: list[str]) -> Slide:
        content_elements = self.elements
        self.elements = []
        self.rect(0.0, 0.0, 13.333, 7.5, fill="FFFFFF", line=None)
        background_elements = self.elements
        self.elements = background_elements + content_elements
        return Slide(self.section, self.number, self.title, self.core, self.elements, notes, list(self.image_rels))


def make_notes_xml(notes: list[str]) -> str:
    paras = []
    for block in notes:
        for line in block.split("\n"):
            line = sanitize_note_line(line)
            if line is None:
                continue
            if not line.strip():
                paras.append("<a:p><a:endParaRPr/></a:p>")
            else:
                paras.append(f'<a:p><a:r><a:t>{xml_text(line.strip())}</a:t></a:r></a:p>')
    paras_xml = "".join(paras) if paras else "<a:p><a:endParaRPr/></a:p>"
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notes xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr><p:sp><p:nvSpPr><p:cNvPr id="2" name="Slide Image Placeholder 1"/><p:cNvSpPr><a:spLocks noGrp="1" noRot="1" noChangeAspect="1"/></p:cNvSpPr><p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr><p:spPr/></p:sp><p:sp><p:nvSpPr><p:cNvPr id="3" name="Notes Placeholder 2"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>{paras_xml}</p:txBody></p:sp><p:sp><p:nvSpPr><p:cNvPr id="4" name="Slide Number Placeholder 3"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="sldNum" idx="5"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr/></a:p></p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>'''


def slide_xml(slide: Slide) -> str:
    body = "".join(e.xml for e in slide.elements)
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}"><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>{body}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'''


def rels_xml(relations: list[tuple[str, str, str]]) -> str:
    items = "".join(
        f'<Relationship Id="{rid}" Type="{rtype}" Target="{xml_text(target)}"/>' for rid, rtype, target in relations
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{REL_NS}">{items}</Relationships>'''


def presentation_xml(slide_count: int) -> bytes:
    pres = etree.Element(f"{{{P_NS}}}presentation", nsmap={"a": NS["a"], "r": R_NS, "p": P_NS})
    master_lst = etree.SubElement(pres, f"{{{P_NS}}}sldMasterIdLst")
    master_id = etree.SubElement(master_lst, f"{{{P_NS}}}sldMasterId")
    master_id.set("id", "2147483648")
    master_id.set(f"{{{R_NS}}}id", "rId1")
    notes_lst = etree.SubElement(pres, f"{{{P_NS}}}notesMasterIdLst")
    notes_id = etree.SubElement(notes_lst, f"{{{P_NS}}}notesMasterId")
    notes_id.set(f"{{{R_NS}}}id", "rId100")
    slide_lst = etree.SubElement(pres, f"{{{P_NS}}}sldIdLst")
    for i in range(1, slide_count + 1):
        sld = etree.SubElement(slide_lst, f"{{{P_NS}}}sldId")
        sld.set("id", str(255 + i))
        sld.set(f"{{{R_NS}}}id", f"rId{i+1}")
    etree.SubElement(pres, f"{{{P_NS}}}sldSz", cx=str(SLIDE_W), cy=str(SLIDE_H))
    etree.SubElement(pres, f"{{{P_NS}}}notesSz", cx="6858000", cy="9144000")
    return etree.tostring(pres, xml_declaration=True, encoding="UTF-8", standalone=True)


def content_types_xml(slide_count: int, media_exts: set[str]) -> str:
    defaults = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
    ]
    if "png" in media_exts:
        defaults.append('<Default Extension="png" ContentType="image/png"/>')
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/notesMasters/notesMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/ppt/theme/theme2.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>',
        '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>',
        '<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for i in range(1, slide_count + 1):
        overrides.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
        overrides.append(f'<Override PartName="/ppt/notesSlides/notesSlide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>')
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">{"".join(defaults + overrides)}</Types>'''


def app_xml(slide_count: int) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Term AI report generator</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{slide_count}</Slides><Notes>{slide_count}</Notes><Paragraphs>0</Paragraphs><Words>0</Words><Company></Company><AppVersion>16.0000</AppVersion></Properties>'''


def core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>TOEIC Vocabulary Judging Methodology Results</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>'''


def fit_plot(builder: SlideBuilder, plot: Path, x: float, y: float, w: float, h: float) -> None:
    builder.rect(x - 0.03, y - 0.03, w + 0.06, h + 0.06, fill="FFFFFF", line="D9DEE5", radius=True)
    builder.image_fit(plot, x, y, w, h, name=plot.stem)


def add_left_claim(builder: SlideBuilder, claims: list[str], accent: str = "1B8A8F") -> None:
    builder.rect(0.48, 1.55, 3.25, 4.85, fill="F7F9FB", line="D9DEE5", radius=True)
    builder.rect(0.48, 1.55, 0.10, 4.85, fill=accent, line=None)
    y = 1.78
    for idx, text in enumerate(claims, 1):
        builder.textbox(0.72, y, 0.35, 0.25, [f"{idx:02d}"], size=8, color=accent, bold=True, margin=0)
        builder.textbox(1.05, y - 0.01, 2.38, 0.55, [text], size=11, color="111827", margin=0, line_spacing=900)
        y += 0.92


def add_two_column_text(builder: SlideBuilder, left_title: str, left_items: list[str], right_title: str, right_items: list[str]) -> None:
    builder.rect(0.55, 1.55, 5.85, 4.95, fill="F7F9FB", line="D9DEE5", radius=True)
    builder.rect(6.85, 1.55, 5.85, 4.95, fill="F7F9FB", line="D9DEE5", radius=True)
    builder.textbox(0.82, 1.84, 5.3, 0.36, [left_title], size=14, color="0B1F33", bold=True, margin=0)
    builder.textbox(1.0, 2.35, 5.0, 3.7, left_items, size=11, color="111827", bullet=True, margin=0, line_spacing=900)
    builder.textbox(7.12, 1.84, 5.3, 0.36, [right_title], size=14, color="0B1F33", bold=True, margin=0)
    builder.textbox(7.30, 2.35, 5.0, 3.7, right_items, size=11, color="111827", bullet=True, margin=0, line_spacing=900)


def add_reference_table(builder: SlideBuilder, rows: list[tuple[str, str, str, str]], y: float = 1.58) -> None:
    cols = [
        ("Ref.", 0.62, 0.85),
        ("Reference item", 1.55, 3.15),
        ("Evidence class", 4.86, 3.25),
        ("Use in presentation", 8.28, 4.20),
    ]
    builder.rect(0.50, y, 12.20, 0.42, fill="F3F5F7", line="D9DEE5")
    for label, x, w in cols:
        builder.textbox(x, y + 0.10, w, 0.18, [label], size=8, color="1B8A8F", bold=True, margin=0)
    current_y = y + 0.48
    for idx, row in enumerate(rows):
        fill = "FFFFFF" if idx % 2 else "F9FAFB"
        builder.rect(0.50, current_y, 12.20, 0.64, fill=fill, line="E5E8EC")
        for (_, x, w), text in zip(cols, row, strict=True):
            color = "0B1F33" if x < 1.0 else "111827"
            builder.textbox(x, current_y + 0.10, w, 0.38, [text], size=7, color=color, bold=(x < 1.0), margin=0)
        current_y += 0.66


def add_arrow(builder: SlideBuilder, x1: float, y1: float, x2: float, y2: float, color: str = "8A8F98") -> None:
    builder.line(x1, y1, x2, y2, color=color, width=19050)
    direction = 1 if x2 >= x1 else -1
    builder.line(x2 - 0.10 * direction, y2 - 0.06, x2, y2, color=color, width=19050)
    builder.line(x2 - 0.10 * direction, y2 + 0.06, x2, y2, color=color, width=19050)


def add_arch_block(builder: SlideBuilder, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    builder.rect(x, y, w, h, fill="F7F9FB", line="D9DEE5", radius=True)
    builder.rect(x, y, 0.10, h, fill=color, line=None)
    builder.textbox(x + 0.22, y + 0.16, w - 0.36, 0.24, [title], size=10, color=color, bold=True, margin=0)
    builder.textbox(x + 0.22, y + 0.52, w - 0.36, h - 0.66, [body], size=8, color="111827", margin=0, line_spacing=950)


def add_model_adapter_diagram(builder: SlideBuilder) -> None:
    builder.textbox(0.62, 1.55, 3.2, 0.28, ["Discriminative path"], size=11, color="1B8A8F", bold=True, margin=0)
    add_arch_block(builder, 0.62, 1.95, 1.75, 0.86, "B0", "Embedding\nsimilarity", "1B8A8F")
    add_arrow(builder, 2.44, 2.38, 2.90, 2.38)
    add_arch_block(builder, 2.98, 1.95, 1.75, 0.86, "B2", "Embedding\n+ scorer reuse", "1B8A8F")
    add_arrow(builder, 4.80, 2.38, 5.26, 2.38)
    add_arch_block(builder, 5.34, 1.95, 1.95, 0.86, "B3", "Question-option\ncross-encoder", "1B8A8F")
    builder.textbox(7.64, 1.55, 4.1, 0.28, ["Local LM + adapter path"], size=11, color="3A9D5D", bold=True, margin=0)
    add_arch_block(builder, 7.64, 1.95, 1.85, 0.86, "Base LM", "Qwen 0.5B\n1.5B / 3B", "3A9D5D")
    add_arrow(builder, 9.56, 2.38, 10.02, 2.38)
    add_arch_block(builder, 10.10, 1.95, 1.95, 0.86, "Adapter", "Existing LoRA\nfinal adapter", "3A9D5D")
    add_arrow(builder, 12.10, 2.38, 12.50, 2.38)
    add_arch_block(builder, 0.62, 3.50, 2.05, 0.92, "G3", "3B + adapter\nquality ceiling", "C84A3A")
    add_arch_block(builder, 3.02, 3.50, 2.05, 0.92, "G5", "0.5B / 1.5B\nsmall local LM", "3A9D5D")
    add_arch_block(builder, 5.42, 3.50, 2.05, 0.92, "G4", "3B adapter\n4bit loading", "7F63B8")
    add_arch_block(builder, 8.10, 3.50, 4.05, 0.92, "H1", "B0 confidence → B3 cross-encoder\n→ G5 fallback", "E79A32")
    builder.line(0.62, 5.04, 12.30, 5.04, color="D9DEE5", width=6350)
    builder.textbox(0.72, 5.35, 11.4, 0.52, [
        "[R10-R14] 모델 계열은 baseline, local LM, adapter, quantized loading, routing policy로 구분한다."
    ], size=12, color="5F6B7A", margin=0)


def add_evidence_plot_card(
    builder: SlideBuilder,
    plot: Path,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    tag: str,
    accent: str,
) -> None:
    builder.rect(x, y, w, h, fill="FFFFFF", line="D9DEE5", radius=True)
    builder.rect(x, y, 0.09, h, fill=accent, line=None)
    builder.textbox(x + 0.20, y + 0.13, w - 0.35, 0.22, [title], size=10, color="0B1F33", bold=True, margin=0)
    builder.textbox(x + 0.20, y + 0.40, w - 0.35, 0.18, [tag], size=7, color=accent, bold=True, margin=0)
    builder.image_fit(plot, x + 0.22, y + 0.70, w - 0.44, h - 0.92, name=plot.stem)


def add_metric_chip(builder: SlideBuilder, x: float, y: float, label: str, value: str, color: str, w: float = 0.96) -> None:
    builder.rect(x, y, w, 0.34, fill="F7F9FB", line="D9DEE5", radius=True)
    builder.textbox(x + 0.08, y + 0.06, w - 0.16, 0.10, [label], size=5, color="5F6B7A", bold=True, margin=0, align="c")
    builder.textbox(x + 0.08, y + 0.17, w - 0.16, 0.12, [value], size=7, color=color, bold=True, margin=0, align="c")


def add_flow_card(
    builder: SlideBuilder,
    x: float,
    y: float,
    w: float,
    h: float,
    model: str,
    role: str,
    nodes: list[str],
    metrics: list[tuple[str, str]],
    accent: str,
) -> None:
    builder.rect(x, y, w, h, fill="FFFFFF", line="D9DEE5", radius=True)
    builder.rect(x, y, 0.10, h, fill=accent, line=None)
    builder.textbox(x + 0.22, y + 0.15, 0.82, 0.28, [model], size=12, color=accent, bold=True, margin=0)
    builder.textbox(x + 1.03, y + 0.17, w - 1.25, 0.22, [role], size=8, color="0B1F33", bold=True, margin=0)
    chip_x = x + w - 0.20 - len(metrics) * 0.88
    for idx, (label, value) in enumerate(metrics):
        add_metric_chip(builder, chip_x + idx * 0.90, y + 0.52, label, value, accent, w=0.80)
    flow_y = y + 1.13
    available_w = w - 0.56
    gap = 0.18
    node_w = (available_w - gap * (len(nodes) - 1)) / len(nodes)
    for idx, node in enumerate(nodes):
        nx = x + 0.28 + idx * (node_w + gap)
        builder.rect(nx, flow_y, node_w, 0.62, fill="F7F9FB", line="D9DEE5", radius=True)
        builder.textbox(nx + 0.08, flow_y + 0.14, node_w - 0.16, 0.25, [node], size=7, color="111827", bold=True, margin=0, align="c")
        if idx < len(nodes) - 1:
            add_arrow(builder, nx + node_w + 0.03, flow_y + 0.31, nx + node_w + gap - 0.04, flow_y + 0.31, color="8A8F98")


def add_distribution_bar(
    builder: SlideBuilder,
    x: float,
    y: float,
    label: str,
    value: int,
    total: int,
    color: str,
    max_w: float = 4.25,
) -> None:
    width = max_w * (value / total if total else 0)
    builder.textbox(x, y, 1.70, 0.18, [label], size=8, color="0B1F33", bold=True, margin=0)
    builder.rect(x + 1.85, y + 0.02, max_w, 0.18, fill="F3F5F7", line="E5E8EC", radius=True)
    builder.rect(x + 1.85, y + 0.02, max(0.04, width), 0.18, fill=color, line=None, radius=True)
    builder.textbox(x + 6.20, y, 0.55, 0.18, [str(value)], size=8, color="5F6B7A", bold=True, margin=0, align="r")


def add_compact_bar(
    builder: SlideBuilder,
    x: float,
    y: float,
    label: str,
    value: int | float,
    total: int | float,
    color: str,
    max_w: float = 1.85,
    value_text: str | None = None,
) -> None:
    width = max_w * (float(value) / float(total) if total else 0.0)
    builder.textbox(x, y, 1.35, 0.16, [label], size=7, color="0B1F33", bold=True, margin=0)
    builder.rect(x + 1.48, y + 0.02, max_w, 0.16, fill="F3F5F7", line="E5E8EC", radius=True)
    builder.rect(x + 1.48, y + 0.02, max(0.035, width), 0.16, fill=color, line=None, radius=True)
    builder.textbox(x + 1.58 + max_w, y, 0.70, 0.16, [value_text or str(value)], size=7, color="5F6B7A", bold=True, margin=0)


def add_count_card(builder: SlideBuilder, x: float, y: float, title: str, value: str, note: str, accent: str) -> None:
    builder.rect(x, y, 2.25, 0.98, fill="F7F9FB", line="D9DEE5", radius=True)
    builder.textbox(x + 0.15, y + 0.13, 1.95, 0.18, [title], size=7, color="5F6B7A", bold=True, margin=0)
    builder.textbox(x + 0.15, y + 0.36, 1.95, 0.28, [value], size=16, color=accent, bold=True, margin=0)
    builder.textbox(x + 0.15, y + 0.72, 1.95, 0.15, [note], size=6, color="5F6B7A", margin=0)


def add_defense_card(
    builder: SlideBuilder,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: list[str],
    accent: str,
    foot: str | None = None,
) -> None:
    builder.rect(x, y, w, h, fill="FFFFFF", line="D9DEE5", radius=True)
    builder.rect(x, y, 0.09, h, fill=accent, line=None)
    builder.textbox(x + 0.20, y + 0.15, w - 0.35, 0.22, [title], size=10, color=accent, bold=True, margin=0)
    body_h = h - (0.78 if foot else 0.48)
    builder.textbox(x + 0.25, y + 0.52, w - 0.48, body_h, body, size=8, color="111827", bullet=True, margin=0, line_spacing=950)
    if foot:
        builder.textbox(x + 0.25, y + h - 0.36, w - 0.48, 0.18, [foot], size=7, color="5F6B7A", bold=True, margin=0)


def add_metric_matrix(
    builder: SlideBuilder,
    rows: list[tuple[str, list[str], str]],
    headers: list[str],
    x: float,
    y: float,
    row_h: float = 0.46,
) -> None:
    widths = [1.45, 1.05, 1.05, 1.05, 1.20, 1.15, 1.30]
    builder.rect(x, y, sum(widths), 0.42, fill="F3F5F7", line="D9DEE5")
    cx = x
    for idx, header in enumerate(headers):
        builder.textbox(cx + 0.08, y + 0.11, widths[idx] - 0.16, 0.16, [header], size=7, color="1B8A8F", bold=True, margin=0, align="c" if idx else "l")
        cx += widths[idx]
    cy = y + 0.46
    for row_idx, (model, values, accent) in enumerate(rows):
        fill = "FFFFFF" if row_idx % 2 else "F9FAFB"
        builder.rect(x, cy, sum(widths), row_h, fill=fill, line="E5E8EC")
        cx = x
        builder.textbox(cx + 0.08, cy + 0.12, widths[0] - 0.16, 0.16, [model], size=7, color=accent, bold=True, margin=0)
        cx += widths[0]
        for idx, value in enumerate(values, start=1):
            builder.textbox(cx + 0.06, cy + 0.12, widths[idx] - 0.12, 0.16, [value], size=7, color="111827", bold=(idx == 1), margin=0, align="c")
            cx += widths[idx]
        cy += row_h


def add_design_matrix(
    builder: SlideBuilder,
    rows: list[tuple[str, str, str, str, str]],
    headers: list[str],
    x: float,
    y: float,
    widths: list[float],
    row_h: float = 0.58,
) -> None:
    total_w = sum(widths)
    builder.rect(x, y, total_w, 0.42, fill="F3F5F7", line="D9DEE5")
    cx = x
    for idx, header in enumerate(headers):
        builder.textbox(cx + 0.08, y + 0.10, widths[idx] - 0.16, 0.18, [header], size=7, color="1B8A8F", bold=True, margin=0)
        cx += widths[idx]
    cy = y + 0.46
    for row_idx, row in enumerate(rows):
        fill = "FFFFFF" if row_idx % 2 else "F9FAFB"
        builder.rect(x, cy, total_w, row_h, fill=fill, line="E5E8EC")
        cx = x
        for idx, text in enumerate(row):
            color = "0B1F33" if idx == 0 else "111827"
            builder.textbox(cx + 0.08, cy + 0.11, widths[idx] - 0.16, row_h - 0.18, [text], size=6, color=color, bold=(idx == 0), margin=0, line_spacing=950)
            cx += widths[idx]
        cy += row_h


def add_pipeline_node(builder: SlideBuilder, x: float, y: float, w: float, title: str, body: str, color: str) -> None:
    builder.rect(x, y, w, 1.05, fill="F7F9FB", line="D9DEE5", radius=True)
    builder.rect(x, y, 0.08, 1.05, fill=color, line=None)
    builder.textbox(x + 0.18, y + 0.17, w - 0.30, 0.20, [title], size=9, color=color, bold=True, margin=0)
    builder.textbox(x + 0.18, y + 0.50, w - 0.30, 0.36, [body], size=7, color="111827", margin=0, line_spacing=950)


def metric_summary(check_root: Path) -> dict[str, dict[str, Any]]:
    return {path.parent.name: read_json(path) for path in check_root.glob("*/metric_log.json")}


def paired_result(check_root: Path, name: str) -> dict[str, Any]:
    report = check_root / "reports" / f"{name}.json"
    return read_json(report) if report.exists() else {}


def eval_subset_summary(check_root: Path) -> dict[str, Any]:
    meta_path = check_root / "_inputs" / "eval_metadata.jsonl"
    if not meta_path.exists():
        return {"split": {}, "status": {}, "task": {}, "answer": {}, "generated_task_answer": {}, "test_n": 0, "all_n": 0}
    rows = read_jsonl(meta_path)
    test_rows = [row for row in rows if row.get("split") == "test"]
    task_counter: Counter[str] = Counter()
    answer_counter: Counter[str] = Counter()
    generated_task_answer: dict[str, Counter[str]] = {}
    for row in test_rows:
        payload = row.get("payload") or {}
        task = payload.get("source_task_type") or payload.get("task_type") or row.get("task_type") or "unknown"
        label = label_from_idx(payload.get("answer_idx", row.get("answer_idx")))
        task_counter[task] += 1
        answer_counter[label] += 1
        if row.get("status") == "aug_judge_pass":
            generated_task_answer.setdefault(task, Counter())[label] += 1
    return {
        "all_n": len(rows),
        "test_n": len(test_rows),
        "split": Counter(row.get("split") or "unknown" for row in rows),
        "status": Counter(row.get("status") or "unknown" for row in test_rows),
        "task": task_counter,
        "answer": answer_counter,
        "generated_task_answer": generated_task_answer,
    }


def sample_cards(root: Path) -> list[dict[str, Any]]:
    ids = ["84f54c4860976be0", "eae4c43e69e885df", "8dee9d1bf7f93456", "90e14c3f85290964"]
    samples = []
    meta_path = root / "_inputs" / "eval_metadata.jsonl"
    if not meta_path.exists():
        return samples
    wanted = set(ids)
    for row in read_jsonl(meta_path):
        if row.get("item_id") in wanted:
            payload = row.get("payload") or {}
            samples.append(
                {
                    "id": row.get("item_id"),
                    "task": payload.get("task_type") or payload.get("source_task_type"),
                    "word": payload.get("word"),
                    "context": payload.get("context"),
                    "options": payload.get("options"),
                    "answer": "ABCD"[int(payload.get("answer_idx", 0))],
                    "rationale": payload.get("rationale") or row.get("teacher_rationale"),
                }
            )
    return samples


def build_slides(check_root: Path, plots_dir: Path) -> list[Slide]:
    m = metric_summary(check_root)
    b0, b2, b3 = m["B0"], m["B2"], m["B3"]
    g05, g15, g3 = m["G5_Qwen0p5_G1"], m["G5_Qwen1p5_G1"], m["G3_Qwen"]
    g4, h1 = m["G4_Qwen_4bit"], m["H1_local_hybrid"]
    samples = sample_cards(check_root)
    subset = eval_subset_summary(check_root)
    slides: list[Slide] = []

    def make(section: str, title: str, core: str, notes: list[str]) -> SlideBuilder:
        builder = SlideBuilder(len(slides) + 1, section, f"{len(slides)+1:02d}", title, core)
        builder.notes = notes
        return builder

    # 1 Cover
    b = make("EXPERIMENT RESULTS", "Can Small / Local AI Replace Large LLM or API for TOEIC Vocabulary Judging?", "설계 질문 재구성 | No-API local 대체 가능성 | 결과 중심", [
        "도입에서는 '비싼 큰 AI를 작고 빠른 내부 모델로 어디까지 대체할 수 있는가'라는 실험 질문을 먼저 제시합니다.",
        "원본 계획 PPTX의 핵심은 Small LM, KD, Quantization, Hybrid를 통해 품질과 운영 비용 사이의 경계를 찾는 것이었습니다. 이번 덱은 그 설계 의도를 결과 중심으로 다시 정렬한 발표본입니다.",
        "주의할 점은 최고 정답률만 고르는 발표가 아니라는 것입니다. 정답률, 답변 시간, 장비 부담, 답안 형식 안정성, hard case에서의 차이를 함께 봅니다.",
    ])
    b.textbox(0.28, 0.28, 1.10, 0.16, ["실험 결과"], size=7, color="1B8A8F", bold=True, margin=0, localize=False)
    b.textbox(12.43, 0.28, 0.40, 0.16, ["01"], size=7, color="6F747D", bold=True, margin=0, align="r", localize=False)
    b.textbox(0.28, 0.61, 8.70, 0.34, ["작은 로컬 AI의 대형 LLM/API 대체 가능성"], size=20, color="0B1F33", bold=True, margin=0, localize=False)
    b.textbox(0.28, 1.05, 8.20, 0.18, ["설계 질문 재구성 | API 없음 로컬 대체 가능성 | 결과 중심"], size=8, color="5F6B7A", margin=0, localize=False)
    b.line(0.28, 1.30, 12.85, 1.30, color="D9DEE5", width=6350)
    b.rect(4.82, 3.12, 7.05, 1.10, fill="FFFFFF", line="6F747D", radius=False)
    b.textbox(4.86, 3.28, 6.72, 0.56, ["실험 발표 초안"], size=30, color="0B1F33", bold=True, margin=0, localize=False)
    b.textbox(0.28, 7.16, 6.20, 0.16, ["TOEIC 비즈니스 어휘 판정 | API 없음 / 재학습 없음 결과 보고"], size=6, color="8A8F98", margin=0, localize=False)
    b.textbox(12.43, 7.16, 0.40, 0.16, ["01"], size=6, color="8A8F98", margin=0, align="r", localize=False)
    slides.append(b.build_plain(["발표 시작에서는 '정답률이 제일 높은 모델은 무엇인가'가 아니라 '큰 모델/API를 줄일 수 있는가'가 핵심 질문이라고 못박습니다. 이후 슬라이드의 모든 그래프는 이 질문을 기준으로 해석합니다."]))

    # 2 Experiment stages
    b = make("EXPERIMENT DESIGN", "B/G/H Experiment Stages and Intent", "실험군 단계 | 기준선-상한-압축-라우팅 | 대체 가능 경계", [
        "이 장에서는 이후 결과표에 등장하는 B0, B2, B3, G3, G4, G5, H1이 각각 무엇을 검증하는지 먼저 정리합니다.",
        "B 계열은 생성형 모델을 쓰기 전에 판별형 채점 구조만으로 어디까지 가능한지 확인하는 단계입니다. B0는 embedding similarity, B2는 기존 MLP scorer, B3는 cross-encoder pairwise reading입니다.",
        "G 계열은 local LM이 직접 문제를 풀 수 있는지, 그리고 큰 모델을 더 작거나 압축된 모델로 대체할 수 있는지 확인하는 단계입니다. H1은 쉬운 문제와 어려운 문제를 나누어 처리하는 운영 정책 실험입니다.",
        "읽는 순서는 B0/B2로 단순 기준선을 만들고, B3로 판별형 한계를 확인한 뒤, G3/G4/G5와 H1에서 local LM, 압축, 라우팅의 대체 가능성을 보는 흐름입니다.",
    ])
    b.textbox(0.70, 1.55, 12.0, 0.35, ["실험군별 의도"], size=15, color="0B1F33", bold=True, margin=0)
    rows = [
        ("B0", "Embedding baseline", "뜻/문장 벡터의 가까움만으로 정답을 고를 수 있는가", "가장 싼 기준선"),
        ("B2", "MLP scorer reuse", "embedding feature 위에 기존 scorer를 얹으면 단순 기준선을 넘는가", "재학습 없이 scorer 재사용"),
        ("B3", "Cross-encoder", "문제와 선택지를 함께 읽으면 판별형 모델만으로 충분한가", "첫 번째 성능 점프"),
        ("G3", "3B local LM", "큰 local LM이 API 없이 품질 상한을 만들 수 있는가", "품질 상한 확인"),
        ("G4", "4bit compressed", "3B 경로를 압축해 runtime을 줄일 수 있는가", "압축/배포성 확인"),
        ("G5", "0.5B / 1.5B local LM", "작은 local LM이 큰 모델을 실용적으로 대체할 수 있는가", "대체 가능 구간 확인"),
        ("H1", "Local hybrid", "confidence에 따라 B/G 경로를 나눠 비용-품질 균형을 만들 수 있는가", "운영 정책 후보"),
    ]
    y = 2.05
    for idx, (run, label, intent, role) in enumerate(rows):
        fill = "F7F9FB" if idx % 2 == 0 else "FFFFFF"
        b.rect(0.70, y, 11.90, 0.50, fill=fill, line="D9DEE5", radius=False)
        b.textbox(0.92, y + 0.12, 0.60, 0.18, [run], size=10, color="1B8A8F", bold=True, margin=0)
        b.textbox(1.65, y + 0.11, 2.10, 0.20, [label], size=9, color="0B1F33", bold=True, margin=0)
        b.textbox(3.95, y + 0.11, 5.40, 0.20, [intent], size=9, color="111827", margin=0)
        b.textbox(9.70, y + 0.11, 2.55, 0.20, [role], size=9, color="5F6B7A", margin=0)
        y += 0.55
    slides.append(b.build(b.notes))

    # 3 Original plan alignment
    b = make("ORIGINAL PLAN", "Original PPTX Intent → Result Reading Frame", "네 가지 설계축 | Discriminative-LM-Compression-Hybrid | 결과 해석 프레임", [
        "원본 계획은 discriminative baseline, small LM, compression, hybrid fallback이라는 네 축을 제안했습니다. 결과 발표에서도 이 순서를 유지해야 설계 의도가 살아납니다.",
        "다만 결과를 보면 raw final test가 매우 쉬워 상위 모델들이 포화됩니다. 그래서 check-500 mixed/hard set을 함께 보면서 방법론 차이를 드러내는 식으로 접근합니다.",
        "이 슬라이드는 이후 결과를 단순 모델 나열이 아니라 방법론 단계의 누적 검증으로 읽기 위한 기준을 제공합니다.",
        "결과 발표의 중심축은 정답률 1등이 아니라 큰 모델/API 의존을 줄이는 경계입니다.",
    ])
    b.textbox(0.70, 1.55, 12.0, 0.42, ["원본 설계의 네 질문"], size=15, color="0B1F33", bold=True, margin=0)
    cards = [
        ("1", "Discriminative baseline", "embedding만으로 충분한가?", "B0/B2/B3"),
        ("2", "Small LM path", "작은 AI가 문제를 직접 풀 수 있는가?", "G3/G5"),
        ("3", "Compression path", "품질 유지하며 더 가볍게 만들 수 있는가?", "G4/G5"),
        ("4", "Hybrid fallback", "쉬운 문제와 어려운 문제를 나눌 수 있는가?", "H1"),
    ]
    for i, (num, title, q, run) in enumerate(cards):
        x = 0.70 + i * 3.05
        b.rect(x, 2.15, 2.72, 2.55, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.18, 2.35, 0.35, 0.28, [num], size=13, color="1B8A8F", bold=True, margin=0)
        b.textbox(x + 0.55, 2.35, 1.95, 0.52, [title], size=12, color="0B1F33", bold=True, margin=0)
        b.textbox(x + 0.18, 3.08, 2.25, 0.72, [q], size=11, color="111827", margin=0)
        b.textbox(x + 0.18, 4.08, 2.25, 0.25, [run], size=9, color="5F6B7A", bold=True, margin=0)
    slides.append(b.build(b.notes))

    # 4 final vs check
    b = make("EVALUATION FRAME", "Raw Final Test vs Check-500", "Raw final 포화 | Check-500 mixed/hard slice | 방법론 대비", [
        "이 그래프는 왜 추가 check-500을 봐야 하는지를 설명합니다. raw final test에서는 B3, G3, G5가 대부분 99% 이상으로 포화되어 보입니다. 이 상태에서는 어떤 방법론이 더 낫고 어디에서 깨지는지 잘 보이지 않습니다.",
        "check-500은 train/dev rows를 유지하면서 test 500개를 뽑은 확인용 set입니다. generated/non-raw rows를 우선 포함하고 나머지를 raw test에서 seed 기반으로 채웠습니다. 그래서 raw meaning뿐 아니라 synonym, antonym, sense, cloze가 섞입니다.",
        "이 set은 새로운 최종 성능표가 아니라 방법론 대비를 드러내기 위한 확인용 mixed/hard slice입니다.",
    ])
    add_left_claim(b, ["기존 final raw test는 상위 모델이 포화", "check-500은 방법론 차이를 더 잘 드러냄", "두 set의 목적이 다르므로 직접 일반화는 조심"], "E79A32")
    fit_plot(b, plots_dir / "13_final_vs_check500_accuracy.png", 4.05, 1.55, 8.35, 4.9)
    slides.append(b.build(b.notes))

    # 5 accuracy methodology
    b = make("RESULT 1", "Embedding-only Baseline Breaks", "Embedding similarity 한계 | Cross-encoder 점프 | 판별 방식 변화", [
        "첫 번째 결과는 가장 직관적입니다. B0는 뜻이 비슷한 선택지를 고르는 방식이고, B2는 그 위에 작은 학습형 점수 계산기를 얹은 방식입니다. 둘 다 60% 전후에 머뭅니다.",
        "반면 B3는 문제와 선택지를 함께 읽고 비교합니다. 같은 discriminative 계열 안에서도 문제를 읽는 방식이 바뀌면 97% 이상으로 올라갑니다.",
        "따라서 첫 번째 결론은 '무조건 큰 AI가 필요하다'가 아닙니다. 먼저 채점 구조 자체가 embedding similarity에서 pairwise comparison으로 바뀌어야 합니다.",
    ])
    add_left_claim(b, [f"B0: {pct(b0['accuracy'])}, {int(b0['n']*(1-b0['accuracy']))} errors", f"B2: {pct(b2['accuracy'])}, 학습 scorer도 한계", f"B3: {pct(b3['accuracy'])}, +34.4pp leap", "첫 극적 대비는 모델 크기가 아니라 읽는 방식"], "1B8A8F")
    fit_plot(b, plots_dir / "01_check500_methodology_accuracy.png", 4.05, 1.55, 8.35, 4.9)
    slides.append(b.build(b.notes))

    # 7 stage ladder
    b = make("RESULT 1", "Methodological Stage Ladder", "Similarity → Pairwise reading → Local LM → Compression", [
        "이 슬라이드는 전체 발표의 backbone입니다. 왼쪽에서 오른쪽으로 갈수록 문제 해결 방식이 바뀝니다. 단순 유사도, embedding classifier, pairwise reranking, 작은 local student, 3B compressed runtime, 3B 품질 상한으로 이어집니다.",
        "가장 큰 점프는 B0/B2에서 B3로 넘어갈 때입니다. 그 뒤 상위 모델들은 96~99%대에서 경쟁합니다. 여기서는 평균 정답률보다 latency, memory, hard case, output contract가 더 중요해집니다.",
        "이 사다리는 이후 accuracy, latency, hard slice, output contract 결과를 같은 축 위에서 해석하기 위한 기준입니다.",
    ])
    fit_plot(b, plots_dir / "05_check500_methodology_stage_ladder.png", 0.70, 1.50, 11.95, 5.10)
    slides.append(b.build(b.notes))

    # 8 task heatmap
    b = make("RESULT 2", "Average Accuracy Hides Hard Slices", "평균 정확도 포화 | Antonym hard slice | 모델별 취약점", [
        "이 heatmap은 평균 정답률만 보면 놓치는 지점을 보여줍니다. Raw, Synonym, Sense, Context는 강한 모델들이 대부분 잘 맞힙니다. 그래서 평균에서는 97%대 모델들이 비슷해 보입니다.",
        "하지만 Antonym Selection 열을 보면 거의 모든 모델이 크게 흔들립니다. 이 유형은 단순히 비슷한 뜻을 찾는 것이 아니라 반대 관계를 판단해야 해서 모델의 이해 방식 차이가 더 크게 드러납니다.",
        "따라서 발표에서 '우리 모델은 99%입니다'라고 끝내면 설계 의도가 약합니다. '쉬운 평균이 아니라 어려운 slice에서 어떤 모델이 버티는가'가 더 설득력 있는 포인트입니다.",
    ])
    add_left_claim(b, ["Raw/Synonym/Sense/Context는 대부분 포화", "Antonym은 모델별 편차가 가장 큼", "Hard slice가 방법론 차이를 설명", "추가 hard eval 설계의 근거"], "C84A3A")
    fit_plot(b, plots_dir / "03_check500_task_slice_heatmap.png", 4.05, 1.55, 8.35, 4.9)
    slides.append(b.build(b.notes))

    # 9 antonym
    b = make("RESULT 2", "Antonym Slice Is the Real Stress Test", "Antonym stress test | Hard slice 격차 | G3 품질 상한", [
        "이 슬라이드는 발표에서 가장 드라마틱한 장면으로 쓸 수 있습니다. 전체 accuracy는 B3, G5, G4, G3가 모두 높지만, Antonym Selection만 보면 격차가 크게 벌어집니다.",
        "G3_Qwen은 14개 중 11개를 맞힌 반면, B3와 G4는 2개, G5 0.5B는 1개 수준입니다. 표본이 14개로 작다는 단서는 반드시 붙여야 하지만, hard set을 추가해야 하는 이유를 매우 잘 보여줍니다.",
        "이 결과는 평균 accuracy가 포화된 뒤에도 hard slice에서는 모델 용량과 학습 방식의 차이가 남는다는 근거입니다.",
    ])
    fit_plot(b, plots_dir / "04_check500_antonym_hard_slice.png", 0.70, 1.50, 8.35, 5.05)
    b.rect(9.35, 1.65, 3.1, 4.8, fill="FFF8F2", line="F2D0A5", radius=True)
    b.textbox(9.62, 1.90, 2.55, 0.32, ["예시: invoice"], size=14, color="0B1F33", bold=True, margin=0)
    b.textbox(9.62, 2.35, 2.55, 2.45, [
        "문장: The supplier agreed to invoice the client...",
        "반대 관계: invoice ↔ pay",
        "혼동 후보: bill / charge / collect",
        "왜 어려운가: bill·charge는 invoice와 매우 가깝고, collect도 결제 맥락에 있어 헷갈린다.",
    ], size=10, color="111827", margin=0, line_spacing=900)
    b.textbox(9.62, 5.18, 2.55, 0.72, ["해석: 단순 유사도나 선택지 점수화는 '반대 관계'를 잘 못 잡을 수 있다."], size=10, color="C84A3A", bold=True, margin=0)
    slides.append(b.build(b.notes))

    # 10 frontier
    b = make("RESULT 3", "Accuracy vs Latency vs Memory Frontier", "정확도 | 답변 시간 | 장비 부담", [
        "여기서부터는 제품 관점입니다. G3는 가장 정확하지만 오른쪽 위에 있습니다. 즉 품질은 좋지만 느리고 무겁습니다.",
        "G5 0.5B와 1.5B는 정확도가 조금 낮거나 비슷하면서도 훨씬 왼쪽에 있습니다. 이는 답변 시간이 짧다는 뜻입니다. 특히 0.5B는 장비 부담도 작습니다.",
        "발표 핵심은 '3B가 제일 좋다'가 아닙니다. bounded vocabulary judging에서는 작은 내부 AI가 큰 모델을 상당 부분 대체할 수 있는 실용 구간이 보인다는 것입니다.",
    ])
    add_left_claim(b, [f"G3: {pct(g3['accuracy'])}, {fmt_ms(g3['latency_p50'])}", f"G5 1.5B: {pct(g15['accuracy'])}, {fmt_ms(g15['latency_p50'])}", f"G5 0.5B: {pct(g05['accuracy'])}, {fmt_mb(g05['peak_VRAM_or_RAM'])}", "정확도-속도-장비 부담을 동시에 비교"], "3A9D5D")
    fit_plot(b, plots_dir / "02_check500_accuracy_latency_memory_frontier.png", 4.05, 1.55, 8.35, 4.9)
    slides.append(b.build(b.notes))

    # 11 runtime
    b = make("RESULT 3", "Runtime Cost Makes the Deployment Story", "500개 실행 시간 | Local runtime 차이 | 운영 비용", [
        "이 그래프는 각 모델의 p50 latency를 500개에 곱해 대략적인 실행 시간을 보여줍니다. 실제 wall-clock과 완전히 같지는 않지만 운영 감각을 전달하기 좋습니다.",
        "B0, B2, B3는 초 단위입니다. G5 0.5B와 1.5B는 몇 분 단위입니다. G4와 G3는 15~21분 수준으로 올라갑니다.",
        "이 결과는 작은 accuracy gain이 큰 runtime 증가를 동반할 수 있음을 보여주므로, 운영 후보 선택에서 latency를 함께 봐야 합니다.",
    ])
    fit_plot(b, plots_dir / "14_check500_runtime_cost_by_method.png", 0.70, 1.50, 11.95, 5.1)
    slides.append(b.build(b.notes))

    # 12 reliability
    b = make("RESULT 4", "Reliability Is Not the Same as Accuracy", "Accuracy | Calibration | Output contract", [
        "이 슬라이드는 제품 자동화에서 매우 중요합니다. G3는 정답률과 calibration은 좋지만 strict parse error가 높습니다. 즉 답은 맞히지만 우리가 요구한 답안 형식을 자주 엄격히 지키지 못합니다.",
        "반대로 G5 0.5B는 정답률은 G3보다 낮지만 strict parse error가 0입니다. 작은 모델이 더 제품 친화적인 출력 습관을 가질 수 있다는 점이 보입니다.",
        "이 결과는 자동 채점 경로에서는 정답률뿐 아니라 output contract 준수가 별도 품질 요건임을 보여줍니다.",
    ])
    add_left_claim(b, ["Accuracy: 맞혔는가", "ECE: 자신감이 믿을 만한가", "Strict parse error: 답안 형식을 지켰는가", "서비스 채점기는 세 지표를 함께 봐야 함"], "7F63B8")
    fit_plot(b, plots_dir / "07_check500_calibration_contract.png", 4.05, 1.55, 8.35, 4.9)
    slides.append(b.build(b.notes))

    # 13 hybrid
    b = make("RESULT 5", "Hybrid Routing Is a Policy Experiment", "Confidence routing | Local fallback | 비용-품질 정책", [
        "Hybrid는 쉬운 문제는 싼 방식으로 처리하고 어려운 문제만 더 똑똑한 모델로 보내는 전략입니다. 이 접근은 운영 비용을 줄이는 데 중요합니다.",
        "하지만 현재 결과를 보면 94.4%가 cross-encoder로 갔습니다. 즉 성능은 좋지만 아직 '대부분을 싼 방식으로 처리했다'고 말하기는 어렵습니다.",
        "따라서 발표에서는 hybrid를 완성된 최적 정책이 아니라 '다음 단계에서 confidence threshold와 cost grid를 조정해 frontier를 만들 대상'이라고 설명해야 안전합니다.",
    ])
    fit_plot(b, plots_dir / "08_check500_hybrid_routing.png", 0.70, 1.50, 11.95, 5.1)
    slides.append(b.build(b.notes))

    # 14 paired stats
    b = make("STATISTICS", "Paired Statistics: Which Gains Are Real?", "Paired comparison | B0 대비 점프 | 상위권 차이 검정", [
        "이 장은 통계적 근거를 제시하는 본문 슬라이드입니다. B0에서 B3나 G5로 올라가는 차이는 30%p 이상이며 통계적으로 매우 뚜렷합니다.",
        "반면 B3와 G5 0.5B 사이 차이는 유의하지 않습니다. 이 말은 'G5 0.5B가 무조건 B3보다 낫다'가 아니라, 이 check-500 범위에서는 둘의 평균 accuracy 차이가 작다는 뜻입니다.",
        "G3와 G5 0.5B, G3와 G4 4bit는 작은 차이지만 CI가 0을 넘지 않습니다. 즉 G3의 품질 이점은 확인되지만, 그 이점이 latency/memory 비용을 정당화하는지는 별도 의사결정입니다.",
    ])
    add_left_claim(b, ["B0 → B3/G5: +33~35pp", "B3 vs G5 0.5B: 차이 작고 유의하지 않음", "G3는 상위권에서도 품질 이점 존재", "통계 결과는 비용 판단과 함께 해석"], "0B1F33")
    fit_plot(b, plots_dir / "06_check500_pairwise_delta_ci.png", 4.05, 1.55, 8.35, 4.9)
    slides.append(b.build(b.notes))

    # 15 original final landscape
    b = make("ORIGINAL FINAL", "Original Final Test Result Landscape", "Raw final test | 상위 모델 포화 | Method contrast 제한", [
        "이 장은 원본 실험 설계의 결과표입니다. raw final test 기준으로 B3, API, G3, G5, G4가 대부분 99% 이상입니다. 이 결과만 보면 '다 잘한다'가 결론처럼 보입니다.",
        "하지만 이 포화 자체가 중요한 관찰입니다. TOEIC business vocabulary의 raw meaning selection은 상위 모델에게 너무 쉬운 평가일 수 있습니다.",
        "그래서 발표에서는 raw final test를 '상위 모델이 기본 task를 해결했음'을 보여주는 배경으로 두고, 실제 방법론 차이는 check-500과 hard slice에서 설명합니다.",
    ])
    fit_plot(b, plots_dir / "10_finaltest_original_methodology_accuracy.png", 0.70, 1.50, 11.95, 5.1)
    slides.append(b.build(b.notes))

    # 16 LM stage final
    b = make("LM PATH", "LM Stages: Accuracy Saturates, Reliability Matters", "LM accuracy saturation | Confidence reliability | Output stability", [
        "원본 계획에서는 zero-shot, SFT, KD를 비교하려 했습니다. 결과를 보면 raw final test에서는 zero-shot도 이미 높습니다. 따라서 'SFT/KD가 accuracy를 폭발적으로 올렸다'는 주장은 약합니다.",
        "대신 G0 Qwen의 ECE가 매우 높고, SFT/KD 이후 calibration이 개선되는 흐름을 볼 수 있습니다. 제품 채점기 관점에서는 정답률만큼 자신감의 신뢰성이 중요합니다.",
        "이 장은 LM-only 발표를 해야 할 때 특히 유용합니다. 메시지는 'fine-tuning improves reliability, compression improves deployability'입니다.",
    ])
    add_left_claim(b, ["Raw final에서는 LM accuracy가 이미 높음", "Zero-shot의 confidence는 불안정할 수 있음", "SFT/KD는 reliability 관점에서 의미", "압축 모델은 배포성 관점에서 의미"], "356EAF")
    fit_plot(b, plots_dir / "11_finaltest_lm_stage_accuracy_ece.png", 4.05, 1.55, 8.35, 4.9)
    slides.append(b.build(b.notes))

    # 17 compression
    b = make("COMPRESSION", "Compression Boundary", "Small local LM | 4bit runtime | 배포 경계", [
        "이 장은 원본 실험 설계의 climax에 가장 가깝습니다. 큰 3B 모델은 거의 완벽하지만 무겁습니다. G5 0.5B와 1.5B는 훨씬 빠르고 가벼운데 raw final test에서는 거의 같은 품질입니다.",
        "4bit는 같은 3B 계열을 압축해 latency를 줄이는 경로입니다. 다만 현재 결과에서는 VRAM 감소를 강하게 주장하지 않고, calibration 재확인이 필요하다는 단서를 붙입니다.",
        "결론은 큰 모델을 무조건 쓰는 것이 아니라, bounded task에서는 작은 내부 AI와 압축 runtime이 실용적인 대체 경계를 만든다는 것입니다.",
    ])
    fit_plot(b, plots_dir / "12_finaltest_compression_boundary.png", 0.70, 1.50, 11.95, 5.1)
    slides.append(b.build(b.notes))

    # Appendix divider
    b = make("APPENDIX", "appendix", "", [
        "부록 진입 전 잠깐 멈추고, 이후 장표는 발표 본문이 아니라 질의 대응용 근거 자료임을 전환합니다.",
    ])
    b.textbox(4.42, 3.10, 4.50, 0.90, ["appendix"], size=54, color="0B1F33", bold=True, margin=0, align="c")
    slides.append(b.build_plain(b.notes))
    split_counter: Counter[str] = subset["split"]
    status_counter: Counter[str] = subset["status"]
    task_counter: Counter[str] = subset["task"]

    # Appendix A experimental design map
    b = make("APPENDIX", "Experimental Design Appendix Map", "data construction | model families | evaluation protocol | evidence boundary", [
        "이 장은 appendix 전체의 읽는 순서를 정리합니다. 이후 부록은 결과 그림만 모은 것이 아니라, 데이터 구성, 증강 검증, 모델군 설정, 평가 지표, claim 경계까지 실험 설계 관점으로 이어집니다.",
        "질문: appendix가 어떤 순서로 실험 설계를 설명하나요?",
        "방어: 먼저 raw vocabulary와 generated task가 어떻게 check set으로 들어왔는지 보여주고, 그 다음 B/G/H 모델군의 평가 방식과 학습 목표를 구분합니다. 마지막으로 metric과 paired statistics로 claim을 제한합니다.",
    ])
    x0 = 0.62
    step = 2.48
    nodes = [
        ("Raw GT", "word / meaning\n4-option MCQ", "3A9D5D"),
        ("Augmentation", "synonym / sense\nantonym / cloze", "7F63B8"),
        ("Validation", "auto filter +\njudge validation", "E79A32"),
        ("Check subset", "500 test items\nraw + generated", "1B8A8F"),
        ("Local runs", "B / G / H\nno API check", "C84A3A"),
    ]
    for idx, (title, body, color) in enumerate(nodes):
        x = x0 + idx * step
        add_pipeline_node(b, x, 1.72, 2.02, title, body, color)
        if idx < len(nodes) - 1:
            add_arrow(b, x + 2.08, 2.24, x + step - 0.14, 2.24)
    b.rect(0.62, 3.38, 11.95, 2.70, fill="FFFFFF", line="D9DEE5", radius=True)
    cards = [
        ("B family", "Discriminative scoring", "B0 similarity, B2 scorer reuse, B3 pairwise cross-encoder", "1B8A8F"),
        ("G family", "Local LM and adapters", "G3 quality ceiling, G4 4bit loading, G5 small local SFT adapters", "3A9D5D"),
        ("H family", "Routing policy", "B0 confidence gate over existing B3/G5 predictions", "E79A32"),
        ("Metrics", "Aggregate evidence only", "accuracy, task slice, ECE, strict parse, p50 latency, memory, paired stats", "7F63B8"),
    ]
    for idx, (title, role, body, color) in enumerate(cards):
        x = 0.92 + idx * 2.88
        b.rect(x, 3.78, 2.48, 1.62, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.15, 3.98, 2.10, 0.20, [title], size=10, color=color, bold=True, margin=0)
        b.textbox(x + 0.15, 4.30, 2.10, 0.18, [role], size=7, color="0B1F33", bold=True, margin=0)
        b.textbox(x + 0.15, 4.62, 2.10, 0.38, [body], size=6, color="111827", margin=0, line_spacing=950)
    badges = [("design evidence", "1B8A8F"), ("model structure", "3A9D5D"), ("aggregate metrics", "7F63B8"), ("visible limits", "C84A3A")]
    x = 0.92
    for label, color in badges:
        b.rect(x, 5.64, 1.55, 0.34, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.10, 5.73, 1.35, 0.12, [label], size=7, color=color, bold=True, margin=0, align="c")
        x += 1.70
    slides.append(b.build(b.notes))

    # Appendix B data and augmentation protocol
    b = make("APPENDIX", "Data and Augmentation Protocol", "raw anchors | generated task types | validation gates | teacher signals", [
        "이 장은 데이터 증강 관련 질문을 실험 설계 관점에서 받기 위한 장표입니다.",
        "질문: 증강 데이터는 어떻게 사용됐나요?",
        "방어: generated rows는 task 다양성을 만들기 위한 보조 데이터입니다. check-500에는 raw test 388개와 judge-validated generated 112개가 섞였고, 학습 설계에서는 SFT용 raw+aug view와 KD용 teacher-score view가 분리됩니다.",
        "질문: teacher score는 정답을 새로 만든 건가요?",
        "방어: 정답 anchor는 payload의 answer label을 유지하고, teacher score는 선택지별 soft confidence를 제공하는 보조 신호입니다. G3 KD에서는 이 soft distribution이 answer-letter KL loss에 들어갑니다.",
        "질문: 이 데이터가 human approved인가요?",
        "방어: 아니라고 답합니다. judge-validated는 자동 필터와 독립 judge validation을 통과했다는 뜻이며 human approval과 분리합니다.",
    ])
    flow = [
        ("Raw anchors", "word / POS\nmeaning", "3A9D5D"),
        ("Task generation", "synonym / sense\nantonym / cloze", "7F63B8"),
        ("Auto filter", "format\nteacher score", "E79A32"),
        ("Judge gate", "semantic\nleakage", "E79A32"),
        ("Check-500", f"raw {status_counter.get('raw_gt', 0)}\ngen {status_counter.get('aug_judge_pass', 0)}", "1B8A8F"),
    ]
    x0 = 0.62
    for idx, (title, body, color) in enumerate(flow):
        x = x0 + idx * 2.42
        add_pipeline_node(b, x, 1.62, 1.98, title, body, color)
        if idx < len(flow) - 1:
            add_arrow(b, x + 2.04, 2.14, x + 2.33, 2.14)

    b.rect(0.62, 3.25, 3.45, 2.64, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(0.88, 3.50, 2.88, 0.22, ["Source mix"], size=11, color="0B1F33", bold=True, margin=0)
    test_total = int(status_counter.get("raw_gt", 0) + status_counter.get("aug_judge_pass", 0))
    add_compact_bar(b, 0.92, 3.98, "Raw", int(status_counter.get("raw_gt", 0)), test_total, "3A9D5D", max_w=1.52)
    add_compact_bar(b, 0.92, 4.40, "Generated", int(status_counter.get("aug_judge_pass", 0)), test_total, "7F63B8", max_w=1.52)
    b.textbox(0.92, 5.12, 2.65, 0.20, ["check subset only"], size=8, color="5F6B7A", bold=True, margin=0)

    b.rect(4.35, 3.25, 3.90, 2.64, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(4.60, 3.50, 3.32, 0.22, ["Generated task coverage"], size=11, color="0B1F33", bold=True, margin=0)
    gen_total = int(status_counter.get("aug_judge_pass", 0))
    gen_rows = [
        ("Synonym", task_counter.get("Synonym Selection", 0), "1B8A8F"),
        ("Sense", task_counter.get("Sense Disambiguation", 0), "E79A32"),
        ("Antonym", task_counter.get("Antonym Selection", 0), "C84A3A"),
        ("Cloze", task_counter.get("Context Cloze", 0), "7F63B8"),
    ]
    y = 3.94
    for label, value, color in gen_rows:
        add_compact_bar(b, 4.65, y, label, int(value), gen_total, color, max_w=1.50)
        y += 0.36
    for idx, (label, color) in enumerate([("stress coverage", "C84A3A"), ("slice view", "1B8A8F")]):
        x = 4.65 + idx * 1.28
        b.rect(x, 5.38, 1.10, 0.32, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.08, 5.47, 0.94, 0.10, [label], size=6, color=color, bold=True, margin=0, align="c")

    b.rect(8.55, 3.25, 3.72, 2.64, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(8.80, 3.50, 3.16, 0.22, ["Teacher signal placement"], size=11, color="0B1F33", bold=True, margin=0)
    add_arch_block(b, 8.88, 3.94, 1.38, 0.78, "Hard label", "A / B / C / D\nkept as target", "0B1F33")
    add_arrow(b, 10.36, 4.33, 10.75, 4.33)
    add_arch_block(b, 10.84, 3.94, 1.08, 0.78, "Soft", "4-way\nscores", "C84A3A")
    for idx, (label, color) in enumerate([("KD soft scores", "C84A3A"), ("G5 hard-label check", "1B8A8F")]):
        x = 8.88 + idx * 1.42
        b.rect(x, 5.08, 1.25, 0.34, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.08, 5.17, 1.09, 0.12, [label], size=6, color=color, bold=True, margin=0, align="c")
    slides.append(b.build(b.notes))

    # Appendix C model protocol matrix
    b = make("APPENDIX", "Model Family Architecture Map", "B/G/H families | component state | evaluation mode", [
        "이 장은 B/G/H 각 실험군이 정확히 무엇을 수행했는지 한 표로 방어하기 위한 설계 부록입니다.",
        "질문: 어떤 모델이 학습됐고 어떤 모델은 평가만 했나요?",
        "방어: 현재 appendix의 no-api/no-retrain 확인은 모든 모델을 평가 모드로 실행했습니다. B2는 기존 scorer 재사용, B3는 fine-tune disabled, G3/G4/G5는 existing adapter loading, H1은 existing prediction outputs를 라우팅합니다.",
        "질문: G5를 압축 KD 결과로 보면 되나요?",
        "방어: 이 표에서 G5는 small local SFT adapter 평가로 분리합니다. teacher-logit KD compression은 별도 경로로 설명해야 합니다.",
    ])
    b.rect(0.60, 1.58, 3.70, 4.85, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(0.88, 1.82, 3.05, 0.22, ["B family: discriminative scoring"], size=11, color="1B8A8F", bold=True, margin=0)
    add_arch_block(b, 0.95, 2.26, 0.92, 0.84, "B0", "embed\nscore", "1B8A8F")
    add_arrow(b, 1.96, 2.68, 2.28, 2.68)
    add_arch_block(b, 2.38, 2.26, 1.22, 0.84, "B2", "MLP scorer\nreuse", "7F63B8")
    add_arch_block(b, 1.47, 3.34, 1.42, 0.88, "B3", "cross-encoder\nFT off", "3A9D5D")
    add_arrow(b, 2.17, 3.16, 2.17, 3.28)
    b.textbox(0.95, 4.76, 2.82, 0.24, [f"Accuracy range: {pct(b2['accuracy'])} → {pct(b3['accuracy'])}"], size=9, color="0B1F33", bold=True, margin=0)
    for idx, (label, color) in enumerate([("option scoring", "1B8A8F"), ("no generation", "5F6B7A")]):
        x = 0.95 + idx * 1.28
        b.rect(x, 5.18, 1.10, 0.32, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.08, 5.27, 0.94, 0.10, [label], size=6, color=color, bold=True, margin=0, align="c")

    b.rect(4.60, 1.58, 4.15, 4.85, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(4.88, 1.82, 3.45, 0.22, ["G family: local LM + adapter"], size=11, color="3A9D5D", bold=True, margin=0)
    add_arch_block(b, 4.95, 2.28, 1.22, 0.72, "G3", "3B\nLoRA KD", "C84A3A")
    add_arrow(b, 6.28, 2.64, 6.76, 2.64)
    add_arch_block(b, 6.86, 2.28, 1.22, 0.72, "G4", "same 3B\n4bit load", "7F63B8")
    add_arch_block(b, 5.78, 3.52, 1.65, 0.78, "G5", "0.5B / 1.5B\nG1 SFT adapter", "1B8A8F")
    b.textbox(5.04, 4.76, 3.15, 0.24, [f"Acc: G3 {pct(g3['accuracy'])}, G5 1.5B {pct(g15['accuracy'])}"], size=9, color="0B1F33", bold=True, margin=0)
    for idx, (label, color) in enumerate([("G5 SFT adapter", "1B8A8F"), ("KD separate path", "C84A3A")]):
        x = 5.04 + idx * 1.36
        b.rect(x, 5.18, 1.18, 0.32, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.08, 5.27, 1.02, 0.10, [label], size=6, color=color, bold=True, margin=0, align="c")

    b.rect(9.05, 1.58, 3.40, 4.85, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(9.33, 1.82, 2.75, 0.22, ["H family: routing policy"], size=11, color="E79A32", bold=True, margin=0)
    add_arch_block(b, 9.42, 2.24, 1.10, 0.70, "Gate", "B0\nconfidence", "E79A32")
    add_arrow(b, 10.62, 2.42, 11.02, 2.16)
    add_arch_block(b, 11.10, 1.90, 0.90, 0.62, "B3", "94.4%", "3A9D5D")
    add_arrow(b, 10.62, 2.74, 11.02, 3.02)
    add_arch_block(b, 11.10, 2.82, 0.90, 0.62, "G5", "2.0%", "1B8A8F")
    b.textbox(9.42, 4.74, 2.25, 0.24, [f"H1 Acc {pct(h1['accuracy'])}"], size=9, color="0B1F33", bold=True, margin=0)
    for idx, (label, color) in enumerate([("routing candidate", "E79A32"), ("cost sweep next", "5F6B7A")]):
        x = 9.42 + idx * 1.16
        b.rect(x, 5.18, 1.02, 0.32, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.06, 5.27, 0.90, 0.10, [label], size=6, color=color, bold=True, margin=0, align="c")
    slides.append(b.build(b.notes))

    # Appendix D objective/inference/metric protocol
    b = make("APPENDIX", "Objective, Inference, and Metric Protocol", "training objective | answer contract | deterministic inference | aggregate metrics", [
        "이 장은 loss, 추론 방식, 지표 산출 방식에 대한 상세 질문을 한 번에 받기 위한 설계 장표입니다.",
        "질문: loss 종류가 무엇인가요?",
        "방어: SFT는 assistant completion CE입니다. G3 KD는 completion CE에 answer-letter soft distribution KL을 더합니다. B/H 계열은 현재 확인 run에서 LM training objective가 없습니다.",
        "질문: LM 출력은 어떻게 채점했나요?",
        "방어: 모델은 구조화된 answer contract를 따르도록 prompt를 받습니다. strict contract를 지키지 않아도 fallback parser로 답을 복원할 수 있으므로, accuracy와 strict parse error를 분리해 보고합니다.",
        "질문: latency와 memory는 무엇을 의미하나요?",
        "방어: 각 item 단위 local inference에서 p50 latency와 peak RAM/VRAM을 집계한 운영 지표입니다. 실제 API 비용이 아니라 local runtime 부담을 비교하는 지표입니다.",
    ])
    b.rect(0.62, 1.62, 5.50, 2.40, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(0.90, 1.88, 4.90, 0.22, ["Training objective paths"], size=12, color="0B1F33", bold=True, margin=0)
    add_arch_block(b, 0.95, 2.36, 1.28, 0.78, "SFT", "assistant\ncompletion CE", "3A9D5D")
    add_arrow(b, 2.34, 2.75, 2.76, 2.75)
    add_arch_block(b, 2.86, 2.36, 1.22, 0.78, "LoRA", "adapter\nupdate", "3A9D5D")
    add_arch_block(b, 4.48, 2.36, 1.18, 0.78, "G5", "G1 SFT\ncheck", "1B8A8F")
    b.textbox(0.98, 3.52, 4.80, 0.20, ["G3 KD objective = completion CE + 0.5 × KL(teacher A/B/C/D)"], size=10, color="C84A3A", bold=True, margin=0)

    b.rect(6.40, 1.62, 5.92, 2.40, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(6.68, 1.88, 5.20, 0.22, ["Inference and answer contract"], size=12, color="0B1F33", bold=True, margin=0)
    add_arch_block(b, 6.78, 2.36, 1.16, 0.78, "Prompt", "MCQ +\ncontract", "7F63B8")
    add_arrow(b, 8.04, 2.75, 8.42, 2.75)
    add_arch_block(b, 8.52, 2.36, 1.18, 0.78, "LM", "deterministic\ngeneration", "7F63B8")
    add_arrow(b, 9.80, 2.75, 10.18, 2.75)
    add_arch_block(b, 10.28, 2.36, 1.40, 0.78, "Parser", "strict +\nfallback", "7F63B8")
    for idx, (label, color) in enumerate([("Accuracy metric", "1B8A8F"), ("Strict parse metric", "7F63B8")]):
        x = 6.78 + idx * 1.74
        b.rect(x, 3.43, 1.54, 0.36, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.10, 3.53, 1.34, 0.12, [label], size=7, color=color, bold=True, margin=0, align="c")

    b.rect(0.62, 4.35, 11.70, 1.70, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(0.90, 4.58, 4.20, 0.20, ["Metric groups"], size=12, color="0B1F33", bold=True, margin=0)
    chips = [
        ("Accuracy", "overall", "1B8A8F"),
        ("Task slice", "raw/syn/ant/cloze", "C84A3A"),
        ("ECE", "confidence", "7F63B8"),
        ("Strict parse", "contract", "7F63B8"),
        ("p50 latency", "runtime", "3A9D5D"),
        ("RAM/VRAM", "footprint", "3A9D5D"),
        ("Paired stats", "same 500 items", "0B1F33"),
    ]
    x = 0.95
    for title, subtitle, color in chips:
        b.rect(x, 5.04, 1.45, 0.54, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.08, 5.13, 1.28, 0.12, [title], size=7, color=color, bold=True, margin=0, align="c")
        b.textbox(x + 0.08, 5.31, 1.28, 0.10, [subtitle], size=5, color="5F6B7A", margin=0, align="c")
        x += 1.58
    slides.append(b.build(b.notes))

    # Appendix A visual evidence boards
    b = make("APPENDIX", "Visual Evidence: Evaluation and Method Gap", "Raw saturation | Check-500 contrast | Method gap", [
        "부록은 표 목록이 아니라, 질문이 들어왔을 때 바로 가리킬 수 있는 근거 화면으로 구성합니다.",
        "각 패널은 본문 메시지와 연결되는 대표 plot입니다. 세부 산출물 원문은 포함하지 않습니다.",
    ])
    add_evidence_plot_card(b, plots_dir / "13_final_vs_check500_accuracy.png", 0.62, 1.58, 5.95, 4.88, "Raw final vs Check-500", "평가 난이도 근거", "E79A32")
    add_evidence_plot_card(b, plots_dir / "01_check500_methodology_accuracy.png", 6.78, 1.58, 5.95, 4.88, "Methodology Accuracy Gap", "B0/B2/B3/G 계열 대비", "1B8A8F")
    slides.append(b.build(b.notes))

    b = make("APPENDIX", "Visual Evidence: Hard Slice and Deployment", "Antonym stress | Latency-memory frontier", [
        "hard slice와 배포성 관련 질문에 대응하는 근거 화면입니다.",
        "평균 정답률 이후의 차이를 hard slice와 runtime frontier에서 분리해 보여줍니다.",
    ])
    add_evidence_plot_card(b, plots_dir / "04_check500_antonym_hard_slice.png", 0.62, 1.58, 5.95, 4.88, "Antonym Hard Slice", "평균 뒤의 취약점", "C84A3A")
    add_evidence_plot_card(b, plots_dir / "02_check500_accuracy_latency_memory_frontier.png", 6.78, 1.58, 5.95, 4.88, "Accuracy-Latency-Memory", "배포 후보 경계", "3A9D5D")
    slides.append(b.build(b.notes))

    # Appendix B evaluation-set evidence
    b = make("APPENDIX", "Evaluation Set Evidence", "500 test items | raw/generated mix | task slice composition", [
        "이 장은 check-500이 어떤 평가 묶음인지 시각적으로 보여주는 근거 장표입니다.",
        "train/dev rows는 기준선 튜닝과 scorer 재사용 검증을 위해 유지했고, 발표에서 직접 비교하는 test rows는 500개입니다.",
    ])
    add_count_card(b, 0.75, 1.60, "Total metadata rows", str(subset["all_n"]), "train/dev/test retained", "1B8A8F")
    add_count_card(b, 3.18, 1.60, "Check test rows", str(subset["test_n"]), "presentation subset", "E79A32")
    add_count_card(b, 5.61, 1.60, "Raw test rows", str(status_counter.get("raw_gt", 0)), "original GT items", "3A9D5D")
    add_count_card(b, 8.04, 1.60, "Generated rows", str(status_counter.get("aug_judge_pass", 0)), "judge-passed items", "7F63B8")
    add_count_card(b, 10.47, 1.60, "No-API runs", "8", "local-only models", "C84A3A")
    b.rect(0.75, 2.95, 5.95, 3.40, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(1.02, 3.18, 5.35, 0.22, ["Split retained for controlled checking"], size=11, color="0B1F33", bold=True, margin=0)
    split_total = sum(split_counter.values())
    y = 3.68
    for label, color in [("train", "1B8A8F"), ("dev", "7F63B8"), ("test", "E79A32")]:
        add_distribution_bar(b, 1.02, y, label, int(split_counter.get(label, 0)), split_total, color, max_w=3.05)
        y += 0.48
    split_badges = [("train/dev retained", "1B8A8F"), ("test-only comparison", "E79A32"), ("B0/B2 guard", "7F63B8")]
    x = 1.02
    for label, color in split_badges:
        b.rect(x, 5.30, 1.42, 0.34, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.08, 5.39, 1.26, 0.12, [label], size=6, color=color, bold=True, margin=0, align="c")
        x += 1.52
    b.rect(7.05, 2.95, 5.25, 3.40, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(7.32, 3.18, 4.75, 0.22, ["Test task composition"], size=11, color="0B1F33", bold=True, margin=0)
    task_total = sum(task_counter.values())
    y = 3.68
    task_colors = ["3A9D5D", "1B8A8F", "E79A32", "C84A3A", "7F63B8"]
    for idx, (label, value) in enumerate(task_counter.most_common()):
        add_distribution_bar(b, 7.32, y, label, int(value), task_total, task_colors[idx % len(task_colors)], max_w=2.40)
        y += 0.44
    slides.append(b.build(b.notes))

    # Appendix C evaluation/augmentation visual audit
    answer_counter: Counter[str] = subset["answer"]
    generated_task_answer: dict[str, Counter[str]] = subset["generated_task_answer"]
    b = make("APPENDIX", "Evaluation Subset Visual Audit", "controlled subset | answer distribution | generated-slice audit", [
        "질문: check-500은 최종 benchmark인가요?",
        "방어: 아닙니다. check-500은 원본 final-test 포화 이후 방법론 차이를 보기 위한 controlled confirmation set입니다. train/dev row는 기준선 튜닝과 scorer 재사용 조건을 유지하기 위한 배경이고, 발표 비교 대상은 test 500개입니다.",
        "질문: generated 데이터는 사람이 검수했나요?",
        "방어: 사람 검수라고 말하지 않습니다. 표현은 judge-validated augmentation으로 제한합니다. 자동 필터와 독립 judge validation을 통과한 항목이라는 의미입니다.",
        "질문: answer-position bias가 있으면 결과가 부풀려진 것 아닌가요?",
        "방어: 전체 500개는 raw 388개가 섞여 A/B/C/D 분포가 완전히 한쪽으로 쏠리지는 않습니다. 다만 generated slice는 Synonym, Sense, Context에서 A 편향이 있으므로 unbiased benchmark가 아니라 stress slice 근거로만 사용한다고 설명합니다.",
    ])
    b.rect(0.65, 1.60, 3.80, 4.82, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(0.92, 1.84, 3.10, 0.24, ["Evaluation construction"], size=11, color="0B1F33", bold=True, margin=0)
    add_arch_block(b, 0.98, 2.22, 2.92, 0.78, "1. Retain context", f"train {split_counter.get('train', 0)} / dev {split_counter.get('dev', 0)}", "1B8A8F")
    add_arch_block(b, 0.98, 3.08, 2.92, 0.78, "2. Select test subset", f"test {subset['test_n']} items, seed fixed", "E79A32")
    add_arch_block(b, 0.98, 3.94, 2.92, 0.78, "3. Mix sources", f"raw {status_counter.get('raw_gt', 0)} + generated {status_counter.get('aug_judge_pass', 0)}", "3A9D5D")
    add_arch_block(b, 0.98, 4.80, 2.92, 0.78, "4. Evaluate only", "no API call / no retraining", "C84A3A")
    check_badges = [("check-only", "C84A3A"), ("fixed seed", "E79A32"), ("local eval", "1B8A8F")]
    x = 0.98
    for label, color in check_badges:
        b.rect(x, 5.74, 0.88, 0.30, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.06, 5.82, 0.76, 0.10, [label], size=6, color=color, bold=True, margin=0, align="c")
        x += 0.96

    b.rect(4.78, 1.60, 3.65, 4.82, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(5.05, 1.84, 3.05, 0.24, ["Answer label distribution"], size=11, color="0B1F33", bold=True, margin=0)
    total_answers = sum(answer_counter.values())
    y = 2.35
    for label, color in [("A", "1B8A8F"), ("B", "3A9D5D"), ("C", "E79A32"), ("D", "7F63B8")]:
        add_compact_bar(b, 5.08, y, label, int(answer_counter.get(label, 0)), total_answers, color, max_w=1.50)
        y += 0.38
    b.line(5.00, 4.08, 8.05, 4.08, color="D9DEE5", width=6350)
    label_badges = [("raw balance", "3A9D5D"), ("generated stress", "C84A3A"), ("aggregate only", "5F6B7A")]
    x = 5.05
    for label, color in label_badges:
        b.rect(x, 4.35, 0.98, 0.34, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.06, 4.44, 0.86, 0.12, [label], size=6, color=color, bold=True, margin=0, align="c")
        x += 1.04
    b.textbox(5.05, 5.50, 2.95, 0.34, ["aggregate label view"], size=8, color="5F6B7A", bold=True, margin=0)

    b.rect(8.76, 1.60, 3.85, 4.82, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(9.03, 1.84, 3.15, 0.24, ["Generated slice audit"], size=11, color="0B1F33", bold=True, margin=0)
    caveat_rows = [
        ("Synonym", generated_task_answer.get("Synonym Selection", Counter()).get("A", 0), sum(generated_task_answer.get("Synonym Selection", Counter()).values())),
        ("Sense", generated_task_answer.get("Sense Disambiguation", Counter()).get("A", 0), sum(generated_task_answer.get("Sense Disambiguation", Counter()).values())),
        ("Cloze", generated_task_answer.get("Context Cloze", Counter()).get("A", 0), sum(generated_task_answer.get("Context Cloze", Counter()).values())),
        ("Antonym", task_counter.get("Antonym Selection", 0), task_counter.get("Antonym Selection", 0)),
    ]
    y = 2.34
    for label, value, total in caveat_rows:
        value_text = f"A {value}/{total}" if label != "Antonym" else f"n={total}"
        add_compact_bar(b, 9.05, y, label, value, total or 1, "C84A3A" if label == "Antonym" else "E79A32", max_w=1.38, value_text=value_text)
        y += 0.40
    b.rect(9.08, 4.20, 0.88, 0.40, fill="FFF8F2", line="F2D0A5", radius=True)
    b.textbox(9.16, 4.31, 0.70, 0.12, ["stress"], size=7, color="C84A3A", bold=True, margin=0, align="c")
    b.rect(10.08, 4.20, 0.88, 0.40, fill="F7F9FB", line="D9DEE5", radius=True)
    b.textbox(10.16, 4.31, 0.70, 0.12, ["not final"], size=7, color="5F6B7A", bold=True, margin=0, align="c")
    b.rect(11.08, 4.20, 0.88, 0.40, fill="F7F9FB", line="D9DEE5", radius=True)
    b.textbox(11.16, 4.31, 0.70, 0.12, ["balance next"], size=7, color="5F6B7A", bold=True, margin=0, align="c")
    b.textbox(9.05, 5.48, 3.12, 0.34, ["generated slice boundary"], size=8, color="5F6B7A", bold=True, margin=0)
    slides.append(b.build(b.notes))

    # Appendix D discriminative model structures
    b = make("APPENDIX", "Model Structures: B0 / B2 / B3", "Discriminative family | per-model flow | no adapter", [
        "B 계열은 생성형 답변을 만들지 않고, 선택지 점수화로 정답을 고르는 판별형 경로입니다.",
        "B0, B2, B3를 하나의 통합 구조가 아니라 모델별 흐름으로 분리해 보여줍니다.",
        "질문: B3도 재학습했나요?",
        "방어: 현재 확인 run에서는 재학습하지 않았습니다. B3는 question-option cross-encoder를 그대로 사용해 네 선택지 점수를 비교한 결과입니다.",
        "질문: 왜 B0/B2가 낮나요?",
        "방어: B0/B2는 문제와 선택지를 따로 embedding하거나 embedding feature를 scorer에 넣습니다. 반면 B3는 question-option pair를 함께 읽기 때문에 동의어, 문맥, 오답 후보 간 미세 차이를 더 잘 잡습니다.",
    ])
    add_flow_card(
        b,
        0.70,
        1.55,
        11.80,
        1.58,
        "B0",
        "Embedding similarity baseline",
        ["문제/선택지", "sentence embedding", "similarity score", "argmax answer"],
        [("Acc", pct(b0["accuracy"])), ("p50", fmt_ms(b0["latency_p50"])), ("Adapter", "none")],
        "1B8A8F",
    )
    add_flow_card(
        b,
        0.70,
        3.25,
        11.80,
        1.58,
        "B2",
        "Embedding feature + reused scorer",
        ["문제/선택지", "embedding features", "existing MLP scorer", "argmax answer"],
        [("Acc", pct(b2["accuracy"])), ("p50", fmt_ms(b2["latency_p50"])), ("Train", "reuse")],
        "7F63B8",
    )
    add_flow_card(
        b,
        0.70,
        4.95,
        11.80,
        1.58,
        "B3",
        "Question-option cross-encoder",
        ["question + option", "cross-encoder", "pairwise score", "ranked answer"],
        [("Acc", pct(b3["accuracy"])), ("p50", fmt_ms(b3["latency_p50"])), ("Adapter", "none")],
        "3A9D5D",
    )
    slides.append(b.build(b.notes))

    # Appendix E G3/G4 model structures
    b = make("APPENDIX", "Model Structures: G3 / G4", "3B local LM | LoRA adapter | 4bit loading", [
        "G3와 G4는 같은 Qwen 3B 계열 adapter를 중심으로 해석합니다.",
        "G3는 품질 상한, G4는 4bit loading을 통한 압축 runtime 확인으로 분리해 설명합니다.",
        "질문: G3의 KD loss는 무엇인가요?",
        "방어: G3는 assistant completion에 대한 causal-LM loss를 기본으로 두고, 정답 letter A/B/C/D 위치의 soft distribution KL loss를 더합니다. 기본 lambda_soft는 0.5입니다. 전체 sequence distillation이라고 말하지 않습니다.",
        "질문: G4는 새 모델을 학습한 건가요?",
        "방어: 아닙니다. 같은 3B adapter 경로를 유지하고 base loading만 4bit로 바꿔 runtime과 memory trade-off를 확인한 것입니다.",
        "질문: 4bit가 손실 없는 압축인가요?",
        "방어: 아닙니다. check-500에서는 memory와 latency가 줄지만 accuracy와 strict parse 안정성은 G3보다 떨어집니다. 따라서 trade-off 실험으로 설명합니다.",
    ])
    add_flow_card(
        b,
        0.70,
        1.68,
        11.80,
        2.05,
        "G3",
        "Qwen 3B local LM + existing LoRA adapter",
        ["MCQ prompt", "Qwen 3B base", "LoRA final adapter", "structured answer"],
        [("Acc", pct(g3["accuracy"])), ("p50", fmt_ms(g3["latency_p50"])), ("RAM", fmt_mb(g3["peak_VRAM_or_RAM"]))],
        "C84A3A",
    )
    add_flow_card(
        b,
        0.70,
        4.18,
        11.80,
        2.05,
        "G4",
        "Qwen 3B 4bit loading + same adapter path",
        ["MCQ prompt", "4bit loaded base", "LoRA adapter merge path", "structured answer"],
        [("Acc", pct(g4["accuracy"])), ("p50", fmt_ms(g4["latency_p50"])), ("RAM", fmt_mb(g4["peak_VRAM_or_RAM"]))],
        "7F63B8",
    )
    slides.append(b.build(b.notes))

    # Appendix F G5 model structures
    b = make("APPENDIX", "Model Structures: G5 Small Local LM", "0.5B / 1.5B | LoRA adapter | deployable candidates", [
        "G5는 작은 local LM을 통해 큰 모델 호출을 줄일 수 있는지 확인하는 경로입니다.",
        "0.5B와 1.5B는 같은 범주의 실험이지만 운영 후보로 보는 기준이 달라 별도 구조로 제시합니다.",
        "질문: 이 G5 결과는 G3 teacher로 distillation한 결과인가요?",
        "방어: 현재 appendix의 no-api/no-retrain 확인 결과는 G5_Qwen0p5_G1, G5_Qwen1p5_G1입니다. 즉 작은 Qwen base에 G1 LoRA SFT adapter를 얹은 평가입니다. G3 teacher logits 기반 KD student 결과라고 말하지 않습니다.",
        "질문: 그러면 G5의 claim은 무엇인가요?",
        "방어: bounded vocabulary judging에서는 0.5B/1.5B 작은 local LM도 높은 accuracy와 낮은 latency를 보여 deployment candidate가 될 수 있다는 claim입니다. KD compression claim은 별도 G5 KD run으로 분리해야 합니다.",
    ])
    add_flow_card(
        b,
        0.70,
        1.72,
        11.80,
        2.02,
        "G5 0.5B",
        "Smallest local student + LoRA adapter",
        ["MCQ prompt", "Qwen 0.5B base", "G1 LoRA adapter", "answer contract"],
        [("Acc", pct(g05["accuracy"])), ("p50", fmt_ms(g05["latency_p50"])), ("RAM", fmt_mb(g05["peak_VRAM_or_RAM"]))],
        "3A9D5D",
    )
    add_flow_card(
        b,
        0.70,
        4.18,
        11.80,
        2.02,
        "G5 1.5B",
        "Small local student + LoRA adapter",
        ["MCQ prompt", "Qwen 1.5B base", "G1 LoRA adapter", "answer contract"],
        [("Acc", pct(g15["accuracy"])), ("p50", fmt_ms(g15["latency_p50"])), ("RAM", fmt_mb(g15["peak_VRAM_or_RAM"]))],
        "1B8A8F",
    )
    slides.append(b.build(b.notes))

    # Appendix G loss and adapter defense
    b = make("APPENDIX", "No-Retrain Evaluation Scope", "reuse | adapter loading | quantized loading | routing", [
        "질문: 모델별 loss가 무엇인가요?",
        "방어: SFT 계열은 assistant completion 영역의 next-token cross entropy입니다. G3 KD는 여기에 answer-letter A/B/C/D 분포 KL을 더한 구조입니다. B0/B2/B3/H1은 현재 확인 run에서 LM 학습 loss가 없습니다.",
        "질문: G5도 KD loss를 썼나요?",
        "방어: 이번 appendix의 G5는 G1 SFT adapter 평가입니다. 따라서 G5 결과를 설명할 때는 SFT adapter 기반 작은 local LM이라고 말하고, teacher-logit KD는 별도 실험 경로로 분리합니다.",
        "질문: no-retrain 확인이라는 말의 근거는 무엇인가요?",
        "방어: B2는 기존 scorer를 재사용했고, B3는 fine-tune 없이 평가했으며, G3/G4/G5는 이미 존재하는 adapter를 불러 평가했습니다. 장표에서는 경로명이 아니라 구조와 역할만 설명합니다.",
    ])
    b.rect(0.65, 1.58, 11.95, 4.78, fill="FFFFFF", line="D9DEE5", radius=True)
    b.textbox(0.92, 1.84, 11.35, 0.22, ["No-API / No-Retrain confirmation scope"], size=12, color="0B1F33", bold=True, margin=0)
    scope_rows = [
        ("B2", "existing MLP scorer reused", "no new scorer training", "7F63B8"),
        ("B3", "cross-encoder scoring only", "fine-tune disabled", "3A9D5D"),
        ("G3", "existing 3B LoRA KD adapter", "evaluation only", "C84A3A"),
        ("G5", "existing 0.5B/1.5B G1 SFT adapter", "not KD in this run", "1B8A8F"),
        ("H1", "policy over existing prediction outputs", "routing check only", "E79A32"),
    ]
    x = 0.95
    for run, structure, scope, color in scope_rows:
        b.rect(x, 2.34, 2.14, 1.04, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.12, 2.48, 0.42, 0.18, [run], size=10, color=color, bold=True, margin=0)
        b.textbox(x + 0.56, 2.43, 1.42, 0.18, [structure], size=6, color="111827", bold=True, margin=0)
        b.textbox(x + 0.56, 2.76, 1.42, 0.16, [scope], size=6, color="5F6B7A", margin=0)
        x += 2.28
    b.line(1.08, 4.05, 12.05, 4.05, color="D9DEE5", width=6350)
    legend = [
        ("SFT", "CE", "3A9D5D"),
        ("KD", "CE + 0.5 KL", "C84A3A"),
        ("4bit", "load only", "7F63B8"),
        ("Hybrid", "compose outputs", "E79A32"),
    ]
    x = 1.08
    for title, body, color in legend:
        b.rect(x, 4.56, 2.28, 0.72, fill="FFFFFF", line="D9DEE5", radius=True)
        b.textbox(x + 0.14, 4.70, 0.78, 0.16, [title], size=9, color=color, bold=True, margin=0)
        b.textbox(x + 0.96, 4.70, 1.05, 0.16, [body], size=8, color="111827", bold=True, margin=0, align="c")
        x += 2.62
    boundary_badges = [("loaded artifacts only", "1B8A8F"), ("no B4/API", "C84A3A"), ("no new LoRA/KD", "7F63B8"), ("evaluation mode", "3A9D5D")]
    x = 1.08
    for label, color in boundary_badges:
        b.rect(x, 5.62, 1.52, 0.34, fill="F7F9FB", line="D9DEE5", radius=True)
        b.textbox(x + 0.08, 5.71, 1.36, 0.12, [label], size=7, color=color, bold=True, margin=0, align="c")
        x += 1.66
    slides.append(b.build(b.notes))

    # Appendix H H1 model/routing structure
    b = make("APPENDIX", "Model Structure: H1 Local Hybrid", "confidence gate | B3 route | G5 fallback", [
        "H1은 단일 모델이 아니라 운영 정책 구조입니다. 그래서 confidence gate와 fallback 경로를 별도 시각화합니다.",
        "현재 확인 결과에서는 fallback 사용률이 낮으므로 완성된 cost-saving 정책이 아니라 routing 후보로 해석합니다.",
        "질문: H1은 비용 절감이 입증된 건가요?",
        "방어: 아직 아닙니다. 현재 selected policy는 accuracy 우선 sweep에서 low threshold 0.75, high threshold 0.85를 선택했고, route 분포는 primary 3.6%, B3 94.4%, G5 fallback 2.0%입니다. 따라서 cost-saving 완성본이 아니라 routing candidate로 말해야 합니다.",
        "질문: H1 성능은 왜 B3와 비슷한가요?",
        "방어: 대부분 B3 route를 사용했기 때문입니다. 이 결과는 라우팅 구조가 성능을 유지할 수 있음을 보여주지만, 비용 최적화까지는 threshold 재설계가 필요합니다.",
    ])
    b.rect(0.70, 1.60, 11.80, 4.95, fill="FFFFFF", line="D9DEE5", radius=True)
    add_arch_block(b, 1.10, 2.05, 2.05, 0.82, "Input", "MCQ item\nwith options", "0B1F33")
    add_arrow(b, 3.22, 2.46, 3.75, 2.46)
    add_arch_block(b, 3.82, 2.05, 2.05, 0.82, "Gate", "B0 confidence\nthreshold", "E79A32")
    add_arrow(b, 5.95, 2.36, 6.68, 1.88)
    add_arch_block(b, 6.78, 1.42, 2.32, 0.92, "Primary route", "B3 cross-encoder\nfast local answer", "3A9D5D")
    add_arrow(b, 5.95, 2.62, 6.68, 3.14)
    add_arch_block(b, 6.78, 2.82, 2.32, 0.92, "Fallback route", "G5 local LM\nharder items", "1B8A8F")
    add_arrow(b, 9.20, 1.88, 9.88, 2.34)
    add_arrow(b, 9.20, 3.24, 9.88, 2.56)
    add_arch_block(b, 9.98, 2.05, 1.92, 0.82, "Output", "selected\nanswer", "7F63B8")
    b.line(1.10, 4.30, 11.90, 4.30, color="D9DEE5", width=6350)
    add_metric_chip(b, 1.30, 4.75, "Acc", pct(h1["accuracy"]), "E79A32", w=1.05)
    add_metric_chip(b, 2.55, 4.75, "p50", fmt_ms(h1["latency_p50"]), "E79A32", w=1.05)
    add_metric_chip(b, 3.80, 4.75, "Fallback", pct(h1.get("fallback_rate", 0.0)), "E79A32", w=1.05)
    add_metric_chip(b, 5.05, 4.75, "Strict parse", pct(h1["strict_parse_error_rate"]), "E79A32", w=1.05)
    add_metric_chip(b, 6.30, 4.75, "Primary", pct(h1.get("primary_accept_rate", 0.0)), "E79A32", w=1.05)
    add_metric_chip(b, 7.55, 4.75, "B3 route", pct(h1.get("cross_encoder_rate", 0.0)), "E79A32", w=1.05)
    b.textbox(8.88, 4.76, 2.88, 0.34, ["Selected gate: low 0.75 / high 0.85"], size=9, color="5F6B7A", bold=True, margin=0)
    slides.append(b.build(b.notes))

    # Appendix I metric evidence boards
    metric_notes = [
        "마지막 부록은 품질 외 지표를 확대 근거 장표로 분리합니다.",
        "질문: 모델별 정답/오답 예시를 보여줄 수 있나요?",
        "방어: 발표 본문과 appendix에서는 개별 정답/오답 사례가 아니라 aggregate 성능 지표만 사용합니다. 개별 사례는 편향된 인상을 줄 수 있어, 성능 판단은 accuracy, task accuracy, ECE, strict parse, latency, memory로 제한합니다.",
        "질문: paired statistics는 무엇을 비교한 건가요?",
        "방어: 같은 500개 item에 대해 두 모델의 prediction을 맞붙여 McNemar와 paired bootstrap delta를 본 것입니다. 독립 표본 비교가 아니라 common-item paired comparison입니다.",
    ]
    b = make("APPENDIX", "Metric Evidence: Reliability", "ECE | output contract | aggregate only", metric_notes)
    add_evidence_plot_card(b, plots_dir / "07_check500_calibration_contract.png", 0.82, 1.55, 11.65, 4.95, "Calibration and Output Contract", "model-level aggregate metrics", "7F63B8")
    slides.append(b.build(b.notes))

    b = make("APPENDIX", "Metric Evidence: Paired Statistics", "common-item comparison | bootstrap CI | McNemar", metric_notes)
    add_evidence_plot_card(b, plots_dir / "06_check500_pairwise_delta_ci.png", 0.82, 1.55, 11.65, 4.95, "Paired Accuracy Delta", "same 500 items", "0B1F33")
    slides.append(b.build(b.notes))

    b = make("APPENDIX", "Metric Evidence: Runtime Cost", "latency | memory | local runtime", metric_notes)
    add_evidence_plot_card(b, plots_dir / "14_check500_runtime_cost_by_method.png", 0.82, 1.55, 11.65, 4.95, "Runtime Cost by Method", "500-item runtime view", "3A9D5D")
    slides.append(b.build(b.notes))

    # Appendix J aggregate metric evidence matrix
    b = make("APPENDIX", "Aggregate Metric Evidence Matrix", "task-level accuracy | reliability | runtime | paired evidence", [
        "질문: 특정 모델이 어떤 문항을 맞히고 틀렸나요?",
        "방어: 이 appendix에서는 개별 문항의 정답/오답을 공개하지 않고, task-level aggregate만 보여줍니다. 성능 claim은 표본 전체 또는 task slice 전체에서 나온 수치로만 방어합니다.",
        "질문: G3가 정말 더 좋은가요?",
        "방어: 전체 accuracy는 G3가 99.2%이고, G4/G5/B3 대비 높습니다. 특히 Antonym slice에서 78.6%로 차이가 큽니다. 다만 Antonym은 n=14라 hard-slice signal로만 설명합니다.",
        "질문: 작은 G5를 쓸 수 있나요?",
        "방어: G5 1.5B는 97.6%, p50 341ms이고 G5 0.5B는 96.6%, p50 302ms, 987MB입니다. 품질 상한은 G3이지만, 운영 후보로는 G5가 설득력 있습니다.",
        "질문: 통계적으로 확실한 비교와 아닌 비교는 무엇인가요?",
        "방어: B0 대비 B3/G5는 매우 큰 차이입니다. 반면 B3와 G5 0.5B는 +0.8pp 수준이고 McNemar p=0.289라 우열 claim을 하지 않습니다.",
    ])
    headers = ["Model", "Acc", "Antonym", "p50", "RAM/VRAM", "ECE", "Strict err"]

    def task_pct(metrics: dict[str, Any], task: str) -> str:
        return pct(float(metrics.get("task_accuracy", {}).get(task, 0.0)))

    matrix_rows = [
        ("B0", [pct(b0["accuracy"]), task_pct(b0, "Antonym Selection"), fmt_ms(b0["latency_p50"]), fmt_mb(b0["peak_VRAM_or_RAM"]), pct(b0["ece"]), pct(b0["strict_parse_error_rate"])], "1B8A8F"),
        ("B3", [pct(b3["accuracy"]), task_pct(b3, "Antonym Selection"), fmt_ms(b3["latency_p50"]), fmt_mb(b3["peak_VRAM_or_RAM"]), pct(b3["ece"]), pct(b3["strict_parse_error_rate"])], "3A9D5D"),
        ("G3", [pct(g3["accuracy"]), task_pct(g3, "Antonym Selection"), fmt_ms(g3["latency_p50"]), fmt_mb(g3["peak_VRAM_or_RAM"]), pct(g3["ece"]), pct(g3["strict_parse_error_rate"])], "C84A3A"),
        ("G4 4bit", [pct(g4["accuracy"]), task_pct(g4, "Antonym Selection"), fmt_ms(g4["latency_p50"]), fmt_mb(g4["peak_VRAM_or_RAM"]), pct(g4["ece"]), pct(g4["strict_parse_error_rate"])], "7F63B8"),
        ("G5 0.5B", [pct(g05["accuracy"]), task_pct(g05, "Antonym Selection"), fmt_ms(g05["latency_p50"]), fmt_mb(g05["peak_VRAM_or_RAM"]), pct(g05["ece"]), pct(g05["strict_parse_error_rate"])], "3A9D5D"),
        ("G5 1.5B", [pct(g15["accuracy"]), task_pct(g15, "Antonym Selection"), fmt_ms(g15["latency_p50"]), fmt_mb(g15["peak_VRAM_or_RAM"]), pct(g15["ece"]), pct(g15["strict_parse_error_rate"])], "1B8A8F"),
        ("H1", [pct(h1["accuracy"]), task_pct(h1, "Antonym Selection"), fmt_ms(h1["latency_p50"]), fmt_mb(h1["peak_VRAM_or_RAM"]), pct(h1["ece"]), pct(h1["strict_parse_error_rate"])], "E79A32"),
    ]
    add_metric_matrix(b, matrix_rows, headers, 0.70, 1.58)
    b.rect(0.70, 5.38, 11.90, 0.96, fill="FFFFFF", line="D9DEE5", radius=True)
    p_b0_b3 = paired_result(check_root, "B0_vs_B3")
    p_b3_g05 = paired_result(check_root, "B3_vs_G5_Qwen0p5_G1")
    p_g3_g4 = paired_result(check_root, "G3_Qwen_vs_G4_Qwen_4bit")

    def paired_chip(x: float, title: str, report: dict[str, Any], accent: str) -> None:
        delta = float((report.get("paired_bootstrap_accuracy_delta") or {}).get("delta", 0.0))
        p_value = float((report.get("mcnemar") or {}).get("p_value", 1.0))
        builder_text = [f"{pp(delta)} delta", f"McNemar p={p_value:.3g}", f"common n={int(report.get('n_common', 0))}"]
        b.textbox(x, 5.58, 2.05, 0.16, [title], size=8, color=accent, bold=True, margin=0)
        b.textbox(x, 5.80, 2.05, 0.42, builder_text, size=7, color="111827", margin=0, line_spacing=950)

    paired_chip(1.00, "B0 vs B3", p_b0_b3, "1B8A8F")
    paired_chip(4.55, "B3 vs G5 0.5B", p_b3_g05, "3A9D5D")
    paired_chip(8.10, "G3 vs G4 4bit", p_g3_g4, "7F63B8")
    b.rect(10.74, 5.62, 1.32, 0.42, fill="F7F9FB", line="D9DEE5", radius=True)
    b.textbox(10.82, 5.75, 1.15, 0.12, ["aggregate only"], size=7, color="5F6B7A", bold=True, margin=0, align="c")
    slides.append(b.build(b.notes))

    return slides


def build_pptx(slides: list[Slide], out_path: Path, template_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    media_map: dict[Path, str] = {}
    media_counter = 1
    media_exts: set[str] = set()

    with ZipFile(template_path) as template, ZipFile(out_path, "w", ZIP_DEFLATED) as out:
        # Copy reusable presentation infrastructure from raw_plan.pptx.
        reusable = [
            "ppt/slideMasters/slideMaster1.xml",
            "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            "ppt/slideLayouts/slideLayout1.xml",
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            "ppt/notesMasters/notesMaster1.xml",
            "ppt/notesMasters/_rels/notesMaster1.xml.rels",
            "ppt/theme/theme1.xml",
            "ppt/theme/theme2.xml",
            "ppt/presProps.xml",
            "ppt/viewProps.xml",
            "ppt/tableStyles.xml",
        ]
        for name in reusable:
            out.writestr(name, template.read(name))

        out.writestr("_rels/.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
        ]))

        pres_rels = [
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml"),
        ]
        for i in range(1, len(slides) + 1):
            pres_rels.append((f"rId{i+1}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", f"slides/slide{i}.xml"))
        pres_rels.extend([
            ("rId100", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster", "notesMasters/notesMaster1.xml"),
            ("rId101", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps", "presProps.xml"),
            ("rId102", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps", "viewProps.xml"),
            ("rId103", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles", "tableStyles.xml"),
            ("rId104", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "theme/theme1.xml"),
        ])
        out.writestr("ppt/_rels/presentation.xml.rels", rels_xml(pres_rels))
        out.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        out.writestr("docProps/core.xml", core_xml())
        out.writestr("docProps/app.xml", app_xml(len(slides)))

        # Write slides, notes, and media.
        for i, slide in enumerate(slides, 1):
            out.writestr(f"ppt/slides/slide{i}.xml", slide_xml(slide))
            slide_rels = [
                ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"),
                ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide", f"../notesSlides/notesSlide{i}.xml"),
            ]
            image_rels: list[tuple[str, str, str]] = []
            for rel_id, image_path in slide.image_rels:
                if image_path not in media_map:
                    media_name = f"image{media_counter}{image_path.suffix.lower()}"
                    media_counter += 1
                    media_map[image_path] = media_name
                    media_exts.add(image_path.suffix.lower().lstrip("."))
                    out.writestr(f"ppt/media/{media_name}", image_path.read_bytes())
                image_rels.append((rel_id, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"../media/{media_map[image_path]}"))
            slide_rels.extend(image_rels)
            out.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rels_xml(slide_rels))

            out.writestr(f"ppt/notesSlides/notesSlide{i}.xml", make_notes_xml(slide.notes))
            out.writestr(
                f"ppt/notesSlides/_rels/notesSlide{i}.xml.rels",
                rels_xml([
                    ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster", "../notesMasters/notesMaster1.xml"),
                    ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", f"../slides/slide{i}.xml"),
                ]),
            )

        out.writestr("[Content_Types].xml", content_types_xml(len(slides), media_exts or {"png"}))


def main() -> None:
    check_root = Path("runs/no_api_no_retrain_check")
    plots_dir = Path("reports/presentation_plots")
    out_path = Path(os.environ.get("OUT_PPTX", "reports/term_ai_methodology_results_presentation.pptx"))
    slides = build_slides(check_root, plots_dir)
    build_pptx(slides, out_path, Path("raw_plan.pptx"))
    print(json.dumps({"output": str(out_path), "slides": len(slides)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
