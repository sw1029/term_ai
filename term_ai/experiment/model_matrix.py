from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    experiment_id: str
    method: str
    training_data: str
    purpose: str
    group: str
    requires_training: bool = False
    requires_generation: bool = False
    uses_kd: bool = False
    quantization: str | None = None
    model_name_or_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_MATRIX: list[ModelSpec] = [
    ModelSpec("B0", "mxbai threshold", "none", "current scoring reproduction", "baseline"),
    ModelSpec("B1", "mxbai + logistic", "raw train", "linear scorer comparison", "baseline", True),
    ModelSpec("B2", "mxbai + MLP", "raw train", "nonlinear scorer comparison", "baseline", True),
    ModelSpec("B3", "cross-encoder/reranker", "raw train or zero-shot", "strong discriminative baseline", "baseline", True),
    ModelSpec("B4", "Qwen/API recheck", "none", "current fallback comparison", "baseline", requires_generation=True),
    ModelSpec("G0-Gemma", "gemma2 2b zero-shot", "none", "small LM zero-shot", "small_lm", requires_generation=True, model_name_or_path="google/gemma-2-2b-it"),
    ModelSpec("G0-Qwen", "qwen2.5 3b zero-shot", "none", "small LM zero-shot", "small_lm", requires_generation=True, model_name_or_path="Qwen/Qwen2.5-3B-Instruct"),
    ModelSpec("G0-BitNet", "bitnet b1.58 2b zero-shot", "none", "1-bit small LM zero-shot", "small_lm", requires_generation=True, model_name_or_path="microsoft/bitnet-b1.58-2B-4T"),
    ModelSpec("G1-Gemma", "gemma2 2b LoRA SFT", "raw train", "SFT effect", "small_lm", True, True, model_name_or_path="google/gemma-2-2b-it"),
    ModelSpec("G1-Qwen", "qwen2.5 3b LoRA SFT", "raw train", "SFT effect", "small_lm", True, True, model_name_or_path="Qwen/Qwen2.5-3B-Instruct"),
    ModelSpec("G1-BitNet", "bitnet b1.58 2b LoRA SFT", "raw train", "SFT effect on 1-bit LM", "small_lm", True, True, model_name_or_path="microsoft/bitnet-b1.58-2B-4T"),
    ModelSpec("G2-Gemma", "gemma2 2b LoRA SFT", "raw + judge-validated aug", "augmentation effect", "small_lm", True, True, model_name_or_path="google/gemma-2-2b-it"),
    ModelSpec("G2-Qwen", "qwen2.5 3b LoRA SFT", "raw + judge-validated aug", "augmentation effect", "small_lm", True, True, model_name_or_path="Qwen/Qwen2.5-3B-Instruct"),
    ModelSpec("G2-BitNet", "bitnet b1.58 2b LoRA SFT", "raw + judge-validated aug", "augmentation effect on 1-bit LM", "small_lm", True, True, model_name_or_path="microsoft/bitnet-b1.58-2B-4T"),
    ModelSpec("G3-Gemma", "gemma2 2b LoRA SFT + KD", "raw + judge-validated aug + teacher score", "KD effect", "small_lm", True, True, True, model_name_or_path="google/gemma-2-2b-it"),
    ModelSpec("G3-Qwen", "qwen2.5 3b LoRA SFT + KD", "raw + judge-validated aug + teacher score", "KD effect", "small_lm", True, True, True, model_name_or_path="Qwen/Qwen2.5-3B-Instruct"),
    ModelSpec("G3-BitNet", "bitnet b1.58 2b LoRA SFT + KD", "raw + judge-validated aug + teacher score", "KD effect on 1-bit LM", "small_lm", True, True, True, model_name_or_path="microsoft/bitnet-b1.58-2B-4T"),
    ModelSpec("G4-8bit", "G3 checkpoint 8bit", "same G3 checkpoint", "quantization effect", "quantization", False, True, True, "8bit"),
    ModelSpec("G4-4bit", "G3 checkpoint 4bit", "same G3 checkpoint", "quantization effect", "quantization", False, True, True, "4bit"),
    ModelSpec("G5-Qwen0p5-ZS", "qwen2.5 0.5b zero-shot", "none", "0.5B student baseline", "g5_student", requires_generation=True, model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct"),
    ModelSpec("G5-Qwen0p5-G1", "qwen2.5 0.5b LoRA SFT", "raw train", "0.5B raw SFT baseline", "g5_student", True, True, model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct"),
    ModelSpec("G5-Qwen0p5-G2", "qwen2.5 0.5b LoRA SFT", "raw + judge-validated aug", "0.5B augmentation baseline", "g5_student", True, True, model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct"),
    ModelSpec("G5-Qwen0p5-GPTKD", "qwen2.5 0.5b LoRA KD", "raw + aug + GPT teacher score", "0.5B existing teacher-score KD", "g5_student", True, True, True, model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct"),
    ModelSpec("G5-Qwen0p5-3BKD-T1", "qwen2.5 0.5b LoRA KD", "raw + aug + G3 Qwen logits T1", "0.5B 3B-teacher compression", "g5_student", True, True, True, model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct"),
    ModelSpec("G5-Qwen0p5-3BKD-T2", "qwen2.5 0.5b LoRA KD", "raw + aug + G3 Qwen logits T2", "0.5B 3B-teacher compression", "g5_student", True, True, True, model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct"),
    ModelSpec("G5-Qwen0p5-3BKD-T4", "qwen2.5 0.5b LoRA KD", "raw + aug + G3 Qwen logits T4", "0.5B 3B-teacher compression", "g5_student", True, True, True, model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct"),
    ModelSpec("G5-Qwen1p5-ZS", "qwen2.5 1.5b zero-shot", "none", "1.5B student baseline", "g5_student", requires_generation=True, model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct"),
    ModelSpec("G5-Qwen1p5-G1", "qwen2.5 1.5b LoRA SFT", "raw train", "1.5B raw SFT baseline", "g5_student", True, True, model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct"),
    ModelSpec("G5-Qwen1p5-G2", "qwen2.5 1.5b LoRA SFT", "raw + judge-validated aug", "1.5B augmentation baseline", "g5_student", True, True, model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct"),
    ModelSpec("G5-Qwen1p5-GPTKD", "qwen2.5 1.5b LoRA KD", "raw + aug + GPT teacher score", "1.5B existing teacher-score KD", "g5_student", True, True, True, model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct"),
    ModelSpec("G5-Qwen1p5-3BKD-T1", "qwen2.5 1.5b LoRA KD", "raw + aug + G3 Qwen logits T1", "1.5B 3B-teacher compression", "g5_student", True, True, True, model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct"),
    ModelSpec("G5-Qwen1p5-3BKD-T2", "qwen2.5 1.5b LoRA KD", "raw + aug + G3 Qwen logits T2", "1.5B 3B-teacher compression", "g5_student", True, True, True, model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct"),
    ModelSpec("G5-Qwen1p5-3BKD-T4", "qwen2.5 1.5b LoRA KD", "raw + aug + G3 Qwen logits T4", "1.5B 3B-teacher compression", "g5_student", True, True, True, model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct"),
    ModelSpec("E1", "embedding scorer KD", "raw + teacher score", "KD without generative LM", "baseline", True, False, True),
    ModelSpec("H1", "scorer + fallback hybrid", "dev-tuned policy", "deployment structure", "hybrid"),
]


def get_model_spec(experiment_id: str) -> ModelSpec:
    for spec in MODEL_MATRIX:
        if spec.experiment_id == experiment_id:
            return spec
    raise KeyError(f"unknown experiment_id: {experiment_id}")


def model_matrix_as_dicts() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in MODEL_MATRIX]
