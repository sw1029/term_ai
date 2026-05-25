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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_MATRIX: list[ModelSpec] = [
    ModelSpec("B0", "mxbai threshold", "none", "current scoring reproduction", "baseline"),
    ModelSpec("B1", "mxbai + logistic", "raw train", "linear scorer comparison", "baseline", True),
    ModelSpec("B2", "mxbai + MLP", "raw train", "nonlinear scorer comparison", "baseline", True),
    ModelSpec("B3", "cross-encoder/reranker", "raw train or zero-shot", "strong discriminative baseline", "baseline", True),
    ModelSpec("B4", "Qwen/API recheck", "none", "current fallback comparison", "baseline", requires_generation=True),
    ModelSpec("G0-Gemma", "gemma2 2b zero-shot", "none", "small LM zero-shot", "small_lm", requires_generation=True),
    ModelSpec("G0-Qwen", "qwen2.5 3b zero-shot", "none", "small LM zero-shot", "small_lm", requires_generation=True),
    ModelSpec("G1-Gemma", "gemma2 2b LoRA SFT", "raw train", "SFT effect", "small_lm", True, True),
    ModelSpec("G1-Qwen", "qwen2.5 3b LoRA SFT", "raw train", "SFT effect", "small_lm", True, True),
    ModelSpec("G2-Gemma", "gemma2 2b LoRA SFT", "raw + approved aug", "augmentation effect", "small_lm", True, True),
    ModelSpec("G2-Qwen", "qwen2.5 3b LoRA SFT", "raw + approved aug", "augmentation effect", "small_lm", True, True),
    ModelSpec("G3-Gemma", "gemma2 2b LoRA SFT + KD", "raw + approved aug + teacher score", "KD effect", "small_lm", True, True, True),
    ModelSpec("G3-Qwen", "qwen2.5 3b LoRA SFT + KD", "raw + approved aug + teacher score", "KD effect", "small_lm", True, True, True),
    ModelSpec("G4-8bit", "G3 checkpoint 8bit", "same G3 checkpoint", "quantization effect", "quantization", False, True, True, "8bit"),
    ModelSpec("G4-4bit", "G3 checkpoint 4bit", "same G3 checkpoint", "quantization effect", "quantization", False, True, True, "4bit"),
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
