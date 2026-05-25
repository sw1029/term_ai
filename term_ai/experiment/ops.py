from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Iterator


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    state: dict[str, float] = {"latency_ms": 0.0}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["latency_ms"] = (time.perf_counter() - start) * 1000


def gpu_memory_snapshot() -> dict[str, float]:
    try:
        import torch
    except ImportError:
        return {"peak_vram_mb": 0.0}
    if not torch.cuda.is_available():
        return {"peak_vram_mb": 0.0}
    return {"peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024))}


def memory_snapshot() -> dict[str, float]:
    try:
        import psutil
    except ImportError:
        return {"ram_mb": 0.0, **gpu_memory_snapshot()}
    process = psutil.Process()
    return {"ram_mb": float(process.memory_info().rss / (1024 * 1024)), **gpu_memory_snapshot()}


def tokens_per_second(output_tokens: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return float(output_tokens) / (latency_ms / 1000)


def cost_per_1000_questions(total_cost: float, question_count: int) -> float:
    if question_count <= 0:
        return 0.0
    return total_cost / question_count * 1000
