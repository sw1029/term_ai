from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from term_ai.augmentation.prompts import PROMPT_VERSION, build_generation_prompt
from term_ai.augmentation.teacher import OpenAITeacherClient
from term_ai.contracts import AugmentationMetadata, TASK_RATIOS, TASK_TYPES, dumps_jsonl, stable_id


def _load_split_anchors(anchors_path: Path, split: str) -> list[dict[str, Any]]:
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    anchors: list[dict[str, Any]] = []
    with anchors_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            anchor = json.loads(line)
            if anchor.get("split") == split and not anchor.get("duplicate_of"):
                anchors.append(anchor)
    return anchors


def _task_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    counts = {task: int(total * ratio) for task, ratio in ratios.items()}
    remaining = total - sum(counts.values())
    ordered = sorted(ratios, key=lambda task: ratios[task], reverse=True)
    for idx in range(remaining):
        counts[ordered[idx % len(ordered)]] += 1
    return counts


def generate_split_batch(
    anchors_path: str | Path,
    output_path: str | Path,
    total: int,
    model: str = "gpt-5.4-mini",
    env_path: str | Path = ".env",
    requests_per_second: float = 1.0,
    ratios: dict[str, float] | None = None,
    split: str = "train",
    teacher_client: Any | None = None,
) -> dict[str, Any]:
    if requests_per_second <= 0:
        raise ValueError("requests_per_second must be positive")
    if total < 0:
        raise ValueError("total must be non-negative")
    ratios = ratios or TASK_RATIOS
    invalid_tasks = set(ratios) - TASK_TYPES
    if invalid_tasks:
        raise ValueError(f"invalid task ratios: {sorted(invalid_tasks)}")

    anchors = _load_split_anchors(Path(anchors_path), split)
    if not anchors:
        raise ValueError(f"no {split} anchors available")

    counts = _task_counts(total, ratios)
    teacher = teacher_client or OpenAITeacherClient(model=model, env_path=str(env_path))
    min_interval = 1.0 / requests_per_second
    last_request_at: float | None = None
    written_by_task: dict[str, int] = defaultdict(int)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        anchor_idx = 0
        for task_type, count in counts.items():
            for _ in range(count):
                anchor = anchors[anchor_idx % len(anchors)]
                anchor_idx += 1
                if last_request_at is not None:
                    elapsed = time.monotonic() - last_request_at
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                prompt = build_generation_prompt(task_type, anchor["word"], anchor["pos"], anchor["meaning"])
                last_request_at = time.monotonic()
                payload = teacher.generate_json(prompt)
                item_id = stable_id("aug", anchor["anchor_id"], task_type, payload)
                metadata = AugmentationMetadata(
                    item_id=item_id,
                    anchor_id=anchor["anchor_id"],
                    word_id=anchor["word_id"],
                    split=split,
                    status="aug_candidate",
                    prompt_version=PROMPT_VERSION,
                    generator_model=model,
                    payload=payload,
                    teacher_rationale=payload.get("rationale"),
                    teacher_scores=payload.get("teacher_scores"),
                )
                handle.write(dumps_jsonl(metadata.to_dict()))
                written_by_task[task_type] += 1

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output),
        "total_requested": total,
        "total_written": sum(written_by_task.values()),
        "task_counts": dict(written_by_task),
        "source_split": split,
        "requests_per_second": requests_per_second,
        "model": model,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def generate_train_batch(
    anchors_path: str | Path,
    output_path: str | Path,
    total: int,
    model: str = "gpt-5.4-mini",
    env_path: str | Path = ".env",
    requests_per_second: float = 1.0,
    ratios: dict[str, float] | None = None,
    teacher_client: Any | None = None,
) -> dict[str, Any]:
    return generate_split_batch(
        anchors_path=anchors_path,
        output_path=output_path,
        total=total,
        model=model,
        env_path=env_path,
        requests_per_second=requests_per_second,
        ratios=ratios,
        split="train",
        teacher_client=teacher_client,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split-aware augmentation orchestration.")
    parser.add_argument("--anchors", default="data/processed/anchors_v1.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--requests-per-second", type=float, default=1.0)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    args = parser.parse_args()
    output = args.output or f"data/aug/{args.split}_aug_candidate_v1.jsonl"
    manifest = generate_split_batch(
        anchors_path=args.anchors,
        output_path=output,
        total=args.total,
        model=args.model,
        env_path=args.env,
        requests_per_second=args.requests_per_second,
        split=args.split,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
