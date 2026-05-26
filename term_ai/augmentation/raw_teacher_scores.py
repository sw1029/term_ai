from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from term_ai.augmentation.teacher import OpenAITeacherClient
from term_ai.contracts import dumps_jsonl


def build_raw_score_prompt(payload: dict[str, Any]) -> str:
    return f"""You are a TOEIC business vocabulary teacher.

Return one valid JSON object only with:
- teacher_scores: exactly four numeric scores in option order, summing approximately to 1
- rationale: short Korean explanation

Scoring rules:
- Score the probability that each option is the correct answer.
- Use the provided answer_idx as ground truth, but still assign soft probability mass to plausible distractors.
- Do not include hidden chain-of-thought.

Raw MCQ payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def load_completed_item_ids(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.exists():
        return set()
    completed: set[str] = set()
    with output.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"existing raw teacher score output has invalid JSON at line {line_number}: {output}") from exc
            item_id = row.get("item_id")
            if item_id:
                completed.add(str(item_id))
    return completed


def _valid_scores(scores: object) -> bool:
    return (
        isinstance(scores, list)
        and len(scores) == 4
        and all(isinstance(score, (int, float)) for score in scores)
        and sum(float(score) for score in scores) > 0
    )


def normalize_scores(scores: list[float]) -> list[float]:
    total = float(sum(scores))
    return [float(score) / total for score in scores]


def generate_raw_teacher_scores(
    metadata_path: str | Path,
    output_path: str | Path,
    model: str = "gpt-5.4-mini",
    env_path: str | Path = ".env",
    requests_per_second: float = 5.0,
    limit: int | None = None,
    reasoning_effort: str | None = None,
    resume: bool = False,
) -> dict[str, int]:
    if requests_per_second <= 0:
        raise ValueError("requests_per_second must be positive")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed_item_ids = load_completed_item_ids(output) if resume else set()
    output_mode = "a" if resume else "w"
    teacher = OpenAITeacherClient(model=model, env_path=str(env_path), reasoning_effort=reasoning_effort)
    min_interval = 1.0 / requests_per_second
    last_request_at: float | None = None
    counts = {"written": 0, "skipped_existing": 0, "invalid_scores": 0}

    with open(metadata_path, "r", encoding="utf-8") as input_handle, output.open(
        output_mode, encoding="utf-8", newline="\n"
    ) as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            if limit is not None and counts["written"] >= limit:
                break
            row = json.loads(line)
            item_id = str(row["item_id"])
            if item_id in completed_item_ids:
                counts["skipped_existing"] += 1
                continue
            if last_request_at is not None:
                elapsed = time.monotonic() - last_request_at
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            last_request_at = time.monotonic()

            result = teacher.generate_json(build_raw_score_prompt(row.get("payload") or {}))
            scores = result.get("teacher_scores")
            if not _valid_scores(scores):
                counts["invalid_scores"] += 1
                continue
            normalized = normalize_scores([float(score) for score in scores])
            payload = dict(row.get("payload") or {})
            payload["teacher_scores"] = normalized
            scored = dict(row)
            scored["teacher_scores"] = normalized
            scored["teacher_rationale"] = result.get("rationale")
            scored["teacher_score_model"] = model
            if reasoning_effort:
                scored["teacher_score_reasoning_effort"] = reasoning_effort
            scored["payload"] = payload
            output_handle.write(dumps_jsonl(scored))
            counts["written"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher_scores for raw MCQ metadata.")
    parser.add_argument("--metadata", default="data/metadata/raw_mcq_v1.jsonl")
    parser.add_argument("--output", default="data/metadata/raw_teacher_scores_v1.jsonl")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--requests-per-second", type=float, default=5.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    counts = generate_raw_teacher_scores(
        metadata_path=args.metadata,
        output_path=args.output,
        model=args.model,
        env_path=args.env,
        requests_per_second=args.requests_per_second,
        limit=args.limit,
        reasoning_effort=args.reasoning_effort,
        resume=args.resume,
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
