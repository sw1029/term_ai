from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from term_ai.augmentation.teacher import OpenAITeacherClient
from term_ai.contracts import dumps_jsonl, normalize_openai_model_id


def build_judge_prompt(payload: dict[str, Any]) -> str:
    return f"""You are an independent validator for TOEIC business vocabulary MCQ data.

Return one valid JSON object only with:
- item_id: copied by the caller if present, or empty string
- semantic_correctness: integer 0, 1, or 2
- distractor_validity: integer 0, 1, or 2
- context_naturalness: integer 0, 1, or 2
- leakage_check: "pass" or "fail"
- final_decision: "accept" or "reject"
- notes: short Korean explanation

Accept only if semantic_correctness=2, distractor_validity>=1,
context_naturalness>=1, and leakage_check="pass".

Candidate payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def judge_metadata(
    metadata_path: str | Path,
    output_path: str | Path,
    model: str = "gpt-5.4-mini",
    env_path: str | Path = ".env",
    requests_per_second: float = 1.0,
    limit: int | None = None,
    generator_model: str | None = None,
    enforce_model_separation: bool = True,
    reasoning_effort: str | None = None,
) -> dict[str, int]:
    if requests_per_second <= 0:
        raise ValueError("requests_per_second must be positive")
    model = normalize_openai_model_id(model)
    generator_model = normalize_openai_model_id(generator_model) if generator_model else None
    if enforce_model_separation and generator_model and model == generator_model:
        raise ValueError("judge model must differ from generator model")
    teacher = OpenAITeacherClient(model=model, env_path=str(env_path), reasoning_effort=reasoning_effort)
    min_interval = 1.0 / requests_per_second
    last_request_at: float | None = None
    counts = {"written": 0, "accept": 0, "reject": 0}

    with open(metadata_path, "r", encoding="utf-8") as input_handle, open(
        output_path, "w", encoding="utf-8", newline="\n"
    ) as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            if limit is not None and counts["written"] >= limit:
                break
            row = json.loads(line)
            row_generator = normalize_openai_model_id(str(row.get("generator_model") or generator_model or ""))
            if enforce_model_separation and row_generator and row_generator == model:
                raise ValueError(f"judge model must differ from generator model for item {row.get('item_id')}")
            if last_request_at is not None:
                elapsed = time.monotonic() - last_request_at
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            last_request_at = time.monotonic()
            result = teacher.generate_json(build_judge_prompt(row.get("payload") or {}))
            result["item_id"] = row["item_id"]
            result.setdefault("judge_model", model)
            if reasoning_effort:
                result.setdefault("judge_reasoning_effort", reasoning_effort)
            decision = str(result.get("final_decision", "reject"))
            counts["accept" if decision == "accept" else "reject"] += 1
            counts["written"] += 1
            output_handle.write(dumps_jsonl(result))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM judge validation for augmentation metadata.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--requests-per-second", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generator-model")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"])
    parser.add_argument("--allow-same-model", action="store_true")
    args = parser.parse_args()
    counts = judge_metadata(
        metadata_path=args.metadata,
        output_path=args.output,
        model=args.model,
        env_path=args.env,
        requests_per_second=args.requests_per_second,
        limit=args.limit,
        generator_model=args.generator_model,
        enforce_model_separation=not args.allow_same_model,
        reasoning_effort=args.reasoning_effort,
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
