from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from term_ai.augmentation.teacher import OpenAITeacherClient
from term_ai.contracts import dumps_jsonl, normalize_openai_model_id


def build_explanation_judge_prompt(row: dict[str, Any]) -> str:
    return f"""You are an independent evaluator for TOEIC business vocabulary explanations.

Return one valid JSON object only with:
- item_id: copied from the input
- semantic_correctness: integer 0, 1, or 2
- reasoning_faithfulness: integer 0, 1, or 2
- hallucination_check: "pass" or "fail"
- final_score: number from 0 to 2
- notes: short Korean explanation

Evaluate whether the answer explanation is semantically correct, whether the
reasoning is faithful to the word/context/options, and whether it invents
unsupported facts.

Prediction row:
{json.dumps(row, ensure_ascii=False, indent=2)}
"""


def judge_explanations(
    predictions_path: str | Path,
    output_path: str | Path,
    judge_model: str,
    generator_model: str | None = None,
    env_path: str | Path = ".env",
    requests_per_second: float = 1.0,
    limit: int | None = None,
    enforce_model_separation: bool = True,
) -> dict[str, int]:
    if requests_per_second <= 0:
        raise ValueError("requests_per_second must be positive")
    judge_model = normalize_openai_model_id(judge_model)
    generator_model = normalize_openai_model_id(generator_model) if generator_model else None
    if enforce_model_separation and generator_model and judge_model == generator_model:
        raise ValueError("explanation judge model must differ from generator model")

    client = OpenAITeacherClient(model=judge_model, env_path=str(env_path))
    min_interval = 1.0 / requests_per_second
    last_request_at: float | None = None
    counts = {"written": 0, "hallucination_fail": 0}

    with open(predictions_path, "r", encoding="utf-8") as input_handle, open(
        output_path, "w", encoding="utf-8", newline="\n"
    ) as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            if limit is not None and counts["written"] >= limit:
                break
            row = json.loads(line)
            row_generator = normalize_openai_model_id(str(row.get("generator_model") or generator_model or ""))
            if enforce_model_separation and row_generator and row_generator == judge_model:
                raise ValueError(f"judge model must differ from generator model for item {row.get('item_id')}")
            if last_request_at is not None:
                elapsed = time.monotonic() - last_request_at
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            last_request_at = time.monotonic()
            result = client.generate_json(build_explanation_judge_prompt(row))
            result["item_id"] = row.get("item_id")
            result.setdefault("judge_model", judge_model)
            if result.get("hallucination_check") == "fail":
                counts["hallucination_fail"] += 1
            counts["written"] += 1
            output_handle.write(dumps_jsonl(result))
    return counts


def summarize_explanation_judgments(judgments_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with open(judgments_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    n = len(rows)
    summary = {
        "n": n,
        "semantic_correctness_avg": sum(float(row.get("semantic_correctness", 0)) for row in rows) / n if n else 0.0,
        "reasoning_faithfulness_avg": sum(float(row.get("reasoning_faithfulness", 0)) for row in rows) / n if n else 0.0,
        "hallucination_fail_rate": sum(1 for row in rows if row.get("hallucination_check") == "fail") / n if n else 0.0,
        "final_score_avg": sum(float(row.get("final_score", 0)) for row in rows) / n if n else 0.0,
    }
    Path(output_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge explanation quality, faithfulness, and hallucination.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    judge = subparsers.add_parser("judge")
    judge.add_argument("--predictions", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--judge-model", required=True)
    judge.add_argument("--generator-model")
    judge.add_argument("--env", default=".env")
    judge.add_argument("--requests-per-second", type=float, default=1.0)
    judge.add_argument("--limit", type=int)
    judge.add_argument("--allow-same-model", action="store_true")

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--judgments", required=True)
    summarize.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "judge":
        result = judge_explanations(
            args.predictions,
            args.output,
            judge_model=args.judge_model,
            generator_model=args.generator_model,
            env_path=args.env,
            requests_per_second=args.requests_per_second,
            limit=args.limit,
            enforce_model_separation=not args.allow_same_model,
        )
    else:
        result = summarize_explanation_judgments(args.judgments, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
