from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

from term_ai.contracts import RAW_GT_STATUS, normalize_openai_model_id, write_jsonl
from term_ai.env import load_key_value_env, resolve_openai_api_key
from term_ai.experiment.metrics import summarize_predictions
from term_ai.experiment.mcq import load_mcq_items, parse_answer_letter, prediction_row
from term_ai.experiment.ops import memory_snapshot, timed
from term_ai.experiment.test_lock import enforce_final_test_once


@dataclass(frozen=True)
class RecheckResponse:
    payload: dict[str, Any]
    raw_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _env_value(env_path: str | Path, key: str) -> str | None:
    return os.environ.get(key) or load_key_value_env(env_path).get(key)


def _resolve_api_key(provider: str, env_path: str | Path, api_key_env: str | None) -> str | None:
    if api_key_env:
        return _env_value(env_path, api_key_env)
    if provider == "openai":
        return resolve_openai_api_key(env_path)
    return (
        _env_value(env_path, "QWEN_API_KEY")
        or _env_value(env_path, "DASHSCOPE_API_KEY")
        or _env_value(env_path, "OPENAI_API_KEY")
    )


def _usage_value(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(name) or 0)
    return int(getattr(usage, name, 0) or 0)


def _load_pricing(
    pricing_path: str | Path | None,
    provider: str,
    model: str,
) -> tuple[float, float, str]:
    if pricing_path is None:
        return 0.0, 0.0, "not_configured"
    path = Path(pricing_path)
    if not path.exists():
        raise FileNotFoundError(f"pricing file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in (f"{provider}/{model}", model, provider, "default"):
        value = data.get(key)
        if isinstance(value, dict):
            return (
                float(value.get("input_cost_per_1m_tokens", 0.0)),
                float(value.get("output_cost_per_1m_tokens", 0.0)),
                f"pricing_file:{key}",
            )
    return 0.0, 0.0, "pricing_file_no_match"


class LLMRecheckClient:
    """OpenAI SDK based recheck client, including OpenAI-compatible Qwen endpoints."""

    def __init__(
        self,
        provider: str,
        model: str,
        env_path: str | Path = ".env",
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = normalize_openai_model_id(model) if provider == "openai" else model
        self.base_url = base_url
        if provider in {"openai-compatible", "qwen-compatible"} and not base_url:
            raise ValueError(f"{provider} requires base_url for an OpenAI-compatible endpoint")
        self.api_key = _resolve_api_key(provider, env_path, api_key_env)
        if not self.api_key:
            raise RuntimeError(f"API key was not found for provider={provider}")

    def generate_json_with_usage(self, prompt: str) -> RecheckResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the llm extra to use API recheck: pip install -e .[llm]") from exc

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)
        messages = [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = client.chat.completions.create(model=self.model, messages=messages, temperature=0)
        text = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {}
        usage = getattr(response, "usage", None)
        return RecheckResponse(
            payload=payload,
            raw_text=text,
            prompt_tokens=_usage_value(usage, "prompt_tokens"),
            completion_tokens=_usage_value(usage, "completion_tokens"),
            total_tokens=_usage_value(usage, "total_tokens"),
        )


def _load_primary_predictions(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["item_id"])] = row
    return rows


def _needs_recheck(item: Any, primary: dict[str, Any] | None, confidence_threshold: float) -> bool:
    if primary is not None and float(primary.get("confidence", 0.0)) < confidence_threshold:
        return True
    tags = set(item.stress_tags)
    if {"short_answer", "high_similarity", "polysemy"} & tags:
        return True
    if item.embedding_top2_similarity is not None and item.embedding_top2_similarity >= 1.0:
        return True
    if item.embedding_top2_gap is not None and item.embedding_top2_gap <= 0.03:
        return True
    if len(item.meaning_ko.replace(" ", "")) <= 3:
        return True
    return False


def run_api_recheck(
    metadata_path: str | Path,
    output_dir: str | Path,
    provider: str = "openai",
    model: str = "gpt-5.4-mini",
    env_path: str | Path = ".env",
    base_url: str | None = None,
    api_key_env: str | None = None,
    eval_split: str = "dev",
    min_status: str = RAW_GT_STATUS,
    requests_per_second: float = 1.0,
    limit: int | None = None,
    cost_per_1000_questions: float = 0.0,
    input_cost_per_1m_tokens: float = 0.0,
    output_cost_per_1m_tokens: float = 0.0,
    pricing_path: str | Path | None = None,
    primary_predictions_path: str | Path | None = None,
    confidence_threshold: float = 0.7,
    fallback_only: bool = True,
    final_test_once: bool = True,
    test_lock_dir: str | Path | None = None,
) -> dict[str, Any]:
    if requests_per_second <= 0:
        raise ValueError("requests_per_second must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enforce_final_test_once(output, "B4", eval_split, enabled=final_test_once, lock_dir=test_lock_dir)

    all_items = [item for item in load_mcq_items(metadata_path, min_status=min_status) if item.split == eval_split]
    if not all_items:
        raise ValueError(f"no API recheck eval items: split={eval_split}, min_status={min_status}")
    items = all_items
    primary_predictions = _load_primary_predictions(primary_predictions_path)
    if fallback_only:
        items = [
            item
            for item in items
            if _needs_recheck(item, primary_predictions.get(item.item_id), confidence_threshold)
        ]
    if limit is not None:
        items = items[:limit]
    client = LLMRecheckClient(
        provider=provider,
        model=model,
        env_path=env_path,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    pricing_input, pricing_output, pricing_source = _load_pricing(pricing_path, provider, client.model)
    if input_cost_per_1m_tokens == 0.0:
        input_cost_per_1m_tokens = pricing_input
    if output_cost_per_1m_tokens == 0.0:
        output_cost_per_1m_tokens = pricing_output
    min_interval = 1.0 / requests_per_second
    last_request_at: float | None = None
    predictions: list[dict[str, Any]] = []
    total_cost = 0.0

    for item in items:
        if last_request_at is not None:
            elapsed = time.monotonic() - last_request_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        with timed() as state:
            last_request_at = time.monotonic()
            result = client.generate_json_with_usage(item.prompt())
        answer, confidence = parse_answer_letter(json.dumps(result.payload, ensure_ascii=False))
        usage_cost = (
            result.prompt_tokens / 1_000_000 * input_cost_per_1m_tokens
            + result.completion_tokens / 1_000_000 * output_cost_per_1m_tokens
        )
        manual_cost = cost_per_1000_questions / 1000 if cost_per_1000_questions and not usage_cost else 0.0
        estimated_cost = usage_cost + manual_cost
        total_cost += estimated_cost
        predictions.append(
            prediction_row(
                item,
                answer or "PARSE_ERROR",
                confidence if confidence is not None else 0.0,
                latency_ms=state["latency_ms"],
                extra={
                    "parse_error": answer is None,
                    "raw_response": result.payload if result.payload else result.raw_text,
                    "provider": provider,
                    "model": client.model,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                    "estimated_cost_usd": estimated_cost,
                    "cost_source": "token_usage" if usage_cost else "manual_per_question" if manual_cost else pricing_source,
                    "fallback_reason": "api_recheck",
                    "primary_confidence": (primary_predictions.get(item.item_id) or {}).get("confidence"),
                    **memory_snapshot(),
                },
            )
        )

    metrics = summarize_predictions(predictions)
    if total_cost > 0:
        metrics["total_estimated_cost_usd"] = total_cost
        metrics["cost_per_1000_questions"] = total_cost / max(1, len(predictions)) * 1000
    else:
        metrics["cost_per_1000_questions"] = cost_per_1000_questions
    metrics["provider"] = provider
    metrics["model"] = client.model
    metrics["fallback_only"] = fallback_only
    metrics["fallback_rate"] = len(predictions) / len(all_items) if all_items else 0.0
    metrics["pricing_source"] = pricing_source
    write_jsonl(output / "prediction_log.jsonl", predictions)
    (output / "metric_log.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B4 API recheck evaluation.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider", choices=["openai", "openai-compatible", "qwen-compatible"], default="openai")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--min-status", default=RAW_GT_STATUS)
    parser.add_argument("--requests-per-second", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cost-per-1000-questions", type=float, default=0.0)
    parser.add_argument("--input-cost-per-1m-tokens", type=float, default=0.0)
    parser.add_argument("--output-cost-per-1m-tokens", type=float, default=0.0)
    parser.add_argument("--pricing-path")
    parser.add_argument("--primary-predictions")
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--test-lock-dir")
    parser.add_argument("--all-items", action="store_true")
    parser.add_argument("--allow-repeat-test", action="store_true")
    args = parser.parse_args()
    metrics = run_api_recheck(
        args.metadata,
        args.output_dir,
        provider=args.provider,
        model=args.model,
        env_path=args.env,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        eval_split=args.eval_split,
        min_status=args.min_status,
        requests_per_second=args.requests_per_second,
        limit=args.limit,
        cost_per_1000_questions=args.cost_per_1000_questions,
        input_cost_per_1m_tokens=args.input_cost_per_1m_tokens,
        output_cost_per_1m_tokens=args.output_cost_per_1m_tokens,
        pricing_path=args.pricing_path,
        primary_predictions_path=args.primary_predictions,
        confidence_threshold=args.confidence_threshold,
        fallback_only=not args.all_items,
        final_test_once=not args.allow_repeat_test,
        test_lock_dir=args.test_lock_dir,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
