from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from term_ai.contracts import validate_sft_record, write_jsonl


PROMPT_TEMPLATE_VARIANTS = {
    "default": {
        "system": None,
        "user_prefix": None,
    },
    "concise": {
        "system": "You are a TOEIC business vocabulary judge. Answer with the best option letter and one short Korean reason.",
        "user_prefix": "Choose the best option for the vocabulary item.",
    },
    "json_strict": {
        "system": "You are a TOEIC business vocabulary judge. Return the selected option letter and a brief Korean reason.",
        "user_prefix": "Select exactly one option. Keep the answer concise.",
    },
}


def _read_sft(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                validate_sft_record(row)
            except Exception as exc:
                raise ValueError(f"SFT row {line_no} violates messages-only contract") from exc
            rows.append(row)
    return rows


def rewrite_sft_prompt_variant(record: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant not in PROMPT_TEMPLATE_VARIANTS:
        raise ValueError(f"unknown prompt variant: {variant}")
    if variant == "default":
        return json.loads(json.dumps(record, ensure_ascii=False))
    spec = PROMPT_TEMPLATE_VARIANTS[variant]
    updated = json.loads(json.dumps(record, ensure_ascii=False))
    messages = updated["messages"]
    if spec["system"] and messages and messages[0]["role"] == "system":
        messages[0]["content"] = spec["system"]
    if spec["user_prefix"]:
        for message in messages:
            if message["role"] == "user":
                message["content"] = f"{spec['user_prefix']}\n\n{message['content']}"
                break
    validate_sft_record(updated)
    return updated


def write_sft_prompt_variants(
    input_jsonl: str | Path,
    output_dir: str | Path,
    variants: list[str] | None = None,
) -> dict[str, int]:
    rows = _read_sft(input_jsonl)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = variants or sorted(PROMPT_TEMPLATE_VARIANTS)
    counts: dict[str, int] = {}
    stem = Path(input_jsonl).stem
    for variant in selected:
        rewritten = [rewrite_sft_prompt_variant(row, variant) for row in rows]
        counts[variant] = write_jsonl(output / f"{stem}_{variant}.jsonl", rewritten)
    manifest = {
        "input_jsonl": str(input_jsonl),
        "variants": selected,
        "counts": counts,
        "contract": "messages_only",
    }
    (output / f"{stem}_prompt_variants_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SFT prompt-template variation JSONL files.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", default="data/sft/prompt_variants")
    parser.add_argument("--variants", nargs="*", choices=sorted(PROMPT_TEMPLATE_VARIANTS))
    args = parser.parse_args()
    counts = write_sft_prompt_variants(args.input_jsonl, args.output_dir, variants=args.variants)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
