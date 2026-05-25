from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from term_ai.augmentation.anchors import extract_anchors
from term_ai.augmentation.filters import AutoFilter
from term_ai.augmentation.prompts import PROMPT_VERSION, build_generation_prompt
from term_ai.augmentation.sft_builder import candidate_payload_to_sft_record
from term_ai.augmentation.split import assign_word_splits, assert_no_word_leakage
from term_ai.augmentation.teacher import OpenAITeacherClient
from term_ai.contracts import (
    AugmentationMetadata,
    JudgeValidation,
    TASK_TYPES,
    dumps_jsonl,
    stable_id,
    status_reaches,
    validate_sft_record,
    write_jsonl,
)


def prepare_artifacts(input_path: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    anchors = extract_anchors(input_path)
    anchors, word_to_split = assign_word_splits(anchors, seed=seed)
    assert_no_word_leakage(word_to_split)

    processed_dir = output_dir / "processed"
    split_dir = output_dir / "splits"
    manifest_dir = output_dir / "manifests"
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    anchors_path = processed_dir / "anchors_v1.jsonl"
    split_path = split_dir / f"word_split_seed{seed}.json"
    manifest_path = manifest_dir / "version_manifest.json"

    write_jsonl(anchors_path, (anchor.to_dict() for anchor in anchors))
    split_path.write_text(json.dumps(word_to_split, ensure_ascii=False, indent=2), encoding="utf-8")

    duplicate_count = sum(1 for anchor in anchors if anchor.duplicate_of)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "anchor_path": str(anchors_path),
        "split_manifest": str(split_path),
        "seed": seed,
        "anchor_count": len(anchors),
        "word_count": len(word_to_split),
        "duplicate_anchor_count": duplicate_count,
        "data_version": "raw_v1",
        "split_version": f"word_split_seed{seed}",
        "schema_contract": "sft_messages_only",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def generate_candidates(
    anchors_path: Path,
    output_path: Path,
    task_type: str,
    model: str,
    env_path: Path,
    limit: int | None = None,
    requests_per_second: float = 1.0,
) -> int:
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task_type: {task_type}")
    if requests_per_second <= 0:
        raise ValueError("requests_per_second must be positive")

    teacher = OpenAITeacherClient(model=model, env_path=str(env_path))
    count = 0
    min_interval = 1.0 / requests_per_second
    last_request_at: float | None = None
    with anchors_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_handle:
        for line in input_handle:
            anchor = json.loads(line)
            if limit is not None and count >= limit:
                break
            if last_request_at is not None:
                elapsed = time.monotonic() - last_request_at
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            prompt = build_generation_prompt(
                task_type=task_type,
                word=anchor["word"],
                pos=anchor["pos"],
                meaning=anchor["meaning"],
            )
            last_request_at = time.monotonic()
            payload = teacher.generate_json(prompt)
            item_id = stable_id("aug", anchor["anchor_id"], task_type, payload)
            metadata = AugmentationMetadata(
                item_id=item_id,
                anchor_id=anchor["anchor_id"],
                word_id=anchor["word_id"],
                split=anchor["split"],
                status="aug_candidate",
                prompt_version=PROMPT_VERSION,
                generator_model=model,
                payload=payload,
                teacher_rationale=payload.get("rationale"),
                teacher_scores=payload.get("teacher_scores"),
            )
            output_handle.write(dumps_jsonl(metadata.to_dict()))
            count += 1
    return count


def auto_filter_metadata(metadata_path: Path, output_path: Path) -> dict[str, int]:
    filterer = AutoFilter()
    counts = {"aug_auto_pass": 0, "rejected": 0}
    with metadata_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_handle:
        for line in input_handle:
            metadata = json.loads(line)
            result = filterer.validate_payload(metadata.get("payload") or {}, item_id=metadata.get("item_id"))
            metadata["status"] = result.status
            metadata["auto_filter"] = result.to_dict()
            output_handle.write(dumps_jsonl(metadata))
            counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def _load_jsonl_by_item_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = row.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(f"validation row {line_no} missing item_id")
            rows[item_id] = row
    return rows


def apply_judge_validation(metadata_path: Path, judge_path: Path, output_path: Path) -> dict[str, int]:
    judge_by_item = _load_jsonl_by_item_id(judge_path)
    counts = {"aug_judge_pass": 0, "rejected": 0, "unchanged": 0}
    with metadata_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_handle:
        for line in input_handle:
            metadata = json.loads(line)
            item_id = metadata.get("item_id")
            judge_row = judge_by_item.get(item_id)
            if judge_row is None:
                counts["unchanged"] += 1
                output_handle.write(dumps_jsonl(metadata))
                continue

            validation = JudgeValidation(
                item_id=item_id,
                semantic_correctness=int(judge_row["semantic_correctness"]),
                distractor_validity=int(judge_row["distractor_validity"]),
                context_naturalness=int(judge_row["context_naturalness"]),
                leakage_check=str(judge_row["leakage_check"]),
                final_decision=str(judge_row["final_decision"]),
                judge_model=judge_row.get("judge_model"),
                notes=judge_row.get("notes"),
            )
            metadata["judge_validation"] = validation.to_dict()
            if validation.accepted() and status_reaches(metadata.get("status", ""), "aug_auto_pass"):
                metadata["status"] = "aug_judge_pass"
            else:
                metadata["status"] = "rejected"
            counts[metadata["status"]] += 1
            output_handle.write(dumps_jsonl(metadata))
    return counts


def apply_human_validation(metadata_path: Path, human_path: Path, output_path: Path) -> dict[str, int]:
    human_by_item = _load_jsonl_by_item_id(human_path)
    counts = {"aug_human_pass": 0, "rejected": 0, "unchanged": 0}
    with metadata_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_handle:
        for line in input_handle:
            metadata = json.loads(line)
            item_id = metadata.get("item_id")
            human_row = human_by_item.get(item_id)
            if human_row is None:
                counts["unchanged"] += 1
                output_handle.write(dumps_jsonl(metadata))
                continue

            decision = str(human_row.get("final_decision", "")).casefold()
            metadata["human_validation"] = human_row
            if decision == "accept" and status_reaches(metadata.get("status", ""), "aug_judge_pass"):
                metadata["status"] = "aug_human_pass"
            else:
                metadata["status"] = "rejected"
            counts[metadata["status"]] += 1
            output_handle.write(dumps_jsonl(metadata))
    return counts


def build_sft_from_metadata(metadata_path: Path, output_path: Path, min_status: str) -> int:
    count = 0
    with metadata_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_handle:
        for line in input_handle:
            metadata = json.loads(line)
            if not status_reaches(metadata.get("status", ""), min_status):
                continue
            record = candidate_payload_to_sft_record(metadata["payload"])
            validate_sft_record(record)
            output_handle.write(dumps_jsonl(record))
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Data augmentation artifacts and contract tooling.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input", default="pharaprased_voca.jsonl")
    prepare.add_argument("--output-dir", default="data")
    prepare.add_argument("--seed", type=int, default=42)

    generate = subparsers.add_parser("generate-candidates")
    generate.add_argument("--anchors", default="data/processed/anchors_v1.jsonl")
    generate.add_argument("--output", required=True)
    generate.add_argument("--task-type", required=True, choices=sorted(TASK_TYPES))
    generate.add_argument("--model", default="gpt-5.4-mini")
    generate.add_argument("--env", default=".env")
    generate.add_argument("--limit", type=int)
    generate.add_argument("--requests-per-second", type=float, default=1.0)

    auto_filter = subparsers.add_parser("auto-filter")
    auto_filter.add_argument("--metadata", required=True)
    auto_filter.add_argument("--output", required=True)

    judge = subparsers.add_parser("apply-judge")
    judge.add_argument("--metadata", required=True)
    judge.add_argument("--judge", required=True)
    judge.add_argument("--output", required=True)

    human = subparsers.add_parser("apply-human")
    human.add_argument("--metadata", required=True)
    human.add_argument("--human", required=True)
    human.add_argument("--output", required=True)

    build_sft = subparsers.add_parser("build-sft")
    build_sft.add_argument("--metadata", required=True)
    build_sft.add_argument("--output", required=True)
    build_sft.add_argument("--min-status", default="aug_human_pass")

    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare_artifacts(Path(args.input), Path(args.output_dir), seed=args.seed)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif args.command == "generate-candidates":
        count = generate_candidates(
            anchors_path=Path(args.anchors),
            output_path=Path(args.output),
            task_type=args.task_type,
            model=args.model,
            env_path=Path(args.env),
            limit=args.limit,
            requests_per_second=args.requests_per_second,
        )
        print(json.dumps({"written": count}, ensure_ascii=False))
    elif args.command == "auto-filter":
        counts = auto_filter_metadata(Path(args.metadata), Path(args.output))
        print(json.dumps(counts, ensure_ascii=False))
    elif args.command == "apply-judge":
        counts = apply_judge_validation(Path(args.metadata), Path(args.judge), Path(args.output))
        print(json.dumps(counts, ensure_ascii=False))
    elif args.command == "apply-human":
        counts = apply_human_validation(Path(args.metadata), Path(args.human), Path(args.output))
        print(json.dumps(counts, ensure_ascii=False))
    elif args.command == "build-sft":
        count = build_sft_from_metadata(Path(args.metadata), Path(args.output), min_status=args.min_status)
        print(json.dumps({"written": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
