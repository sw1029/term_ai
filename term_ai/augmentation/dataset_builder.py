from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable

from term_ai.augmentation.sft_builder import candidate_payload_to_sft_record
from term_ai.contracts import (
    RAW_GT_STATUS,
    SYSTEM_PROMPT,
    TASK_CONTEXT_CLOZE,
    TASK_RAW_MEANING_SELECTION,
    TASK_SENSE_DISAMBIGUATION,
    dumps_jsonl,
    make_sft_record,
    stable_id,
    status_reaches,
    write_jsonl,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_records(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        split = record.get("split")
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"record missing valid split: {record}")
        grouped[split].append(record)
    return dict(grouped)


def _meaning_options(
    anchor: dict[str, Any],
    split_pool: list[dict[str, Any]],
    rng: random.Random,
    option_count: int = 4,
) -> tuple[list[str], int]:
    correct = str(anchor["meaning"]).strip()
    candidates = [
        str(row["meaning"]).strip()
        for row in split_pool
        if row.get("anchor_id") != anchor.get("anchor_id") and str(row.get("meaning", "")).strip() != correct
    ]
    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) < option_count - 1:
        raise ValueError(f"not enough distractor meanings in split={anchor.get('split')}")
    distractors = rng.sample(unique_candidates, option_count - 1)
    options = distractors + [correct]
    rng.shuffle(options)
    return options, options.index(correct)


def raw_anchor_to_sft(anchor: dict[str, Any], options: list[str], answer_idx: int) -> dict[str, Any]:
    user = (
        "Task: Sense Disambiguation\n\n"
        "다음 영어 단어와 품사에 해당하는 한국어 뜻으로 가장 가까운 것을 고르시오.\n\n"
        f"Word: {anchor['word']}\n"
        f"Part of speech: {anchor['pos']}\n\n"
        + "\n".join(f"{chr(ord('A') + idx)}) {option}" for idx, option in enumerate(options))
    )
    answer = options[answer_idx]
    assistant = (
        f"{chr(ord('A') + answer_idx)}) {answer}\n\n"
        f"원천 anchor에서 {anchor['word']}({anchor['pos']})의 한국어 뜻은 '{answer}'로 기록되어 있습니다."
    )
    return make_sft_record(SYSTEM_PROMPT, user, assistant)


def raw_anchor_to_mcq_metadata(anchor: dict[str, Any], options: list[str], answer_idx: int) -> dict[str, Any]:
    payload = {
        "task_type": TASK_SENSE_DISAMBIGUATION,
        "source_task_type": TASK_RAW_MEANING_SELECTION,
        "word": str(anchor["word"]),
        "meaning_ko": str(anchor["meaning"]),
        "context": f"Word: {anchor['word']}\nPart of speech: {anchor['pos']}",
        "options": options,
        "answer_idx": answer_idx,
        "rationale": (
            f"원천 anchor에서 {anchor['word']}({anchor['pos']})의 한국어 뜻은 "
            f"'{options[answer_idx]}'로 기록되어 있습니다."
        ),
    }
    split = str(anchor["split"])
    dataset_view = "test_raw" if split == "test" else f"raw_{split}"
    return {
        "item_id": stable_id("raw_mcq", anchor["anchor_id"], options, answer_idx),
        "anchor_id": anchor["anchor_id"],
        "word_id": anchor["word_id"],
        "split": split,
        "status": RAW_GT_STATUS,
        "source": "raw_gt",
        "dataset_view": dataset_view,
        "payload": payload,
    }


def build_raw_sft_from_anchors(
    anchors_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    skip_duplicates: bool = True,
) -> dict[str, int]:
    anchors = load_jsonl(anchors_path)
    if skip_duplicates:
        anchors = [anchor for anchor in anchors if not anchor.get("duplicate_of")]
    grouped = split_records(anchors)
    rng = random.Random(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for split, rows in grouped.items():
        records: list[dict[str, Any]] = []
        for anchor in rows:
            try:
                options, answer_idx = _meaning_options(anchor, rows, rng)
            except ValueError:
                continue
            records.append(raw_anchor_to_sft(anchor, options, answer_idx))
        counts[split] = write_jsonl(output / f"raw_{split}_sft_v1.jsonl", records)
    return counts


def build_raw_mcq_metadata_from_anchors(
    anchors_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    skip_duplicates: bool = True,
) -> dict[str, int]:
    anchors = load_jsonl(anchors_path)
    if skip_duplicates:
        anchors = [anchor for anchor in anchors if not anchor.get("duplicate_of")]
    grouped = split_records(anchors)
    rng = random.Random(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    all_records: list[dict[str, Any]] = []
    for split, rows in grouped.items():
        records: list[dict[str, Any]] = []
        for anchor in rows:
            try:
                options, answer_idx = _meaning_options(anchor, rows, rng)
            except ValueError:
                continue
            records.append(raw_anchor_to_mcq_metadata(anchor, options, answer_idx))
        counts[split] = write_jsonl(output / f"raw_{split}_mcq_v1.jsonl", records)
        all_records.extend(records)
    counts["all"] = write_jsonl(output / "raw_mcq_v1.jsonl", all_records)
    return counts


def build_approved_sft_by_split(
    metadata_path: str | Path,
    output_dir: str | Path,
    min_status: str = "aug_human_pass",
) -> dict[str, int]:
    rows = load_jsonl(metadata_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if not status_reaches(row.get("status", ""), min_status):
            continue
        split = row.get("split")
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"metadata item missing split: {row.get('item_id')}")
        grouped[split].append(candidate_payload_to_sft_record(row["payload"]))

    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        counts[split] = write_jsonl(output / f"approved_{split}_sft_v1.jsonl", grouped.get(split, []))
    return counts


def build_strict_eval_sets(
    raw_metadata_path: str | Path,
    approved_metadata_path: str | Path,
    output_dir: str | Path,
    min_status: str = "aug_human_pass",
) -> dict[str, int]:
    """Write final evaluation views with raw GT and generated cloze separated."""

    raw_rows = load_jsonl(raw_metadata_path)
    approved_rows = load_jsonl(approved_metadata_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    test_raw = [
        row
        for row in raw_rows
        if row.get("status") == RAW_GT_STATUS and row.get("split") == "test" and row.get("source") == "raw_gt"
    ]
    test_cloze = [
        row
        for row in approved_rows
        if row.get("split") == "test"
        and status_reaches(row.get("status", ""), min_status)
        and (row.get("payload") or {}).get("task_type") == TASK_CONTEXT_CLOZE
    ]
    dev_raw = [
        row
        for row in raw_rows
        if row.get("status") == RAW_GT_STATUS and row.get("split") == "dev" and row.get("source") == "raw_gt"
    ]
    train_raw = [
        row
        for row in raw_rows
        if row.get("status") == RAW_GT_STATUS and row.get("split") == "train" and row.get("source") == "raw_gt"
    ]

    counts = {
        "train_raw": write_jsonl(output / "train_raw_v1.jsonl", train_raw),
        "dev_raw": write_jsonl(output / "dev_raw_v1.jsonl", dev_raw),
        "test_raw": write_jsonl(output / "test_raw_v1.jsonl", test_raw),
        "test_cloze_validated": write_jsonl(output / "test_cloze_validated_v1.jsonl", test_cloze),
    }
    policy = {
        "raw_gt_source": str(raw_metadata_path),
        "approved_aug_source": str(approved_metadata_path),
        "approved_min_status": min_status,
        "rules": [
            "test_raw contains only raw_gt rows from test split",
            "test_cloze_validated contains only approved Context Cloze rows from test split",
            "generated test cloze is never merged into raw test",
        ],
        "counts": counts,
    }
    (output / "eval_set_policy_v1.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts


def add_stress_tags(
    metadata_path: str | Path,
    anchors_path: str | Path,
    output_path: str | Path,
    high_similarity_threshold: float = 0.92,
) -> dict[str, int]:
    anchors = load_jsonl(anchors_path)
    meanings_by_word: dict[str, set[str]] = defaultdict(set)
    for anchor in anchors:
        meanings_by_word[str(anchor["word"])].add(str(anchor["meaning"]))

    counts = {"written": 0, "short_answer": 0, "polysemy": 0, "high_similarity": 0}
    with open(metadata_path, "r", encoding="utf-8") as input_handle, open(
        output_path, "w", encoding="utf-8", newline="\n"
    ) as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            payload = row.get("payload") or {}
            meaning = str(payload.get("meaning_ko") or "").strip()
            word = str(payload.get("word") or row.get("word") or "")
            tags: list[str] = []
            if meaning and len(meaning.replace(" ", "")) <= 3:
                tags.append("short_answer")
            if len(meanings_by_word.get(word, set())) >= 2:
                tags.append("polysemy")
            similarity = payload.get("embedding_top2_similarity")
            if isinstance(similarity, (int, float)) and float(similarity) >= high_similarity_threshold:
                tags.append("high_similarity")
            row["stress_tags"] = sorted(set(tags))
            for tag in tags:
                counts[tag] += 1
            counts["written"] += 1
            output_handle.write(dumps_jsonl(row))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw and approved SFT datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw = subparsers.add_parser("raw-sft")
    raw.add_argument("--anchors", default="data/processed/anchors_v1.jsonl")
    raw.add_argument("--output-dir", default="data/sft")
    raw.add_argument("--seed", type=int, default=42)

    raw_mcq = subparsers.add_parser("raw-mcq")
    raw_mcq.add_argument("--anchors", default="data/processed/anchors_v1.jsonl")
    raw_mcq.add_argument("--output-dir", default="data/metadata")
    raw_mcq.add_argument("--seed", type=int, default=42)

    approved = subparsers.add_parser("approved-sft")
    approved.add_argument("--metadata", required=True)
    approved.add_argument("--output-dir", default="data/sft")
    approved.add_argument("--min-status", default="aug_human_pass")

    eval_sets = subparsers.add_parser("eval-sets")
    eval_sets.add_argument("--raw-metadata", required=True)
    eval_sets.add_argument("--approved-metadata", required=True)
    eval_sets.add_argument("--output-dir", default="data/eval")
    eval_sets.add_argument("--min-status", default="aug_human_pass")

    tags = subparsers.add_parser("stress-tags")
    tags.add_argument("--metadata", required=True)
    tags.add_argument("--anchors", default="data/processed/anchors_v1.jsonl")
    tags.add_argument("--output", required=True)
    tags.add_argument("--high-similarity-threshold", type=float, default=0.92)

    args = parser.parse_args()
    if args.command == "raw-sft":
        counts = build_raw_sft_from_anchors(args.anchors, args.output_dir, seed=args.seed)
    elif args.command == "raw-mcq":
        counts = build_raw_mcq_metadata_from_anchors(args.anchors, args.output_dir, seed=args.seed)
    elif args.command == "approved-sft":
        counts = build_approved_sft_by_split(args.metadata, args.output_dir, min_status=args.min_status)
    elif args.command == "eval-sets":
        counts = build_strict_eval_sets(
            args.raw_metadata,
            args.approved_metadata,
            args.output_dir,
            min_status=args.min_status,
        )
    else:
        counts = add_stress_tags(
            args.metadata,
            args.anchors,
            args.output,
            high_similarity_threshold=args.high_similarity_threshold,
        )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
