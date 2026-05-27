from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable

from term_ai.augmentation.sft_builder import candidate_payload_to_sft_record
from term_ai.contracts import (
    APPROVED_AUG_STATUS,
    DEFAULT_TRAINABLE_AUG_STATUS,
    HUMAN_APPROVED_AUG_STATUS,
    JUDGE_VALIDATED_AUG_STATUS,
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


def _status_output_prefix(min_status: str) -> str:
    if min_status == JUDGE_VALIDATED_AUG_STATUS:
        return "judge_validated"
    if min_status == HUMAN_APPROVED_AUG_STATUS:
        return "human_approved"
    return min_status.replace("aug_", "").replace("_pass", "").replace("-", "_")


def _has_teacher_scores(row: dict[str, Any]) -> bool:
    payload = row.get("payload") or {}
    scores = row.get("teacher_scores") or payload.get("teacher_scores")
    return isinstance(scores, list) and len(scores) == 4 and all(isinstance(score, (int, float)) for score in scores)


def _teacher_scores(row: dict[str, Any]) -> list[float] | None:
    payload = row.get("payload") or {}
    scores = row.get("teacher_scores") or payload.get("teacher_scores")
    if isinstance(scores, list) and len(scores) == 4 and all(isinstance(score, (int, float)) for score in scores):
        total = float(sum(scores))
        if total > 0:
            return [float(score) / total for score in scores]
    return None


def _load_teacher_score_index(path: str | Path | None) -> dict[str, list[float]]:
    if path is None:
        return {}
    index: dict[str, list[float]] = {}
    for row in load_jsonl(path):
        scores = _teacher_scores(row)
        if scores is None:
            continue
        for key in ("item_id", "anchor_id", "word_id"):
            value = row.get(key)
            if value:
                index[str(value)] = scores
    return index


def _attach_teacher_scores(row: dict[str, Any], score_index: dict[str, list[float]]) -> tuple[dict[str, Any], bool]:
    current = _teacher_scores(row)
    updated = dict(row)
    payload = dict(updated.get("payload") or {})
    if current is None:
        for key in ("item_id", "anchor_id", "word_id"):
            value = updated.get(key)
            if value and str(value) in score_index:
                current = score_index[str(value)]
                break
    if current is None:
        return updated, False
    updated["teacher_scores"] = current
    payload["teacher_scores"] = current
    updated["payload"] = payload
    return updated, True


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
    rationale = f"원천 anchor에서 {anchor['word']}({anchor['pos']})의 한국어 뜻은 '{answer}'로 기록되어 있습니다."
    assistant = json.dumps(
        {"answer": chr(ord("A") + answer_idx), "confidence": 1.0, "rationale": rationale},
        ensure_ascii=False,
        sort_keys=True,
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


def build_validated_aug_sft_by_split(
    metadata_path: str | Path,
    output_dir: str | Path,
    min_status: str = DEFAULT_TRAINABLE_AUG_STATUS,
    output_prefix: str | None = None,
) -> dict[str, int]:
    rows = load_jsonl(metadata_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prefix = output_prefix or _status_output_prefix(min_status)

    for row in rows:
        if not status_reaches(row.get("status", ""), min_status):
            continue
        split = row.get("split")
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"metadata item missing split: {row.get('item_id')}")
        grouped[split].append(candidate_payload_to_sft_record(row["payload"]))

    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        counts[split] = write_jsonl(output / f"{prefix}_{split}_sft_v1.jsonl", grouped.get(split, []))
    policy = {
        "source_metadata": str(metadata_path),
        "min_status": min_status,
        "output_prefix": prefix,
        "human_spot_check_required_for_human_approved": min_status == HUMAN_APPROVED_AUG_STATUS,
        "notes": [
            "judge_validated means auto filter plus independent judge validation",
            "human_approved is reserved for aug_human_pass and must not be inferred from aug_judge_pass",
        ],
        "counts": counts,
    }
    (output / f"{prefix}_sft_policy_v1.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts


def build_approved_sft_by_split(
    metadata_path: str | Path,
    output_dir: str | Path,
    min_status: str = APPROVED_AUG_STATUS,
) -> dict[str, int]:
    if min_status != APPROVED_AUG_STATUS:
        raise ValueError(
            "approved-sft is reserved for human-approved aug_human_pass. "
            "Use validated-sft for aug_judge_pass data."
        )
    return build_validated_aug_sft_by_split(
        metadata_path,
        output_dir,
        min_status=min_status,
        output_prefix="approved",
    )


def build_raw_aug_sft_by_split(
    raw_sft_dir: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    min_status: str = DEFAULT_TRAINABLE_AUG_STATUS,
    output_prefix: str = "raw_judge_aug",
) -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grouped_aug: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(metadata_path):
        if not status_reaches(row.get("status", ""), min_status):
            continue
        split = row.get("split")
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"metadata item missing split: {row.get('item_id')}")
        grouped_aug[str(split)].append(candidate_payload_to_sft_record(row["payload"]))

    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        raw_path = Path(raw_sft_dir) / f"raw_{split}_sft_v1.jsonl"
        raw_records = load_jsonl(raw_path)
        merged = raw_records + grouped_aug.get(split, [])
        counts[split] = write_jsonl(output / f"{output_prefix}_{split}_sft_v1.jsonl", merged)
        counts[f"{split}_raw"] = len(raw_records)
        counts[f"{split}_aug"] = len(grouped_aug.get(split, []))
    policy = {
        "raw_sft_dir": str(raw_sft_dir),
        "aug_metadata": str(metadata_path),
        "min_status": min_status,
        "output_prefix": output_prefix,
        "counts": counts,
    }
    (output / f"{output_prefix}_sft_policy_v1.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts


def build_kd_metadata_view(
    raw_metadata_path: str | Path,
    generated_metadata_path: str | Path,
    output_path: str | Path,
    min_status: str = DEFAULT_TRAINABLE_AUG_STATUS,
    raw_teacher_scores_path: str | Path | None = None,
    include_raw: bool = True,
    require_raw_teacher_scores: bool = True,
) -> dict[str, int]:
    score_index = _load_teacher_score_index(raw_teacher_scores_path)
    output_rows: list[dict[str, Any]] = []
    missing_raw_scores: list[str] = []
    counts = {
        "raw_written": 0,
        "generated_written": 0,
        "generated_missing_teacher_scores": 0,
        "raw_missing_teacher_scores": 0,
    }

    if include_raw:
        for row in load_jsonl(raw_metadata_path):
            if row.get("status") != RAW_GT_STATUS:
                continue
            updated, has_scores = _attach_teacher_scores(row, score_index)
            if not has_scores:
                counts["raw_missing_teacher_scores"] += 1
                missing_raw_scores.append(str(row.get("item_id")))
                continue
            updated["kd_source"] = "raw_gt_teacher_scored"
            output_rows.append(updated)
            counts["raw_written"] += 1

    if require_raw_teacher_scores and missing_raw_scores:
        sample = ", ".join(missing_raw_scores[:5])
        raise ValueError(
            "raw metadata rows are missing teacher_scores; provide --raw-teacher-scores "
            f"or disable raw inclusion. sample item_ids: {sample}"
        )

    for row in load_jsonl(generated_metadata_path):
        if not status_reaches(row.get("status", ""), min_status):
            continue
        if not _has_teacher_scores(row):
            counts["generated_missing_teacher_scores"] += 1
            continue
        updated = dict(row)
        updated["kd_source"] = "judge_validated_generated"
        output_rows.append(updated)
        counts["generated_written"] += 1

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts["written"] = write_jsonl(output, output_rows)
    policy = {
        "raw_metadata": str(raw_metadata_path),
        "generated_metadata": str(generated_metadata_path),
        "raw_teacher_scores": str(raw_teacher_scores_path) if raw_teacher_scores_path else None,
        "min_status": min_status,
        "include_raw": include_raw,
        "require_raw_teacher_scores": require_raw_teacher_scores,
        "counts": counts,
        "notes": [
            "raw rows are included only when explicit teacher_scores are present or supplied",
            "generated rows are included only when they pass min_status and carry teacher_scores",
        ],
    }
    output.with_suffix(".policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts


def build_strict_eval_sets(
    raw_metadata_path: str | Path,
    validated_metadata_path: str | Path,
    output_dir: str | Path,
    min_status: str = DEFAULT_TRAINABLE_AUG_STATUS,
) -> dict[str, int]:
    """Write final evaluation views with raw GT and generated cloze separated."""

    raw_rows = load_jsonl(raw_metadata_path)
    validated_rows = load_jsonl(validated_metadata_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    test_raw = [
        row
        for row in raw_rows
        if row.get("status") == RAW_GT_STATUS and row.get("split") == "test" and row.get("source") == "raw_gt"
    ]
    test_cloze = [
        row
        for row in validated_rows
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
        "validated_aug_source": str(validated_metadata_path),
        "validated_min_status": min_status,
        "rules": [
            "test_raw contains only raw_gt rows from test split",
            "test_cloze_validated contains only validated Context Cloze rows from test split",
            "generated test cloze is never merged into raw test",
            "aug_judge_pass is strict judge validated, not human approved",
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
    parser = argparse.ArgumentParser(description="Build raw, validated, approved, and KD dataset views.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw = subparsers.add_parser("raw-sft")
    raw.add_argument("--anchors", default="data/processed/anchors_v1.jsonl")
    raw.add_argument("--output-dir", default="data/sft")
    raw.add_argument("--seed", type=int, default=42)

    raw_mcq = subparsers.add_parser("raw-mcq")
    raw_mcq.add_argument("--anchors", default="data/processed/anchors_v1.jsonl")
    raw_mcq.add_argument("--output-dir", default="data/metadata")
    raw_mcq.add_argument("--seed", type=int, default=42)

    validated = subparsers.add_parser("validated-sft")
    validated.add_argument("--metadata", required=True)
    validated.add_argument("--output-dir", default="data/sft")
    validated.add_argument("--min-status", default=DEFAULT_TRAINABLE_AUG_STATUS)
    validated.add_argument("--output-prefix")

    approved = subparsers.add_parser("approved-sft")
    approved.add_argument("--metadata", required=True)
    approved.add_argument("--output-dir", default="data/sft")
    approved.add_argument("--min-status", default=APPROVED_AUG_STATUS)

    raw_aug = subparsers.add_parser("raw-aug-sft")
    raw_aug.add_argument("--raw-sft-dir", default="data/sft")
    raw_aug.add_argument("--metadata", required=True)
    raw_aug.add_argument("--output-dir", default="data/sft")
    raw_aug.add_argument("--min-status", default=DEFAULT_TRAINABLE_AUG_STATUS)
    raw_aug.add_argument("--output-prefix", default="raw_judge_aug")

    kd_view = subparsers.add_parser("kd-metadata")
    kd_view.add_argument("--raw-metadata", required=True)
    kd_view.add_argument("--generated-metadata", required=True)
    kd_view.add_argument("--output", default="data/metadata/kd_train_view_v1.jsonl")
    kd_view.add_argument("--min-status", default=DEFAULT_TRAINABLE_AUG_STATUS)
    kd_view.add_argument("--raw-teacher-scores")
    kd_view.add_argument("--exclude-raw", action="store_true")
    kd_view.add_argument("--allow-missing-raw-teacher-scores", action="store_true")

    eval_sets = subparsers.add_parser("eval-sets")
    eval_sets.add_argument("--raw-metadata", required=True)
    eval_sets.add_argument("--validated-metadata")
    eval_sets.add_argument("--approved-metadata", dest="validated_metadata")
    eval_sets.add_argument("--output-dir", default="data/eval")
    eval_sets.add_argument("--min-status", default=DEFAULT_TRAINABLE_AUG_STATUS)

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
    elif args.command == "validated-sft":
        counts = build_validated_aug_sft_by_split(
            args.metadata,
            args.output_dir,
            min_status=args.min_status,
            output_prefix=args.output_prefix,
        )
    elif args.command == "approved-sft":
        counts = build_approved_sft_by_split(args.metadata, args.output_dir, min_status=args.min_status)
    elif args.command == "raw-aug-sft":
        counts = build_raw_aug_sft_by_split(
            args.raw_sft_dir,
            args.metadata,
            args.output_dir,
            min_status=args.min_status,
            output_prefix=args.output_prefix,
        )
    elif args.command == "kd-metadata":
        counts = build_kd_metadata_view(
            args.raw_metadata,
            args.generated_metadata,
            args.output,
            min_status=args.min_status,
            raw_teacher_scores_path=args.raw_teacher_scores,
            include_raw=not args.exclude_raw,
            require_raw_teacher_scores=not args.allow_missing_raw_teacher_scores,
        )
    elif args.command == "eval-sets":
        if not args.validated_metadata:
            parser.error("eval-sets requires --validated-metadata")
        counts = build_strict_eval_sets(
            args.raw_metadata,
            args.validated_metadata,
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
