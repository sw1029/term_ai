from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class KDRecord:
    item_id: str
    hard_label: int
    teacher_scores: list[float]
    split: str
    task_type: str


def load_kd_records(metadata_path: str | Path, min_status: str = "aug_human_pass") -> list[KDRecord]:
    from term_ai.contracts import status_reaches

    records: list[KDRecord] = []
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not status_reaches(row.get("status", ""), min_status):
                continue
            payload: dict[str, Any] = row.get("payload") or {}
            scores = row.get("teacher_scores") or payload.get("teacher_scores")
            answer_idx = payload.get("answer_idx")
            if not isinstance(answer_idx, int):
                raise ValueError(f"metadata line {line_no} missing integer answer_idx")
            if not isinstance(scores, list) or len(scores) != 4:
                raise ValueError(f"metadata line {line_no} missing 4 teacher_scores")
            records.append(
                KDRecord(
                    item_id=row["item_id"],
                    hard_label=answer_idx,
                    teacher_scores=[float(score) for score in scores],
                    split=row["split"],
                    task_type=payload.get("task_type", "unknown"),
                )
            )
    return records


def write_kd_training_view(metadata_path: str | Path, output_path: str | Path, min_status: str = "aug_human_pass") -> int:
    records = load_kd_records(metadata_path, min_status=min_status)
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.__dict__, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(records)
