from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from term_ai.contracts import RawAnchor, iter_jsonl, normalize_key, normalize_space, stable_id


def extract_anchors(input_path: str | Path) -> list[RawAnchor]:
    anchors: list[RawAnchor] = []
    seen: dict[tuple[str, str, str], str] = {}

    for line_no, record in enumerate(iter_jsonl(input_path), start=1):
        word = normalize_space(str(record.get("word", "")))
        definitions = record.get("definitions")
        if not word or not isinstance(definitions, list):
            raise ValueError(f"line {line_no} must contain word and definitions[]")

        word_id = stable_id("word", normalize_key(word), length=12)
        for definition_index, definition in enumerate(definitions):
            if not isinstance(definition, dict):
                raise ValueError(f"line {line_no} definition {definition_index} must be an object")
            pos = normalize_space(str(definition.get("pos", "")))
            meaning = normalize_space(str(definition.get("meaning", "")))
            if not pos or not meaning:
                raise ValueError(f"line {line_no} definition {definition_index} must contain pos and meaning")

            anchor_id = stable_id("anchor", normalize_key(word), normalize_key(pos), normalize_key(meaning), length=16)
            key = (normalize_key(word), normalize_key(pos), normalize_key(meaning))
            duplicate_of = seen.get(key)
            if duplicate_of is None:
                seen[key] = anchor_id

            anchors.append(
                RawAnchor(
                    anchor_id=anchor_id,
                    word_id=word_id,
                    word=word,
                    pos=pos,
                    meaning=meaning,
                    source_line=line_no,
                    definition_index=definition_index,
                    duplicate_of=duplicate_of,
                )
            )

    return anchors


def group_anchors_by_word(anchors: list[RawAnchor]) -> dict[str, list[RawAnchor]]:
    grouped: dict[str, list[RawAnchor]] = defaultdict(list)
    for anchor in anchors:
        grouped[anchor.word].append(anchor)
    return dict(grouped)


def anchors_to_dicts(anchors: list[RawAnchor]) -> list[dict[str, Any]]:
    return [anchor.to_dict() for anchor in anchors]
