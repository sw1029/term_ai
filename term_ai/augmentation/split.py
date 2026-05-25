from __future__ import annotations

import hashlib
from typing import Iterable

from term_ai.contracts import RawAnchor, SPLIT_RATIOS


def _hash_for_split(seed: int, word: str) -> str:
    payload = f"{seed}\x1f{word.casefold()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_word_splits(
    anchors: Iterable[RawAnchor],
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> tuple[list[RawAnchor], dict[str, str]]:
    ratios = ratios or SPLIT_RATIOS
    words = sorted({anchor.word for anchor in anchors}, key=lambda word: _hash_for_split(seed, word))
    total = len(words)
    train_cut = int(total * ratios["train"])
    dev_cut = train_cut + int(total * ratios["dev"])

    word_to_split: dict[str, str] = {}
    for idx, word in enumerate(words):
        if idx < train_cut:
            split = "train"
        elif idx < dev_cut:
            split = "dev"
        else:
            split = "test"
        word_to_split[word] = split

    assigned = [
        RawAnchor(
            anchor_id=anchor.anchor_id,
            word_id=anchor.word_id,
            word=anchor.word,
            pos=anchor.pos,
            meaning=anchor.meaning,
            source_line=anchor.source_line,
            definition_index=anchor.definition_index,
            duplicate_of=anchor.duplicate_of,
            split=word_to_split[anchor.word],
        )
        for anchor in anchors
    ]
    return assigned, word_to_split


def assert_no_word_leakage(word_to_split: dict[str, str]) -> None:
    invalid = [word for word, split in word_to_split.items() if split not in {"train", "dev", "test"}]
    if invalid:
        raise ValueError(f"invalid splits for words: {invalid[:5]}")
