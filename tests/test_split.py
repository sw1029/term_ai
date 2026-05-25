from term_ai.augmentation.split import assign_word_splits
from term_ai.contracts import RawAnchor


def test_word_level_split_keeps_same_word_together():
    anchors = [
        RawAnchor("a1", "w1", "contract", "noun", "계약", 1, 0),
        RawAnchor("a2", "w1", "contract", "verb", "계약하다", 1, 1),
        RawAnchor("a3", "w2", "invoice", "noun", "청구서", 2, 0),
        RawAnchor("a4", "w3", "audit", "noun", "감사", 3, 0),
    ]
    assigned, _ = assign_word_splits(anchors, seed=7)
    contract_splits = {anchor.split for anchor in assigned if anchor.word == "contract"}
    assert len(contract_splits) == 1
