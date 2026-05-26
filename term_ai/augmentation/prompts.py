from __future__ import annotations

from term_ai.contracts import TASK_TYPES


PROMPT_VERSION = "aug_prompt_v1"


def build_generation_prompt(
    task_type: str,
    word: str,
    pos: str,
    meaning: str,
) -> str:
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task_type: {task_type}")

    return f"""You generate candidate data for a TOEIC business vocabulary dataset.

Use only the given anchor as the source:
- word: {word}
- part_of_speech: {pos}
- Korean meaning: {meaning}
- task_type: {task_type}

Return one valid JSON object only. Do not include markdown.

Required JSON fields:
- task_type: exactly "{task_type}"
- word: exactly "{word}"
- meaning_ko: Korean meaning or a Korean paraphrase based on the anchor
- context: business English sentence if the task needs context; use a blank "___" for Context Cloze
- options: exactly four answer options
- answer_idx: integer from 0 to 3
- rationale: short Korean reason, no hidden chain-of-thought
- teacher_scores: four numeric scores in option order, summing approximately to 1

Dataset rules:
- This is only an augmentation candidate, not ground truth.
- Do not put metadata fields inside the final SFT record.
- Context Cloze context must be a natural business English sentence of 20 to 35 words.
- Context Cloze must use exactly one "___" blank and the correct option must fit grammatically when inserted.
- Context Cloze must not expose the correct option in the context.
- Only Context Cloze may use "___" in the context.
- Synonym Selection and Antonym Selection options must not include the target word itself.
- Options must not be duplicates or near duplicates.
- The correct option must be unambiguous in TOEIC business context.
"""
