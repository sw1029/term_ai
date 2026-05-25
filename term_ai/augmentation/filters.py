from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from term_ai.contracts import (
    AutoFilterResult,
    TASK_ANTONYM,
    TASK_CONTEXT_CLOZE,
    TASK_SYNONYM,
    TASK_TYPES,
    normalize_key,
    stable_id,
)


def _near_duplicate(left: str, right: str, threshold: float = 0.92) -> bool:
    return SequenceMatcher(None, normalize_key(left), normalize_key(right)).ratio() >= threshold


def _contains_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    return normalize_key(term) in normalize_key(text)


class AutoFilter:
    """Documented quality gate before any generated item can become SFT data."""

    def __init__(
        self,
        min_context_words: int = 20,
        max_context_words: int = 80,
        low_score_margin: float = 0.15,
        high_similarity_threshold: float = 0.92,
        blocking_warnings: set[str] | None = None,
    ) -> None:
        self.min_context_words = min_context_words
        self.max_context_words = max_context_words
        self.low_score_margin = low_score_margin
        self.high_similarity_threshold = high_similarity_threshold
        self.blocking_warnings = blocking_warnings or {
            "context_length_outlier",
            "low_teacher_score_margin",
            "near_duplicate_options",
        }

    def validate_payload(self, payload: dict[str, Any], item_id: str | None = None) -> AutoFilterResult:
        errors: list[str] = []
        warnings: list[str] = []
        item_id = item_id or stable_id("payload", payload)

        task_type = payload.get("task_type")
        if task_type not in TASK_TYPES:
            errors.append("missing_or_invalid_task_type")

        word = payload.get("word")
        if not isinstance(word, str) or not word.strip():
            errors.append("missing_word")

        options = payload.get("options")
        if not isinstance(options, list):
            errors.append("missing_options")
            options = []
        elif len(options) != 4:
            errors.append("options_count_not_4")
        elif any(not isinstance(option, str) or not option.strip() for option in options):
            errors.append("blank_option")

        answer_idx = payload.get("answer_idx")
        if not isinstance(answer_idx, int) or answer_idx < 0 or answer_idx >= len(options):
            errors.append("answer_idx_out_of_range")

        normalized_options = [normalize_key(option) for option in options]
        if len(set(normalized_options)) != len(normalized_options):
            errors.append("duplicate_options")
        else:
            for i, left in enumerate(options):
                for right in options[i + 1 :]:
                    if _near_duplicate(left, right):
                        warnings.append("near_duplicate_options")
                        break

        if task_type in {TASK_SYNONYM, TASK_ANTONYM} and isinstance(word, str):
            if normalize_key(word) in normalized_options:
                errors.append("target_word_in_options")

        context = payload.get("context")
        if task_type != TASK_CONTEXT_CLOZE and isinstance(context, str) and "___" in context:
            errors.append("blank_context_for_non_cloze")

        if task_type in {TASK_CONTEXT_CLOZE}:
            if not isinstance(context, str) or not context.strip():
                errors.append("missing_context")
            else:
                word_count = len(context.split())
                if word_count < self.min_context_words or word_count > self.max_context_words:
                    errors.append("context_length_outlier")

                if isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
                    answer = options[answer_idx]
                    if _contains_term(context, answer):
                        errors.append("answer_leakage_in_context")

                meaning_ko = payload.get("meaning_ko")
                if isinstance(meaning_ko, str) and _contains_term(context, meaning_ko):
                    errors.append("meaning_leakage_in_context")

        scores = payload.get("teacher_scores")
        if scores is not None:
            if not isinstance(scores, list) or len(scores) != len(options):
                errors.append("invalid_teacher_scores")
            elif all(isinstance(score, (int, float)) for score in scores):
                sorted_scores = sorted((float(score) for score in scores), reverse=True)
                if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < self.low_score_margin:
                    warnings.append("low_teacher_score_margin")
            else:
                errors.append("invalid_teacher_scores")

        similarity = payload.get("embedding_top2_similarity")
        if isinstance(similarity, (int, float)) and float(similarity) >= self.high_similarity_threshold:
            warnings.append("high_embedding_top2_similarity")

        blocking = set(warnings) & self.blocking_warnings
        status = "aug_auto_pass" if not errors and not blocking else "rejected"
        return AutoFilterResult(item_id=item_id, status=status, errors=errors, warnings=warnings)


def promote_metadata_with_auto_filter(
    metadata: dict[str, Any],
    result: AutoFilterResult,
) -> dict[str, Any]:
    updated = dict(metadata)
    updated["status"] = result.status
    validation_ids = list(updated.get("validation_ids") or [])
    validation_ids.append(stable_id("auto_filter", result.item_id, result.status))
    updated["validation_ids"] = validation_ids
    return updated
