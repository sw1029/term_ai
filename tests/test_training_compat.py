import json

from term_ai.experiment.lora_kd import _find_answer_token_position
from term_ai.experiment.training import (
    _format_chat,
    _normalize_assistant_json_answer,
    _tokenize_chat_completion,
    _trainer_tokenizer_kwargs,
)


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "System guidance."},
        {"role": "user", "content": "Question text."},
        {"role": "assistant", "content": "A) answer"},
    ]


class SystemRejectingTokenizer:
    chat_template = "raise_exception('System role not supported')"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert messages[0]["role"] == "user"
        assert "System guidance." in messages[0]["content"]
        assert "Question text." in messages[0]["content"]
        suffix = "|assistant:" if add_generation_prompt else ""
        return "|".join(f"{message['role']}:{message['content']}" for message in messages) + suffix


class SystemAcceptingTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert messages[0]["role"] == "system"
        return "|".join(f"{message['role']}:{message['content']}" for message in messages)


def test_format_chat_folds_system_message_for_templates_that_reject_system_role():
    text = _format_chat(SystemRejectingTokenizer(), {"messages": _messages()})
    assert text.startswith("user:System guidance.")
    assert "assistant:A) answer" in text


def test_format_chat_preserves_system_role_for_templates_that_accept_it():
    text = _format_chat(SystemAcceptingTokenizer(), {"messages": _messages()})
    assert text.startswith("system:System guidance.")


def test_format_chat_retries_when_template_reports_system_role_error():
    class RetryTokenizer(SystemAcceptingTokenizer):
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            if messages[0]["role"] == "system":
                raise RuntimeError("System role not supported")
            assert tokenize is False
            assert add_generation_prompt is True
            return "|".join(f"{message['role']}:{message['content']}" for message in messages)

    text = _format_chat(RetryTokenizer(), {"messages": _messages()}, add_generation_prompt=True)
    assert text.startswith("user:System guidance.")


def test_trainer_tokenizer_kwargs_supports_new_and_old_transformers_signatures():
    class NewTrainer:
        def __init__(self, processing_class=None) -> None:
            pass

    class OldTrainer:
        def __init__(self, tokenizer=None) -> None:
            pass

    tokenizer = object()
    assert _trainer_tokenizer_kwargs(NewTrainer, tokenizer) == {"processing_class": tokenizer}
    assert _trainer_tokenizer_kwargs(OldTrainer, tokenizer) == {"tokenizer": tokenizer}


def test_sft_assistant_target_can_be_normalized_to_json_answer():
    messages = _normalize_assistant_json_answer(_messages())
    assistant = json.loads(messages[-1]["content"])
    assert assistant["answer"] == "A"
    assert assistant["confidence"] == 1.0


def test_tokenize_chat_completion_masks_prompt_tokens():
    class SimpleTokenizer(SystemAcceptingTokenizer):
        def __call__(self, text: str, truncation: bool, max_length: int) -> dict[str, list[int]]:
            ids = [ord(char) for char in text][:max_length]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    tokenizer = SimpleTokenizer()
    encoded = _tokenize_chat_completion(tokenizer, _messages(), max_length=512, normalize_assistant_json=True)
    assistant_start = encoded["assistant_start"]

    assert assistant_start > 0
    assert all(label == -100 for label in encoded["labels"][:assistant_start])
    assert any(label != -100 for label in encoded["labels"][assistant_start:])


def test_kd_answer_position_finds_json_answer_value_token():
    class CharTokenizer(SystemAcceptingTokenizer):
        def __call__(self, text: str, truncation: bool, max_length: int) -> dict[str, list[int]]:
            ids = [ord(char) for char in text][:max_length]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    messages = [
        {"role": "system", "content": "System guidance."},
        {"role": "user", "content": "Question text."},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "answer": "C",
                    "confidence": 0.7,
                    "distribution": {"A": 0.1, "B": 0.1, "C": 0.7, "D": 0.1},
                },
                sort_keys=True,
            ),
        },
    ]
    tokenizer = CharTokenizer()
    encoded = _tokenize_chat_completion(tokenizer, messages, max_length=512)
    position = _find_answer_token_position(
        tokenizer,
        messages,
        encoded["input_ids"],
        encoded["assistant_start"],
        answer_idx=2,
        letter_token_ids=[ord("A"), ord("B"), ord("C"), ord("D")],
        max_length=512,
    )

    assert position is not None
    assert chr(encoded["input_ids"][position]) == "C"
