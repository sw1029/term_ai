from term_ai.experiment.training import _format_chat, _trainer_tokenizer_kwargs


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
