from __future__ import annotations

import json
from typing import Protocol

from term_ai.env import resolve_openai_api_key


class TeacherClient(Protocol):
    def generate_json(self, prompt: str) -> dict:
        ...


class OpenAITeacherClient:
    def __init__(self, model: str, api_key: str | None = None, env_path: str = ".env") -> None:
        self.model = model
        self.api_key = api_key or resolve_openai_api_key(env_path)
        if not self.api_key:
            raise RuntimeError("OpenAI API key was not found in OPENAI_API_KEY or api-key")

    def generate_json(self, prompt: str) -> dict:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the llm extra to use OpenAI generation: pip install -e .[llm]") from exc

        client = OpenAI(api_key=self.api_key)

        # Prefer Responses API when present, but keep a Chat Completions fallback
        # because SDK versions vary across environments.
        try:
            response = client.responses.create(model=self.model, input=prompt)
            text = getattr(response, "output_text", None)
            if text:
                return json.loads(text)
        except Exception:
            pass

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        return json.loads(text or "{}")
