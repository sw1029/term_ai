from __future__ import annotations

from pathlib import Path


def load_key_value_env(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_openai_api_key(env_path: str | Path = ".env") -> str | None:
    values = load_key_value_env(env_path)
    return values.get("OPENAI_API_KEY") or values.get("api-key") or values.get("OPENAI_APIKEY")
