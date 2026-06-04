from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
from typing import Any

from term_ai.experiment.hf_loading import from_pretrained_with_trust


DEFAULT_G5_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]
MODEL_WEIGHT_FILENAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}


@dataclass(frozen=True)
class ModelDownloadResult:
    model_id: str
    status: str
    cache_dir: str | None
    cached_path: str | None
    verified: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(str(os.environ["HF_HUB_CACHE"])).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(str(os.environ["HF_HOME"])).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_cache_path(model_id: str, cache_dir: str | Path | None = None) -> Path:
    cache = Path(cache_dir).expanduser() if cache_dir else default_hf_cache_dir()
    return cache / f"models--{model_id.replace('/', '--')}"


def model_appears_cached(model_id: str, cache_dir: str | Path | None = None) -> bool:
    root = model_cache_path(model_id, cache_dir)
    snapshots = root / "snapshots"
    if not root.exists() or not snapshots.exists():
        return False
    for snapshot in snapshots.iterdir():
        if not snapshot.is_dir():
            continue
        if (snapshot / "config.json").exists() and any((snapshot / name).exists() for name in MODEL_WEIGHT_FILENAMES):
            return True
    return False


def _verify_local_load(model_id: str, cache_dir: str | Path | None, trust_remote_code: bool) -> None:
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install train dependencies first: pip install -e .[train]") from exc
    kwargs: dict[str, Any] = {"local_files_only": True}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    from_pretrained_with_trust(AutoConfig, model_id, trust_remote_code, **kwargs)
    from_pretrained_with_trust(AutoTokenizer, model_id, trust_remote_code, **kwargs)


def ensure_model_cached(
    model_id: str,
    *,
    cache_dir: str | Path | None = None,
    download_if_missing: bool = False,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> ModelDownloadResult:
    cache = Path(cache_dir).expanduser() if cache_dir else default_hf_cache_dir()
    cached = model_appears_cached(model_id, cache)
    if not cached and not download_if_missing:
        return ModelDownloadResult(
            model_id=model_id,
            status="missing",
            cache_dir=str(cache),
            cached_path=None,
            verified=False,
        )
    if not cached and local_files_only:
        return ModelDownloadResult(
            model_id=model_id,
            status="missing_local_only",
            cache_dir=str(cache),
            cached_path=None,
            verified=False,
        )
    downloaded = False
    if not cached:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub before model download") from exc
        snapshot_download(repo_id=model_id, cache_dir=str(cache), local_files_only=False)
        downloaded = True
        cached = model_appears_cached(model_id, cache)

    try:
        _verify_local_load(model_id, cache, trust_remote_code)
    except Exception as exc:
        return ModelDownloadResult(
            model_id=model_id,
            status="downloaded_unverified" if downloaded else "cached_unverified",
            cache_dir=str(cache),
            cached_path=str(model_cache_path(model_id, cache)) if cached else None,
            verified=False,
            error=str(exc),
        )

    return ModelDownloadResult(
        model_id=model_id,
        status="downloaded" if downloaded else "cached",
        cache_dir=str(cache),
        cached_path=str(model_cache_path(model_id, cache)),
        verified=True,
    )


def ensure_models_cached(
    model_ids: list[str],
    *,
    cache_dir: str | Path | None = None,
    download_if_missing: bool = False,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> list[ModelDownloadResult]:
    return [
        ensure_model_cached(
            model_id,
            cache_dir=cache_dir,
            download_if_missing=download_if_missing,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        for model_id in model_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache-first Hugging Face model download and local-load verification.")
    parser.add_argument("--model-id", action="append", dest="model_ids", help="HF model id. Repeatable.")
    parser.add_argument("--cache-dir")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", help="Optional JSON file for download/cache verification results.")
    args = parser.parse_args()
    model_ids = args.model_ids or DEFAULT_G5_MODELS
    results = ensure_models_cached(
        model_ids,
        cache_dir=args.cache_dir,
        download_if_missing=args.download_if_missing,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    payload = [result.to_dict() for result in results]
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
