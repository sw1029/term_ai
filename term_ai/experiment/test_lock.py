from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


def default_test_lock_dir() -> Path:
    """Use a repository-scoped lock so changing output_dir cannot re-open test."""

    configured = os.environ.get("TERM_AI_TEST_LOCK_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "runs" / "_test_locks"


def enforce_final_test_once(
    output_dir: str | Path,
    experiment_id: str,
    eval_split: str,
    enabled: bool = True,
    lock_dir: str | Path | None = None,
) -> Path | None:
    if not enabled or eval_split != "test":
        return None

    output = Path(output_dir)
    locks = Path(lock_dir) if lock_dir is not None else default_test_lock_dir()
    locks.mkdir(parents=True, exist_ok=True)
    lock_path = locks / f"{experiment_id}_test.lock.json"
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "eval_split": eval_split,
        "output_dir": str(output),
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "policy": "final test can be evaluated only once unless the lock is explicitly removed",
    }
    try:
        with lock_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except FileExistsError as exc:
        existing = lock_path.read_text(encoding="utf-8")
        raise RuntimeError(f"final test is already locked for {experiment_id}: {lock_path}\n{existing}") from exc
    return lock_path
