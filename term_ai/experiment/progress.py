from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
from types import FrameType
from typing import Any, Callable, Iterable

from term_ai.contracts import dumps_jsonl
from term_ai.experiment.metrics import summarize_predictions


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(output)


def write_jsonl_atomic(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(dumps_jsonl(record))
            count += 1
    tmp.replace(output)
    return count


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(dumps_jsonl(record))
        handle.flush()


def load_json(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    rows: list[dict[str, Any]] = []
    with candidate.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {candidate}:{line_no}") from exc
    return rows


def _step_from_checkpoint(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def resolve_latest_checkpoint(output_dir: str | Path) -> Path | None:
    checkpoints = Path(output_dir) / "checkpoints"
    if not checkpoints.exists():
        return None
    candidates = [path for path in checkpoints.glob("checkpoint-*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (_step_from_checkpoint(path), path.stat().st_mtime))


def resolve_latest_epoch_checkpoint(output_dir: str | Path, prefix: str) -> Path | None:
    checkpoints = Path(output_dir) / "checkpoints"
    if not checkpoints.exists():
        return None
    pattern = re.compile(rf"^{re.escape(prefix)}_epoch_(\d+)\.pt$")
    candidates: list[tuple[int, Path]] = []
    for path in checkpoints.glob(f"{prefix}_epoch_*.pt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1].stat().st_mtime))[1]


def backup_artifact(path: str | Path, output_dir: str | Path, name: str | None = None) -> Path:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    backup_root = Path(output_dir) / "backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    target_name = f"{name or source.name}_{utc_timestamp()}"
    if source.is_file() and source.suffix:
        target_name += source.suffix
    target = backup_root / target_name
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return target


@dataclass
class RunState:
    status: str
    stage: str
    completed_count: int = 0
    total_count: int | None = None
    latest_checkpoint: str | None = None
    final_artifact: str | None = None
    partial_metrics: dict[str, Any] | None = None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_iso)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["updated_at"] = utc_iso()
        return payload


class ProgressLogger:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        resume: bool = True,
        progress_interval_items: int = 1,
        stage: str = "running",
        total_count: int | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.progress_interval_items = max(1, int(progress_interval_items))
        self.stage = stage
        self.total_count = total_count
        self.prediction_log = self.output_dir / "prediction_log.jsonl"
        self.partial_prediction_log = self.output_dir / "prediction_log.partial.jsonl"
        self.metric_log = self.output_dir / "metric_log.json"
        self.partial_metric_log = self.output_dir / "metric_log.partial.json"
        self.metric_history = self.output_dir / "metric_history.jsonl"
        self.resume_state = self.output_dir / "resume_state.json"
        if not resume:
            for path in (self.partial_prediction_log, self.partial_metric_log):
                if path.exists():
                    path.unlink()
        self.predictions = self._load_seed_predictions() if resume else []
        self.completed_item_ids = {
            str(row["item_id"])
            for row in self.predictions
            if row.get("item_id") is not None
        }
        self.write_state("running", stage, completed_count=len(self.predictions), total_count=total_count)

    def _load_seed_predictions(self) -> list[dict[str, Any]]:
        rows = load_jsonl(self.partial_prediction_log)
        if not rows:
            rows = load_jsonl(self.prediction_log)
        return dedupe_predictions(rows)

    def has_prediction(self, item_id: object) -> bool:
        return str(item_id) in self.completed_item_ids

    def predictions_for_items(self, item_ids: Iterable[object]) -> list[dict[str, Any]]:
        wanted = {str(item_id) for item_id in item_ids}
        return [row for row in self.predictions if str(row.get("item_id")) in wanted]

    def completed_metrics_if_available(self, item_ids: Iterable[object]) -> dict[str, Any] | None:
        wanted = {str(item_id) for item_id in item_ids}
        if not wanted:
            return None
        final_rows = dedupe_predictions(load_jsonl(self.prediction_log))
        final_ids = {str(row.get("item_id")) for row in final_rows}
        if wanted <= final_ids and self.metric_log.exists():
            metrics = load_json(self.metric_log)
            if metrics is not None:
                return metrics
        if wanted <= self.completed_item_ids and self.predictions:
            rows = self.predictions_for_items(wanted)
            metrics = summarize_predictions(rows)
            self.finalize_predictions(metrics, rows)
            return metrics
        return None

    def append_prediction(self, row: dict[str, Any]) -> None:
        item_id = row.get("item_id")
        if item_id is not None and str(item_id) in self.completed_item_ids:
            return
        append_jsonl(self.partial_prediction_log, row)
        self.predictions.append(row)
        if item_id is not None:
            self.completed_item_ids.add(str(item_id))
        if len(self.predictions) % self.progress_interval_items == 0:
            self.flush_prediction_metrics()

    def flush_prediction_metrics(self, *, stage: str | None = None) -> dict[str, Any]:
        metrics = summarize_predictions(self.predictions)
        metrics["partial"] = True
        metrics["completed_count"] = len(self.predictions)
        if self.total_count is not None:
            metrics["total_count"] = self.total_count
        atomic_write_json(self.partial_metric_log, metrics)
        self.record_metrics(metrics, stage=stage or self.stage, event="progress", append_history=True)
        return metrics

    def record_metrics(
        self,
        metrics: dict[str, Any],
        *,
        stage: str | None = None,
        event: str = "metric",
        step: int | None = None,
        epoch: float | int | None = None,
        latest_checkpoint: str | Path | None = None,
        append_history: bool = True,
    ) -> None:
        payload = {
            "event": event,
            "stage": stage or self.stage,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "created_at": utc_iso(),
        }
        if latest_checkpoint is not None:
            payload["latest_checkpoint"] = str(latest_checkpoint)
        if append_history:
            append_jsonl(self.metric_history, payload)
        atomic_write_json(self.partial_metric_log, metrics)
        self.write_state(
            "running",
            stage or self.stage,
            completed_count=len(self.predictions),
            total_count=self.total_count,
            latest_checkpoint=str(latest_checkpoint) if latest_checkpoint is not None else None,
            partial_metrics=metrics,
        )

    def write_state(
        self,
        status: str,
        stage: str,
        *,
        completed_count: int = 0,
        total_count: int | None = None,
        latest_checkpoint: str | Path | None = None,
        final_artifact: str | Path | None = None,
        partial_metrics: dict[str, Any] | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        existing = load_json(self.resume_state) or {}
        state = RunState(
            status=status,
            stage=stage,
            completed_count=completed_count,
            total_count=total_count if total_count is not None else existing.get("total_count"),
            latest_checkpoint=str(latest_checkpoint) if latest_checkpoint is not None else existing.get("latest_checkpoint"),
            final_artifact=str(final_artifact) if final_artifact is not None else existing.get("final_artifact"),
            partial_metrics=partial_metrics if partial_metrics is not None else existing.get("partial_metrics"),
            reason=reason,
            details=details or existing.get("details") or {},
        )
        payload = dict(existing)
        payload.update(state.to_dict())
        atomic_write_json(self.resume_state, payload)

    def mark_interrupted(
        self,
        reason: str,
        *,
        stage: str | None = None,
        latest_checkpoint: str | Path | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        metrics = summarize_predictions(self.predictions) if self.predictions else None
        if metrics is not None:
            atomic_write_json(self.partial_metric_log, metrics)
        self.write_state(
            "interrupted",
            stage or self.stage,
            completed_count=len(self.predictions),
            total_count=self.total_count,
            latest_checkpoint=latest_checkpoint,
            partial_metrics=metrics,
            reason=reason,
            details=details,
        )

    def mark_failed(
        self,
        reason: str,
        *,
        stage: str | None = None,
        latest_checkpoint: str | Path | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.write_state(
            "failed",
            stage or self.stage,
            completed_count=len(self.predictions),
            total_count=self.total_count,
            latest_checkpoint=latest_checkpoint,
            reason=reason,
            details=details,
        )

    def finalize_predictions(
        self,
        metrics: dict[str, Any],
        rows: list[dict[str, Any]] | None = None,
        *,
        final_artifact: str | Path | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        final_rows = dedupe_predictions(rows if rows is not None else self.predictions)
        write_jsonl_atomic(self.prediction_log, final_rows)
        atomic_write_json(self.metric_log, metrics)
        self.record_metrics(metrics, stage="completed", event="completed", append_history=True)
        atomic_write_json(self.partial_metric_log, {**metrics, "partial": False})
        self.write_state(
            "completed",
            "completed",
            completed_count=len(final_rows),
            total_count=self.total_count,
            final_artifact=final_artifact,
            partial_metrics=metrics,
            details=details,
        )


class InterruptGuard:
    def __init__(
        self,
        progress: ProgressLogger | None,
        *,
        stage: str = "running",
        checkpoint_callback: Callable[[], str | Path | dict[str, Any] | None] | None = None,
    ) -> None:
        self.progress = progress
        self.stage = stage
        self.checkpoint_callback = checkpoint_callback
        self._previous: dict[int, Any] = {}
        self._handled = False

    def __enter__(self) -> "InterruptGuard":
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
            except (ValueError, RuntimeError):
                continue
        return self

    def _handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        name = signal.Signals(signum).name
        self._mark_interrupted(f"received {name}")
        raise KeyboardInterrupt(name)

    def _run_checkpoint_callback(self) -> tuple[str | None, dict[str, Any]]:
        details: dict[str, Any] = {}
        latest_checkpoint: str | None = None
        if self.checkpoint_callback is None:
            return latest_checkpoint, details
        result = self.checkpoint_callback()
        if isinstance(result, dict):
            details.update(result)
            if result.get("latest_checkpoint"):
                latest_checkpoint = str(result["latest_checkpoint"])
        elif result is not None:
            latest_checkpoint = str(result)
        return latest_checkpoint, details

    def _mark_interrupted(self, reason: str) -> None:
        if self._handled:
            return
        self._handled = True
        latest_checkpoint, details = self._run_checkpoint_callback()
        if self.progress is not None:
            self.progress.mark_interrupted(
                reason,
                stage=self.stage,
                latest_checkpoint=latest_checkpoint,
                details=details,
            )

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, _tb: Any) -> bool:
        for signum, previous in self._previous.items():
            try:
                signal.signal(signum, previous)
            except (ValueError, RuntimeError):
                continue
        if exc_type is None:
            return False
        if issubclass(exc_type, KeyboardInterrupt):
            self._mark_interrupted(str(exc) if exc else "keyboard interrupt")
            return False
        latest_checkpoint, details = self._run_checkpoint_callback()
        if self.progress is not None:
            self.progress.mark_failed(
                f"{exc_type.__name__}: {exc}",
                stage=self.stage,
                latest_checkpoint=latest_checkpoint,
                details=details,
            )
        return False


def dedupe_predictions(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for row in rows:
        item_id = row.get("item_id")
        if item_id is None:
            anonymous.append(row)
        else:
            deduped[str(item_id)] = row
    return anonymous + list(deduped.values())
