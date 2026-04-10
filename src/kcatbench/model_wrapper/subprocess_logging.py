import json
import logging
from dataclasses import dataclass
from typing import Any, Optional


WORKER_LOG_PREFIX = "KCB_WORKER_LOG "


@dataclass(frozen=True)
class WorkerLogEvent:
    levelno: int
    levelname: str
    logger_name: str
    message: str
    live: bool
    event: Optional[str] = None


def encode_worker_log_record(record: logging.LogRecord) -> str:
    """Serialize a log record into a parseable worker transport line."""
    payload = {
        "levelno": int(record.levelno),
        "levelname": str(record.levelname),
        "logger_name": str(record.name),
        "message": str(record.getMessage()),
        "live": bool(getattr(record, "kcb_live", False)),
    }

    event = getattr(record, "kcb_event", None)
    if event:
        payload["event"] = str(event)

    return WORKER_LOG_PREFIX + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def parse_worker_log_line(line: str) -> Optional[WorkerLogEvent]:
    if not line.startswith(WORKER_LOG_PREFIX):
        return None

    raw_payload = line[len(WORKER_LOG_PREFIX):]
    try:
        payload: dict[str, Any] = json.loads(raw_payload)
    except json.JSONDecodeError:
        return None

    levelno = payload.get("levelno", logging.INFO)
    if not isinstance(levelno, int):
        try:
            levelno = int(levelno)
        except (TypeError, ValueError):
            levelno = logging.INFO

    levelname = payload.get("levelname", logging.getLevelName(levelno))
    logger_name = payload.get("logger_name", "worker")
    message = payload.get("message", "")
    live = bool(payload.get("live", False))
    event = payload.get("event")
    if event is not None and not isinstance(event, str):
        event = str(event)

    return WorkerLogEvent(
        levelno=levelno,
        levelname=str(levelname),
        logger_name=str(logger_name),
        message=str(message),
        live=live,
        event=event,
    )


def render_worker_log_event(event: WorkerLogEvent) -> str:
    return f"{event.levelname} | {event.logger_name} | {event.message}"
