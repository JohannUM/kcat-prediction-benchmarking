import logging


def emit_progress(logger: logging.Logger, stage: str, status: str, message: str, *args: object) -> None:
    """Emit a top-level wrapper progress event for live forwarding."""
    logger.info(
        message,
        *args,
        extra={
            "kcb_live": True,
            "kcb_event": f"progress.{stage}.{status}",
        },
    )


def progress_started(logger: logging.Logger, stage: str, message: str, *args: object) -> None:
    emit_progress(logger, stage, "started", message, *args)


def progress_completed(logger: logging.Logger, stage: str, message: str, *args: object) -> None:
    emit_progress(logger, stage, "completed", message, *args)
