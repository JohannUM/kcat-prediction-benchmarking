import argparse
import logging
import sys
import traceback
import pandas as pd
from pathlib import Path
from collections.abc import Callable
from typing import Optional

from kcatbench.util import read_csv_with_schema
from kcatbench.model_wrapper.subprocess_logging import encode_worker_log_record
from kcatbench.model_wrapper.wrapper_progress import progress_completed, progress_started


WORKER_LOGGER = logging.getLogger(__name__)
_LOGGING_CONFIGURED = False


class _WorkerTransportHandler(logging.Handler):
    """Emits parseable worker log lines to stderr for parent-process forwarding."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = encode_worker_log_record(record)
            print(line, file=sys.stderr, flush=True)
        except Exception:
            self.handleError(record)


def _configure_worker_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    kcatbench_logger = logging.getLogger("kcatbench")
    kcatbench_logger.setLevel(logging.DEBUG)
    kcatbench_logger.propagate = False
    kcatbench_logger.handlers.clear()
    kcatbench_logger.addHandler(_WorkerTransportHandler())

    _LOGGING_CONFIGURED = True


def _predict_dlkcat(input:pd.DataFrame) -> pd.DataFrame:
    from kcatbench.model_wrapper.inner_wrapper.dlkcat_wrapper import DLKcatWrapper
    model = DLKcatWrapper()
    return model.predict(input)

def _predict_catapro(input:pd.DataFrame) -> pd.DataFrame:
    from kcatbench.model_wrapper.inner_wrapper.catapro_wrapper import CataProWrapper
    model = CataProWrapper()
    return model.predict(input)

def _predict_catpred(input:pd.DataFrame) -> pd.DataFrame:
    from kcatbench.model_wrapper.inner_wrapper.catpred_wrapper import CatPredWrapper
    model = CatPredWrapper()
    return model.predict(input)

def _predict_turnup(input:pd.DataFrame) -> pd.DataFrame:
    from kcatbench.model_wrapper.inner_wrapper.turnup_wrapper import TurNuPWrapper
    model = TurNuPWrapper()
    return model.predict(input)

def _predict_unikp(input:pd.DataFrame) -> pd.DataFrame:
    from kcatbench.model_wrapper.inner_wrapper.unikp_wrapper import UniKPWrapper
    model = UniKPWrapper()
    return model.predict(input)

def _predict_mmkcat(input:pd.DataFrame) -> pd.DataFrame:
    from kcatbench.model_wrapper.inner_wrapper.mmkcat_wrapper import MMKcatWrapper
    model = MMKcatWrapper()
    return model.predict(input)

PREDICT_HANDLERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "dlkcat": _predict_dlkcat,
    "catapro": _predict_catapro,
    "catpred": _predict_catpred,
    "turnup": _predict_turnup,
    "unikp": _predict_unikp,
    "mmkcat": _predict_mmkcat
}

def run_predict(model:str, input_path:Path, output_path:Path):
    _configure_worker_logging()

    if model not in PREDICT_HANDLERS:
        raise ValueError(f"Unknown model '{model}'.")

    progress_started(WORKER_LOGGER, "worker", "Worker started for model '%s'.", model)

    try:
        input = read_csv_with_schema(input_path)
        progress_completed(
            WORKER_LOGGER,
            "worker.input",
            "Loaded worker input rows=%s columns=%s.",
            len(input.index),
            len(input.columns),
        )

        progress_started(WORKER_LOGGER, "worker.predict", "Running model '%s' predict call.", model)
        output = PREDICT_HANDLERS[model](input)
        progress_completed(
            WORKER_LOGGER,
            "worker.predict",
            "Model '%s' predict call completed.",
            model,
        )

        output.to_csv(output_path, index=False)
        progress_completed(
            WORKER_LOGGER,
            "worker.output",
            "Worker output rows=%s written to %s.",
            len(output.index),
            output_path,
        )

    except Exception:
        print(f"--- WORKER EXCEPTION IN MODEL: {model} ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Conda worker that runs a kcat prediction model")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", type=str, required=True)
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)

    # Maybe in the future
    # train_parser = subparsers.add_parser("train")

    args = parser.parse_args(argv)

    if args.command == "predict":
        run_predict(args.model, args.input, args.output)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()