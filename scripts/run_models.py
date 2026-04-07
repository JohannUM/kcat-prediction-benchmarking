from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from kcatbench.model_wrapper.model import ENVIRONMENT_NAMES, Model
from kcatbench.util import RESULT_DIR, read_csv_with_schema


VALID_INPUT_SUFFIXES = {".csv", ".pkl", ".pickle"}
PREDICTIONS_DIR = RESULT_DIR / "predictions"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run one or more kcat models sequentially on an input table and "
			"checkpoint results after each model."
		)
	)
	parser.add_argument(
		"--models",
		nargs="+",
		required=True,
		help=(
			"Model identifiers to run in order (example: --models unikp dlkcat). "
			f"Valid values: {', '.join(sorted(ENVIRONMENT_NAMES.keys()))}."
		),
	)
	parser.add_argument(
		"--input",
		dest="input_path",
		type=Path,
		required=True,
		help="Input file path (.csv, .pkl, .pickle).",
	)
	parser.add_argument(
		"--output",
		dest="output_name",
		type=str,
		default=None,
		help=(
			"Optional output file name only (for example: my_run.csv). "
			"The file is always written under RESULT_DIR/predictions."
		),
	)
	parser.add_argument(
		"--log-file",
		dest="log_file",
		type=Path,
		default=None,
		help="Optional explicit log file path. If omitted, a log file next to output is used.",
	)
	parser.add_argument(
		"--fail-fast",
		action="store_true",
		help="Stop after the first model failure. By default, the runner continues.",
	)
	return parser.parse_args(argv)


def normalize_model_ids(model_ids: Sequence[str]) -> list[str]:
	normalized = [model_id.strip().lower() for model_id in model_ids if model_id.strip()]
	if not normalized:
		raise ValueError("No model identifiers were provided.")

	valid = set(ENVIRONMENT_NAMES.keys())
	invalid = [model_id for model_id in normalized if model_id not in valid]
	if invalid:
		valid_names = ", ".join(sorted(valid))
		raise ValueError(
			f"Invalid model identifiers: {invalid}. Valid model identifiers are: {valid_names}."
		)
	return normalized


def build_default_output_path(timestamp: str) -> Path:
	return PREDICTIONS_DIR / f"output_{timestamp}.csv"


def resolve_paths(output_name: Optional[str], log_file: Optional[Path], timestamp: str) -> tuple[Path, Path]:
	if output_name is None:
		output_path = build_default_output_path(timestamp)
	else:
		output_name_path = Path(output_name)
		if output_name_path.name != output_name or output_name_path.parent != Path("."):
			raise ValueError(
				"--output only accepts a file name (no directory components)."
			)
		output_path = PREDICTIONS_DIR / output_name

	if log_file is None:
		log_file = PREDICTIONS_DIR / f"{output_path.stem}.log"

	PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	log_file.parent.mkdir(parents=True, exist_ok=True)
	return output_path, log_file


def configure_logging(log_file: Path) -> logging.Logger:
	logger = logging.getLogger("run_models")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()

	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

	stream_handler = logging.StreamHandler(sys.stdout)
	stream_handler.setFormatter(formatter)

	file_handler = logging.FileHandler(log_file, encoding="utf-8")
	file_handler.setFormatter(formatter)

	logger.addHandler(stream_handler)
	logger.addHandler(file_handler)
	return logger


def load_input_dataframe(input_path: Path) -> pd.DataFrame:
	if not input_path.exists():
		raise FileNotFoundError(f"Input file does not exist: {input_path}")

	suffix = input_path.suffix.lower()
	if suffix not in VALID_INPUT_SUFFIXES:
		allowed = ", ".join(sorted(VALID_INPUT_SUFFIXES))
		raise ValueError(
			f"Unsupported input format '{suffix}' for file '{input_path}'. Supported formats: {allowed}."
		)

	if suffix == ".csv":
		df = read_csv_with_schema(input_path)
	else:
		df = pd.read_pickle(input_path)

	if not isinstance(df, pd.DataFrame):
		raise TypeError("Loaded input object is not a pandas DataFrame.")

	required_columns = {"sequence", "substrates"}
	missing_columns = required_columns - set(df.columns)
	if missing_columns:
		missing = ", ".join(sorted(missing_columns))
		raise ValueError(f"Input is missing required column(s): {missing}.")

	return df


def atomic_write_csv(df: pd.DataFrame, output_path: Path) -> None:
	tmp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
	df.to_csv(tmp_path, index=False)
	tmp_path.replace(output_path)


def run_models(
	model_ids: Sequence[str],
	initial_df: pd.DataFrame,
	output_path: Path,
	logger: logging.Logger,
	fail_fast: bool,
) -> dict[str, Any]:
	current_df = initial_df
	successful_models: list[str] = []
	failed_models: list[dict[str, str]] = []

	total = len(model_ids)
	for i, model_id in enumerate(model_ids, start=1):
		model_start = datetime.now()
		logger.info("Starting model %s (%d/%d)", model_id, i, total)

		model: Optional[Model] = None
		encountered_error = False
		error_message = ""

		try:
			model = Model(model_id)
			current_df = model.predict(current_df)
			elapsed_seconds = (datetime.now() - model_start).total_seconds()
			successful_models.append(model_id)
			logger.info("Model %s completed in %.2f seconds", model_id, elapsed_seconds)
		except Exception as exc:
			encountered_error = True
			error_message = str(exc)
			failed_models.append({"model": model_id, "error": error_message})
			logger.exception("Model %s failed", model_id)
		finally:
			if model is not None:
				del model
			gc.collect()
			atomic_write_csv(current_df, output_path)
			logger.info("Checkpoint written after model %s to %s", model_id, output_path)

		if encountered_error and fail_fast:
			logger.error("Fail-fast is enabled; stopping model execution.")
			break

	return {
		"successful_models": successful_models,
		"failed_models": failed_models,
		"rows": int(len(current_df)),
		"columns": list(current_df.columns),
	}


def write_status_file(output_path: Path, status_payload: dict[str, Any]) -> Path:
	status_path = output_path.with_name(f"{output_path.stem}.status.json")
	with open(status_path, "w", encoding="utf-8") as f:
		json.dump(status_payload, f, indent=2)
	return status_path


def main(argv: Optional[list[str]] = None) -> int:
	args = parse_args(argv)
	run_start = datetime.now()
	run_timestamp = run_start.strftime("%Y%m%d_%H%M%S")

	try:
		model_ids = normalize_model_ids(args.models)
		input_path = args.input_path.resolve()
		output_path, log_file = resolve_paths(args.output_name, args.log_file, run_timestamp)
	except Exception as exc:
		print(f"Argument validation failed: {exc}", file=sys.stderr)
		return 2

	logger = configure_logging(log_file)
	logger.info("Input file: %s", input_path)
	logger.info("Output file: %s", output_path)
	logger.info("Log file: %s", log_file)
	logger.info("Model order: %s", model_ids)
	logger.info("Process id: %s", os.getpid())

	try:
		input_df = load_input_dataframe(input_path)
		if "turnup" in model_ids and "products" not in input_df.columns:
			raise ValueError("Model 'turnup' requires a 'products' column in the input data.")

		summary = run_models(
			model_ids=model_ids,
			initial_df=input_df,
			output_path=output_path,
			logger=logger,
			fail_fast=args.fail_fast,
		)
	except Exception:
		logger.exception("Fatal error before completion")
		return 2

	run_end = datetime.now()
	duration_seconds = (run_end - run_start).total_seconds()

	status_payload = {
		"input": str(input_path),
		"output": str(output_path),
		"log_file": str(log_file),
		"model_ids": model_ids,
		"successful_models": summary["successful_models"],
		"failed_models": summary["failed_models"],
		"rows": summary["rows"],
		"columns": summary["columns"],
		"started_at": run_start.isoformat(),
		"finished_at": run_end.isoformat(),
		"duration_seconds": duration_seconds,
	}
	status_path = write_status_file(output_path, status_payload)

	if summary["failed_models"]:
		logger.error(
			"Run completed with failures. Successful: %d, Failed: %d",
			len(summary["successful_models"]),
			len(summary["failed_models"]),
		)
		logger.error("Status file: %s", status_path)
		return 1

	logger.info(
		"Run completed successfully. Models run: %d. Status file: %s",
		len(summary["successful_models"]),
		status_path,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
