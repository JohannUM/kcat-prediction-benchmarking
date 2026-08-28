from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from kcatbench.dataset.util import aggregate_rows_by_columns, compare_dataset_overlap
from kcatbench.util import DATA_DIR, read_csv_with_schema


VALID_DATASET_SUFFIXES = {".csv", ".json", ".pkl", ".pickle"}
DEFAULT_OUTPUT_DIRNAME = "leakage"
DEFAULT_STATUS_FILENAME = "leakage_audit.status.json"


@dataclass(frozen=True)
class DatasetSpec:
	path: Path
	format: str
	loader: str


@dataclass(frozen=True)
class ColumnSpec:
	sequence: str
	substrates: str
	ec_number: str
	products: str | None = None


@dataclass(frozen=True)
class AggregateSpec:
	key_columns: list[str]
	aggregation_strategies: dict[str, str]


@dataclass(frozen=True)
class AuditSpec:
	name: str
	training: DatasetSpec
	training_columns: ColumnSpec
	benchmark_columns: ColumnSpec | None
	output_filename: str | None
	aggregate: AggregateSpec | None
	sequence_similarity_thresholds: tuple[float, ...]


@dataclass(frozen=True)
class SetupSpec:
	benchmark: DatasetSpec
	benchmark_columns: ColumnSpec
	audits: list[AuditSpec]
	output_dirname: str
	combined_output_filename: str | None
	status_filename: str


VALID_FORMAT_TO_LOADER = {
	"csv": "read_csv_with_schema",
	"json": "read_json",
	"pkl": "read_pickle",
	"pickle": "read_pickle",
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run one benchmark-vs-training leakage audit from a JSON setup file and "
			"checkpoint report outputs under DATA_DIR/leakage."
		)
	)
	parser.add_argument(
		"--setup-file",
		type=Path,
		required=True,
		help=(
			"JSON setup file containing the benchmark entry and one or more audit "
			"definitions. Dataset paths in the file are resolved relative to DATA_DIR."
		),
	)
	parser.add_argument(
		"--log-file",
		type=Path,
		default=None,
		help=(
			"Optional explicit log file path. If omitted, a timestamped log file is "
			"created under DATA_DIR/leakage."
		),
	)
	return parser.parse_args(argv)


def configure_logging(log_file: Path) -> logging.Logger:
	logger = logging.getLogger("run_leakage_audit")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	logger.propagate = False

	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

	stream_handler = logging.StreamHandler(sys.stdout)
	stream_handler.setFormatter(formatter)

	file_handler = logging.FileHandler(log_file, encoding="utf-8")
	file_handler.setFormatter(formatter)

	logger.addHandler(stream_handler)
	logger.addHandler(file_handler)
	return logger


def _resolve_data_path(relative_path: str | Path) -> Path:
	path = Path(relative_path)
	if path.is_absolute():
		return path
	return DATA_DIR / path


def _ensure_file_name_only(file_name: str, argument_name: str) -> str:
	candidate = Path(str(file_name).strip())
	if not candidate.name or candidate.name != str(file_name).strip() or candidate.parent != Path("."):
		raise ValueError(f"{argument_name} must be a file name only, without directories.")
	return candidate.name


def _load_json_file(path: Path) -> dict[str, Any]:
	if not path.exists():
		raise FileNotFoundError(f"Setup file does not exist: {path}")
	with open(path, "r", encoding="utf-8") as f:
		payload = json.load(f)
	if not isinstance(payload, dict):
		raise ValueError("Setup file must contain a JSON object at the top level.")
	return payload


def _normalize_format(format_value: str | None, file_path: Path) -> str:
	if format_value is not None:
		normalized = format_value.strip().lower()
		if normalized not in VALID_FORMAT_TO_LOADER:
			valid = ", ".join(sorted(VALID_FORMAT_TO_LOADER))
			raise ValueError(
				f"Unsupported dataset format '{format_value}'. Supported formats: {valid}."
			)
		return normalized

	suffix = file_path.suffix.lower()
	if suffix not in VALID_DATASET_SUFFIXES:
		valid = ", ".join(sorted(VALID_DATASET_SUFFIXES))
		raise ValueError(
			f"Could not infer dataset format from '{file_path}'. Supported suffixes: {valid}."
		)
	return suffix.lstrip(".")


def _load_dataset(spec: DatasetSpec) -> pd.DataFrame:
	resolved_path = _resolve_data_path(spec.path)
	if not resolved_path.exists():
		raise FileNotFoundError(f"Dataset file not found: {resolved_path}")

	if spec.loader == "read_csv_with_schema":
		return read_csv_with_schema(resolved_path)
	if spec.loader == "read_json":
		return pd.read_json(resolved_path)
	if spec.loader == "read_pickle":
		return pd.read_pickle(resolved_path)

	raise ValueError(f"Unsupported loader '{spec.loader}'.")


def _parse_dataset_spec(payload: dict[str, Any], default_loader: str | None = None) -> DatasetSpec:
	path_value = payload.get("path")
	if not isinstance(path_value, str) or not path_value.strip():
		raise ValueError("Each dataset spec must define a non-empty 'path'.")
	path = Path(path_value.strip())

	format_value = payload.get("format")
	format_name = _normalize_format(format_value, path)
	loader = payload.get("loader") or VALID_FORMAT_TO_LOADER[format_name]
	if default_loader is not None and payload.get("loader") is None:
		loader = default_loader
	return DatasetSpec(path=path, format=format_name, loader=loader)


def _parse_column_spec(payload: dict[str, Any], *, allow_missing_products: bool = False) -> ColumnSpec:
	sequence = payload.get("sequence")
	substrates = payload.get("substrates")
	ec_number = payload.get("ec_number")
	products = payload.get("products")
	if not all(isinstance(value, str) and value.strip() for value in (sequence, substrates, ec_number)):
		raise ValueError(
			"Column mapping must define non-empty string values for 'sequence', 'substrates', and 'ec_number'."
		)
	if products is None and not allow_missing_products:
		raise ValueError("Column mapping must define 'products' or set it to null explicitly.")
	if products is not None and not isinstance(products, str):
		raise ValueError("'products' must be a string column name or null.")
	return ColumnSpec(
		sequence=sequence.strip(),
		substrates=substrates.strip(),
		ec_number=ec_number.strip(),
		products=products.strip() if isinstance(products, str) else None,
	)


def _parse_aggregate_spec(payload: dict[str, Any] | None) -> AggregateSpec | None:
	if payload is None:
		return None
	if not isinstance(payload, dict):
		raise ValueError("'aggregate' must be an object if provided.")

	key_columns = payload.get("key_columns")
	strategies = payload.get("aggregation_strategies")
	if not isinstance(key_columns, list) or not all(isinstance(item, str) and item.strip() for item in key_columns):
		raise ValueError("'aggregate.key_columns' must be a list of non-empty strings.")
	if strategies is None:
		aggregation_strategies: dict[str, str] = {}
	elif isinstance(strategies, dict):
		aggregation_strategies = {}
		for column_name, strategy in strategies.items():
			if not isinstance(column_name, str) or not column_name.strip():
				raise ValueError("Aggregation strategy keys must be non-empty strings.")
			if not isinstance(strategy, str) or not strategy.strip():
				raise ValueError("Aggregation strategy values must be non-empty strings.")
			aggregation_strategies[column_name.strip()] = strategy.strip()
	else:
		raise ValueError("'aggregate.aggregation_strategies' must be an object if provided.")

	return AggregateSpec(
		key_columns=[item.strip() for item in key_columns],
		aggregation_strategies=aggregation_strategies,
	)


def parse_setup_file(setup_path: Path) -> SetupSpec:
	payload = _load_json_file(setup_path)

	benchmark_payload = payload.get("benchmark")
	if not isinstance(benchmark_payload, dict):
		raise ValueError("Setup file must contain a 'benchmark' object.")
	benchmark_dataset = _parse_dataset_spec(benchmark_payload)
	benchmark_columns = _parse_column_spec(benchmark_payload.get("columns", {}), allow_missing_products=True)

	audits_payload = payload.get("audits")
	if not isinstance(audits_payload, list) or not audits_payload:
		raise ValueError("Setup file must contain a non-empty 'audits' list.")

	audits: list[AuditSpec] = []
	for audit_index, audit_payload in enumerate(audits_payload, start=1):
		if not isinstance(audit_payload, dict):
			raise ValueError(f"Audit entry at index {audit_index} must be an object.")
		name = audit_payload.get("name")
		if not isinstance(name, str) or not name.strip():
			raise ValueError(f"Audit entry at index {audit_index} must define a non-empty 'name'.")

		training_payload = audit_payload.get("training")
		if not isinstance(training_payload, dict):
			raise ValueError(f"Audit '{name}' must contain a 'training' object.")
		training_dataset = _parse_dataset_spec(training_payload)
		training_columns = _parse_column_spec(training_payload.get("columns", {}), allow_missing_products=True)

		benchmark_columns_override_payload = audit_payload.get("benchmark_columns")
		benchmark_columns_override = (
			_parse_column_spec(benchmark_columns_override_payload, allow_missing_products=True)
			if benchmark_columns_override_payload is not None
			else None
		)

		output_filename = audit_payload.get("output_filename")
		if output_filename is not None:
			if not isinstance(output_filename, str) or not output_filename.strip():
				raise ValueError(f"Audit '{name}' has an invalid 'output_filename'.")
			output_filename = _ensure_file_name_only(output_filename, f"Audit '{name}' output_filename")

		aggregate = _parse_aggregate_spec(audit_payload.get("aggregate"))

		thresholds_payload = audit_payload.get("sequence_similarity_thresholds", [0.5, 0.7, 0.9])
		if not isinstance(thresholds_payload, list) or not thresholds_payload:
			raise ValueError(
				f"Audit '{name}' must define 'sequence_similarity_thresholds' as a non-empty list."
			)
		thresholds: list[float] = []
		for threshold in thresholds_payload:
			if not isinstance(threshold, (int, float)):
				raise ValueError(
					f"Audit '{name}' has a non-numeric sequence similarity threshold: {threshold!r}."
				)
			thresholds.append(float(threshold))

		audits.append(
			AuditSpec(
				name=name.strip(),
				training=training_dataset,
				training_columns=training_columns,
				benchmark_columns=benchmark_columns_override,
				output_filename=output_filename,
				aggregate=aggregate,
				sequence_similarity_thresholds=tuple(thresholds),
			)
		)

	output_dirname = payload.get("output_dirname", DEFAULT_OUTPUT_DIRNAME)
	if not isinstance(output_dirname, str) or not output_dirname.strip():
		raise ValueError("'output_dirname' must be a non-empty string if provided.")

	combined_output_filename = payload.get("combined_output_filename", "leakage_audit_summary.csv")
	if combined_output_filename is not None:
		if not isinstance(combined_output_filename, str) or not combined_output_filename.strip():
			raise ValueError("'combined_output_filename' must be a non-empty string if provided.")
		combined_output_filename = _ensure_file_name_only(
			combined_output_filename, "combined_output_filename"
		)

	status_filename = payload.get("status_filename", DEFAULT_STATUS_FILENAME)
	if not isinstance(status_filename, str) or not status_filename.strip():
		raise ValueError("'status_filename' must be a non-empty string if provided.")
	status_filename = _ensure_file_name_only(status_filename, "status_filename")

	return SetupSpec(
		benchmark=benchmark_dataset,
		benchmark_columns=benchmark_columns,
		audits=audits,
		output_dirname=output_dirname.strip(),
		combined_output_filename=combined_output_filename,
		status_filename=status_filename,
	)


def resolve_run_paths(
	log_file: Optional[Path], output_dirname: str, timestamp: str, status_filename: str
) -> tuple[Path, Path, Path]:
	output_dir = DATA_DIR / output_dirname
	output_dir.mkdir(parents=True, exist_ok=True)

	if log_file is None:
		log_path = output_dir / f"leakage_audit_{timestamp}.log"
	else:
		log_path = log_file.resolve()
		log_path.parent.mkdir(parents=True, exist_ok=True)

	status_path = output_dir / status_filename
	return output_dir, log_path, status_path


def configure_logger(log_path: Path) -> logging.Logger:
	logger = logging.getLogger("run_leakage_audit")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	logger.propagate = False

	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
	stdout_handler = logging.StreamHandler(sys.stdout)
	stdout_handler.setFormatter(formatter)
	file_handler = logging.FileHandler(log_path, encoding="utf-8")
	file_handler.setFormatter(formatter)
	logger.addHandler(stdout_handler)
	logger.addHandler(file_handler)
	return logger


def _read_setup_preview(setup_path: Path) -> str:
	return str(setup_path.resolve())


def _benchmark_progress_callback(logger: logging.Logger, audit_name: str) -> Any:
	def emit(message: str) -> None:
		logger.info("[%s] %s", audit_name, message)

	return emit


def _load_and_prepare_benchmark(
	setup: SetupSpec,
	logger: logging.Logger,
) -> pd.DataFrame:
	benchmark_path = _resolve_data_path(setup.benchmark.path)
	logger.info("Loading benchmark dataset from %s", benchmark_path)
	benchmark_df = _load_dataset(setup.benchmark)
	logger.info("Loaded benchmark rows: %d, columns: %d", len(benchmark_df), len(benchmark_df.columns))
	if setup.benchmark_columns.products is not None and setup.benchmark_columns.products not in benchmark_df.columns:
		raise ValueError(
			f"Benchmark dataset is missing product column '{setup.benchmark_columns.products}'."
		)
	return benchmark_df


def _prepare_benchmark_for_audit(
	benchmark_df: pd.DataFrame,
	setup: SetupSpec,
	audit: AuditSpec,
	logger: logging.Logger,
) -> pd.DataFrame:
	if audit.aggregate is None:
		return benchmark_df

	logger.info(
		"[%s] Aggregating benchmark rows using key_columns=%s and strategies=%s",
		audit.name,
		audit.aggregate.key_columns,
		audit.aggregate.aggregation_strategies,
	)
	return aggregate_rows_by_columns(
		benchmark_df,
		key_columns=audit.aggregate.key_columns,
		aggregation_strategies=audit.aggregate.aggregation_strategies,
	)


def run_audits(
	setup: SetupSpec,
	benchmark_df: pd.DataFrame,
	output_dir: Path,
	logger: logging.Logger,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
	combined_frames: list[pd.DataFrame] = []
	manifest: list[dict[str, Any]] = []
	total_audits = len(setup.audits)

	for audit_index, audit in enumerate(setup.audits, start=1):
		audit_start = datetime.now()
		logger.info("[%s] Starting audit %d/%d", audit.name, audit_index, total_audits)

		training_path = _resolve_data_path(audit.training.path)
		logger.info("[%s] Loading training dataset from %s", audit.name, training_path)
		training_df = _load_dataset(audit.training)
		logger.info(
			"[%s] Loaded training rows: %d, columns: %d",
			audit.name,
			len(training_df),
			len(training_df.columns),
		)

		benchmark_for_audit = _prepare_benchmark_for_audit(benchmark_df, setup, audit, logger)

		benchmark_columns = audit.benchmark_columns or setup.benchmark_columns

		audit_output_filename = audit.output_filename or f"{audit.name}.csv"
		report_path = output_dir / audit_output_filename
		logger.info("[%s] Writing report to %s", audit.name, report_path)

		try:
			result_df = compare_dataset_overlap(
				benchmark_df=benchmark_for_audit,
				training_df=training_df,
				benchmark_sequence_column=benchmark_columns.sequence,
				training_sequence_column=audit.training_columns.sequence,
				benchmark_substrate_column=benchmark_columns.substrates,
				training_substrate_column=audit.training_columns.substrates,
				benchmark_ec_column=benchmark_columns.ec_number,
				training_ec_column=audit.training_columns.ec_number,
				benchmark_product_column=benchmark_columns.products,
				training_product_column=audit.training_columns.products,
				sequence_similarity_thresholds=audit.sequence_similarity_thresholds,
				output_filename=audit_output_filename,
				progress_callback=_benchmark_progress_callback(logger, audit.name),
			)
			result_df = result_df.copy()
			result_df.insert(0, "audit_name", audit.name)
			result_df.insert(1, "benchmark_dataset", str(setup.benchmark.path))
			result_df.insert(2, "training_dataset", str(audit.training.path))
			combined_frames.append(result_df)

			elapsed_seconds = (datetime.now() - audit_start).total_seconds()
			manifest.append(
				{
					"audit_name": audit.name,
					"training_dataset": str(audit.training.path),
					"report_file": str(report_path),
					"rows": len(result_df),
					"duration_seconds": elapsed_seconds,
					"status": "completed",
				}
			)
			logger.info("[%s] Completed audit in %.2f seconds", audit.name, elapsed_seconds)
		except Exception as exc:
			elapsed_seconds = (datetime.now() - audit_start).total_seconds()
			manifest.append(
				{
					"audit_name": audit.name,
					"training_dataset": str(audit.training.path),
					"report_file": str(report_path),
					"rows": None,
					"duration_seconds": elapsed_seconds,
					"status": "failed",
					"error": str(exc),
				}
			)
			logger.exception("[%s] Audit failed; continuing to next audit", audit.name)

	combined_df = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
	return combined_df, manifest


def write_status_file(status_path: Path, payload: dict[str, Any]) -> None:
	with open(status_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def main(argv: Optional[list[str]] = None) -> int:
	args = parse_args(argv)
	start_time = datetime.now()
	timestamp = start_time.strftime("%Y%m%d_%H%M%S")

	try:
		setup = parse_setup_file(args.setup_file)
		output_dir, log_path, status_path = resolve_run_paths(
			args.log_file, setup.output_dirname, timestamp, setup.status_filename
		)
		logger = configure_logger(log_path)
		expected_setup_path = _read_setup_preview(args.setup_file)
		logger.info("Setup file: %s", expected_setup_path)
		logger.info("Output directory: %s", output_dir)
		logger.info("Log file: %s", log_path)
		logger.info("Status file: %s", status_path)
		logger.info("Audit count: %d", len(setup.audits))
		logger.info("Process id: %s", os.getpid())

		benchmark_df = _load_and_prepare_benchmark(setup, logger)
		combined_df, manifest = run_audits(setup, benchmark_df, output_dir, logger)

		combined_output_path: Path | None = None
		if setup.combined_output_filename is not None:
			combined_output_path = output_dir / setup.combined_output_filename
			logger.info("Writing combined summary to %s", combined_output_path)
			combined_df.to_csv(combined_output_path, index=False)

		status_payload = {
			"setup_file": expected_setup_path,
			"output_dir": str(output_dir),
			"log_file": str(log_path),
			"combined_output_file": str(combined_output_path) if combined_output_path else None,
			"audit_manifest": manifest,
			"started_at": start_time.isoformat(),
			"finished_at": datetime.now().isoformat(),
			"duration_seconds": (datetime.now() - start_time).total_seconds(),
		}
		write_status_file(status_path, status_payload)
		has_failures = any(entry.get("status") != "completed" for entry in manifest)

		logger.info("Leakage audit run completed. Status file: %s", status_path)
		return 1 if has_failures else 0
	except Exception:
		try:
			logger.exception("Leakage audit run failed")
		except Exception:
			print("Leakage audit run failed", file=sys.stderr)
			return 2
		return 2


if __name__ == "__main__":
	raise SystemExit(main())
