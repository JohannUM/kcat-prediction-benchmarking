from __future__ import annotations

import ast
import json
from collections.abc import Callable
from functools import lru_cache
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner

from kcatbench.util import DATA_DIR


AggregationStrategy = Literal["mean", "median", "max", "geometric_mean"]

_SUPPORTED_AGGREGATION_STRATEGIES: set[str] = {
	"mean",
	"median",
	"max",
	"geometric_mean",
}


def _geometric_mean(series: pd.Series, column_name: str) -> float:
	values = series.dropna()
	if values.empty:
		return np.nan

	if (values < 0).any():
		raise ValueError(
			f"Column '{column_name}' contains negative values. "
			"Geometric mean is undefined for negative values."
		)

	positive_values = values[values > 0]
	if positive_values.empty:
		return np.nan

	return float(np.exp(np.log(positive_values).mean()))


def _validate_columns_exist(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
	missing_columns = [column for column in columns if column not in df.columns]
	if missing_columns:
		missing = ", ".join(sorted(missing_columns))
		raise ValueError(f"Missing {context} column(s): {missing}")


def _make_hashable_key_value(value: Any) -> Any:
	"""Convert list-like key values into hashable objects for internal grouping only."""
	if isinstance(value, list):
		return tuple(_make_hashable_key_value(item) for item in value)
	if isinstance(value, tuple):
		return tuple(_make_hashable_key_value(item) for item in value)
	if isinstance(value, dict):
		return tuple(
			sorted((str(key), _make_hashable_key_value(val)) for key, val in value.items())
		)
	return value


def aggregate_rows_by_columns(
	df: pd.DataFrame,
	key_columns: Sequence[str],
	aggregation_strategies: Mapping[str, AggregationStrategy] | None = None,
) -> pd.DataFrame:
	"""Aggregate duplicate rows by exact key-column matches.

	Rows are grouped when all values in ``key_columns`` match exactly and none of the
	key values are missing. Rows with missing values in any key column are preserved
	as individual rows and are not grouped.

	Args:
		df: Input DataFrame.
		key_columns: Columns used to identify duplicate groups.
		aggregation_strategies: Mapping of numeric column names to aggregation strategy.
			Supported strategies are ``mean``, ``median``, ``max``, and
			``geometric_mean``.

	Returns:
		Aggregated DataFrame with the same columns as the input. For columns not listed
		in ``aggregation_strategies``, the value from the first row in each group is used.

	Raises:
		ValueError: If key columns or aggregation columns are missing, if an unsupported
			aggregation strategy is provided, if aggregation columns are non-numeric, or
			if geometric mean is requested for data containing negative values.
	"""
	if not key_columns:
		raise ValueError("key_columns must contain at least one column name.")

	_validate_columns_exist(df=df, columns=key_columns, context="key")

	strategies = dict(aggregation_strategies or {})
	_validate_columns_exist(df=df, columns=list(strategies), context="aggregation")

	overlapping_columns = sorted(set(key_columns).intersection(strategies))
	if overlapping_columns:
		overlap = ", ".join(overlapping_columns)
		raise ValueError(
			"Columns cannot be used as both key and aggregation columns: "
			f"{overlap}"
		)

	invalid_strategies = sorted(
		{
			strategy
			for strategy in strategies.values()
			if strategy not in _SUPPORTED_AGGREGATION_STRATEGIES
		}
	)
	if invalid_strategies:
		names = ", ".join(invalid_strategies)
		supported = ", ".join(sorted(_SUPPORTED_AGGREGATION_STRATEGIES))
		raise ValueError(
			f"Unsupported aggregation strategy value(s): {names}. "
			f"Supported values are: {supported}"
		)

	non_numeric_columns = [
		column
		for column in strategies
		if not pd.api.types.is_numeric_dtype(df[column])
	]
	if non_numeric_columns:
		columns = ", ".join(sorted(non_numeric_columns))
		raise ValueError(
			"Aggregation strategies can only be applied to numeric columns. "
			f"Non-numeric column(s): {columns}"
		)

	if df.empty:
		return df.copy().reset_index(drop=True)

	working_df = df.copy()
	working_df["__original_order"] = np.arange(len(working_df))

	group_key_columns: list[str] = []
	existing_columns = set(working_df.columns)
	for index, key_column in enumerate(key_columns):
		base_group_key_column = f"__group_key_{index}"
		group_key_column = base_group_key_column
		suffix = 1
		while group_key_column in existing_columns:
			group_key_column = f"{base_group_key_column}_{suffix}"
			suffix += 1

		group_key_columns.append(group_key_column)
		existing_columns.add(group_key_column)
		working_df[group_key_column] = working_df[key_column].map(_make_hashable_key_value)

	key_complete_mask = working_df[key_columns].notna().all(axis=1)
	grouped_source_df = working_df.loc[key_complete_mask]
	passthrough_df = working_df.loc[~key_complete_mask].copy()

	if grouped_source_df.empty:
		result_df = passthrough_df
	else:
		grouped = grouped_source_df.groupby(group_key_columns, sort=False, dropna=False)
		grouped_first_rows = grouped.head(1).copy().set_index(group_key_columns)

		for column_name, strategy in strategies.items():
			if strategy == "geometric_mean":
				grouped_first_rows[column_name] = grouped[column_name].apply(
					lambda series: _geometric_mean(series, column_name)
				)
			else:
				grouped_first_rows[column_name] = grouped[column_name].agg(strategy)

		grouped_result_df = grouped_first_rows.reset_index(drop=True)
		result_df = pd.concat([grouped_result_df, passthrough_df], ignore_index=True)

	result_df = result_df.sort_values("__original_order", kind="stable")
	result_df = result_df.drop(columns=["__original_order", *group_key_columns])
	return result_df.reset_index(drop=True)


def _is_missing_value(value: Any) -> bool:
	if value is None:
		return True

	if isinstance(value, (list, tuple, set, dict)):
		return False

	try:
		missing = pd.isna(value)
	except (TypeError, ValueError):
		return False

	return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _normalize_text_value(value: Any) -> str | None:
	if _is_missing_value(value):
		return None

	if isinstance(value, str):
		normalized = value.strip()
		return normalized or None

	normalized = str(value).strip()
	return normalized or None


def _parse_list_text(value: str) -> list[Any] | None:
	if not value:
		return []

	try:
		parsed = json.loads(value)
	except json.JSONDecodeError:
		try:
			parsed = ast.literal_eval(value)
		except (ValueError, SyntaxError):
			return None

	if isinstance(parsed, (list, tuple, set)):
		return list(parsed)

	return None


def _normalize_list_like_value(value: Any) -> list[str]:
	if _is_missing_value(value):
		return []

	if isinstance(value, list):
		raw_values: Sequence[Any] = value
	elif isinstance(value, tuple):
		raw_values = list(value)
	elif isinstance(value, set):
		raw_values = sorted(value, key=str)
	elif isinstance(value, str):
		text = value.strip()
		if not text:
			return []

		if (text.startswith("[") and text.endswith("]")) or (
			text.startswith("(") and text.endswith(")")
		):
			parsed_values = _parse_list_text(text)
			if parsed_values is None:
				raw_values = [text]
			else:
				raw_values = parsed_values
		else:
			return [text]
	else:
		normalized = _normalize_text_value(value)
		return [normalized] if normalized is not None else []

	normalized_values: list[str] = []
	for item in raw_values:
		normalized_item = _normalize_text_value(item)
		if normalized_item is not None:
			normalized_values.append(normalized_item)

	return normalized_values


def _first_list_item(value: Any) -> str | None:
	normalized_values = _normalize_list_like_value(value)
	if not normalized_values:
		return None
	return normalized_values[0]


def _full_list_signature(value: Any) -> tuple[str, ...] | None:
	normalized_values = _normalize_list_like_value(value)
	if not normalized_values:
		return None
	return tuple(sorted(normalized_values))


def _clean_values(values: Sequence[Any]) -> list[Any]:
	return [value for value in values if value is not None]


def _overlap_summary_row(
	*,
	feature: str,
	representation: str,
	analysis_scope: str,
	benchmark_total_rows: int,
	training_total_rows: int,
	benchmark_values: Sequence[Any],
	training_values: Sequence[Any],
	threshold: float | None = None,
	best_match_scores: Sequence[float] | None = None,
) -> dict[str, Any]:
	benchmark_valid_values = _clean_values(benchmark_values)
	training_valid_values = _clean_values(training_values)

	benchmark_unique_values = len(set(benchmark_valid_values))
	training_unique_values = len(set(training_valid_values))
	benchmark_missing_rows = benchmark_total_rows - len(benchmark_valid_values)
	training_missing_rows = training_total_rows - len(training_valid_values)

	benchmark_set = set(benchmark_valid_values)
	training_set = set(training_valid_values)
	shared_unique_values = len(benchmark_set.intersection(training_set))

	if best_match_scores is None:
		benchmark_row_hits = sum(value in training_set for value in benchmark_valid_values)
		training_row_hits: int | float = sum(
			value in benchmark_set for value in training_valid_values
		)
		benchmark_valid_count = len(benchmark_valid_values)
		training_valid_count = len(training_valid_values)
		best_match_identity_mean = np.nan
		best_match_identity_median = np.nan
		best_match_identity_min = np.nan
		best_match_identity_max = np.nan
		best_match_hit_count = np.nan
		best_match_hit_rate = np.nan
	else:
		benchmark_row_hits = sum(score >= (threshold or 0.0) for score in best_match_scores)
		training_row_hits = np.nan
		benchmark_valid_count = len(best_match_scores)
		training_valid_count = len(training_valid_values)
		best_match_identity_mean = float(np.mean(best_match_scores)) if best_match_scores else np.nan
		best_match_identity_median = (
			float(np.median(best_match_scores)) if best_match_scores else np.nan
		)
		best_match_identity_min = float(np.min(best_match_scores)) if best_match_scores else np.nan
		best_match_identity_max = float(np.max(best_match_scores)) if best_match_scores else np.nan
		best_match_hit_count = int(benchmark_row_hits)
		best_match_hit_rate = (
			benchmark_row_hits / benchmark_valid_count if benchmark_valid_count else np.nan
		)

	benchmark_row_hit_rate = (
		benchmark_row_hits / benchmark_valid_count if benchmark_valid_count else np.nan
	)
	training_row_hit_rate = (
		training_row_hits / training_valid_count
		if isinstance(training_row_hits, (int, np.integer)) and training_valid_count
		else np.nan
	)
	benchmark_unique_hit_rate = (
		shared_unique_values / benchmark_unique_values if benchmark_unique_values else np.nan
	)
	training_unique_hit_rate = (
		shared_unique_values / training_unique_values if training_unique_values else np.nan
	)

	return {
		"feature": feature,
		"representation": representation,
		"analysis_scope": analysis_scope,
		"comparison_type": "sequence_similarity" if best_match_scores is not None else "exact_overlap",
		"similarity_threshold": threshold,
		"benchmark_total_rows": benchmark_total_rows,
		"benchmark_valid_rows": benchmark_valid_count,
		"benchmark_missing_rows": benchmark_missing_rows,
		"training_total_rows": training_total_rows,
		"training_valid_rows": training_valid_count,
		"training_missing_rows": training_missing_rows,
		"benchmark_unique_values": benchmark_unique_values,
		"training_unique_values": training_unique_values,
		"shared_unique_values": shared_unique_values,
		"benchmark_row_hits": int(benchmark_row_hits),
		"training_row_hits": int(training_row_hits) if isinstance(training_row_hits, (int, np.integer)) else np.nan,
		"benchmark_row_hit_rate": benchmark_row_hit_rate,
		"training_row_hit_rate": training_row_hit_rate,
		"benchmark_unique_hit_rate": benchmark_unique_hit_rate,
		"training_unique_hit_rate": training_unique_hit_rate,
		"best_match_identity_mean": best_match_identity_mean,
		"best_match_identity_median": best_match_identity_median,
		"best_match_identity_min": best_match_identity_min,
		"best_match_identity_max": best_match_identity_max,
		"best_match_hit_count": best_match_hit_count,
		"best_match_hit_rate": best_match_hit_rate,
	}


@lru_cache(maxsize=200_000)
def _sequence_identity(sequence_a: str, sequence_b: str) -> float:
	if sequence_a == sequence_b:
		return 1.0

	if not sequence_a or not sequence_b:
		return 0.0

	aligner = PairwiseAligner()
	aligner.mode = "global"
	aligner.match_score = 1.0
	aligner.mismatch_score = 0.0
	aligner.open_gap_score = 0.0
	aligner.extend_gap_score = 0.0
	return float(aligner.score(sequence_a, sequence_b)) / max(len(sequence_a), len(sequence_b))


def _best_match_sequence_scores(
	benchmark_sequences: Sequence[str], training_sequences: Sequence[str]
) -> list[float]:
	unique_training_sequences = list(dict.fromkeys(training_sequences))
	best_scores: list[float] = []

	for benchmark_sequence in dict.fromkeys(benchmark_sequences):
		best_score = 0.0
		benchmark_length = len(benchmark_sequence)

		for training_sequence in unique_training_sequences:
			longer_length = max(benchmark_length, len(training_sequence))
			if longer_length == 0:
				continue

			max_possible_score = min(benchmark_length, len(training_sequence)) / longer_length
			if max_possible_score <= best_score:
				continue

			score = _sequence_identity(benchmark_sequence, training_sequence)
			if score > best_score:
				best_score = score
				if best_score == 1.0:
					break

		best_scores.append(best_score)

	return best_scores


def compare_dataset_overlap(
	benchmark_df: pd.DataFrame,
	training_df: pd.DataFrame,
	*,
	benchmark_sequence_column: str,
	training_sequence_column: str,
	benchmark_substrate_column: str,
	training_substrate_column: str,
	benchmark_ec_column: str,
	training_ec_column: str,
	benchmark_product_column: str | None = None,
	training_product_column: str | None = None,
	sequence_similarity_thresholds: Sequence[float] = (0.5, 0.7, 0.9),
	output_filename: str | None = None,
	progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
	"""Summarize overlap between a benchmark dataframe and a training dataframe.

	The report focuses on exact overlap for sequence, EC number, substrate, product,
	and substrate + sequence pairs. For list-valued substrate and product columns,
	both first-item overlap and full-list overlap are reported. Sequence similarity
	is computed as best-match alignment-based identity against the training set and
	reported at the requested thresholds.

	Args:
		benchmark_df: Benchmark dataframe to audit.
		training_df: Training dataframe to compare against.
		benchmark_sequence_column: Sequence column in benchmark_df.
		training_sequence_column: Sequence column in training_df.
		benchmark_substrate_column: Substrate column in benchmark_df.
		training_substrate_column: Substrate column in training_df.
		benchmark_ec_column: EC number column in benchmark_df.
		training_ec_column: EC number column in training_df.
		benchmark_product_column: Optional product column in benchmark_df.
		training_product_column: Optional product column in training_df.
		sequence_similarity_thresholds: Thresholds to report for best-match sequence
			similarity. Values should be between 0 and 1.
		output_filename: Optional CSV file name to write under DATA_DIR / "leakage".
		progress_callback: Optional callable that receives progress messages.

	Returns:
		A dataframe with one row per overlap category or similarity threshold.

	Raises:
		ValueError: If required columns are missing.
	"""
	def emit_progress(message: str) -> None:
		if progress_callback is not None:
			progress_callback(message)

	required_benchmark_columns = [
		benchmark_sequence_column,
		benchmark_substrate_column,
		benchmark_ec_column,
	]
	required_training_columns = [
		training_sequence_column,
		training_substrate_column,
		training_ec_column,
	]
	if benchmark_product_column is not None:
		required_benchmark_columns.append(benchmark_product_column)
	if training_product_column is not None:
		required_training_columns.append(training_product_column)

	_validate_columns_exist(benchmark_df, required_benchmark_columns, "benchmark")
	_validate_columns_exist(training_df, required_training_columns, "training")
	emit_progress("Validated required columns")

	if len(sequence_similarity_thresholds) == 0:
		raise ValueError("sequence_similarity_thresholds must contain at least one value.")
	emit_progress("Starting report construction")

	benchmark_rows_total = len(benchmark_df)
	training_rows_total = len(training_df)

	benchmark_sequence_values = [
		_normalize_text_value(value)
		for value in benchmark_df[benchmark_sequence_column].tolist()
	]
	training_sequence_values = [
		_normalize_text_value(value)
		for value in training_df[training_sequence_column].tolist()
	]

	benchmark_ec_values = [
		_normalize_text_value(value)
		for value in benchmark_df[benchmark_ec_column].tolist()
	]
	training_ec_values = [
		_normalize_text_value(value)
		for value in training_df[training_ec_column].tolist()
	]

	benchmark_substrate_first_values = [
		_first_list_item(value)
		for value in benchmark_df[benchmark_substrate_column].tolist()
	]
	training_substrate_first_values = [
		_first_list_item(value)
		for value in training_df[training_substrate_column].tolist()
	]
	benchmark_substrate_full_values = [
		_full_list_signature(value)
		for value in benchmark_df[benchmark_substrate_column].tolist()
	]
	training_substrate_full_values = [
		_full_list_signature(value)
		for value in training_df[training_substrate_column].tolist()
	]

	benchmark_product_first_values: list[str | None] = []
	training_product_first_values: list[str | None] = []
	benchmark_product_full_values: list[tuple[str, ...] | None] = []
	training_product_full_values: list[tuple[str, ...] | None] = []
	products_available = (
		benchmark_product_column is not None and training_product_column is not None
	)
	if products_available:
		benchmark_product_first_values = [
			_first_list_item(value)
			for value in benchmark_df[benchmark_product_column].tolist()
		]
		training_product_first_values = [
			_first_list_item(value)
			for value in training_df[training_product_column].tolist()
		]
		benchmark_product_full_values = [
			_full_list_signature(value)
			for value in benchmark_df[benchmark_product_column].tolist()
		]
		training_product_full_values = [
			_full_list_signature(value)
			for value in training_df[training_product_column].tolist()
		]
	emit_progress("Loaded and normalized benchmark and training columns")

	benchmark_pair_first_values = [
		(
			sequence_value,
			substrate_value,
		)
		if sequence_value is not None and substrate_value is not None
		else None
		for sequence_value, substrate_value in zip(
			benchmark_sequence_values, benchmark_substrate_first_values
		)
	]
	training_pair_first_values = [
		(
			sequence_value,
			substrate_value,
		)
		if sequence_value is not None and substrate_value is not None
		else None
		for sequence_value, substrate_value in zip(
			training_sequence_values, training_substrate_first_values
		)
	]
	benchmark_pair_full_values = [
		(
			sequence_value,
			substrate_value,
		)
		if sequence_value is not None and substrate_value is not None
		else None
		for sequence_value, substrate_value in zip(
			benchmark_sequence_values, benchmark_substrate_full_values
		)
	]
	training_pair_full_values = [
		(
			sequence_value,
			substrate_value,
		)
		if sequence_value is not None and substrate_value is not None
		else None
		for sequence_value, substrate_value in zip(
			training_sequence_values, training_substrate_full_values
		)
	]

	report_rows: list[dict[str, Any]] = [
		_overlap_summary_row(
			feature="sequence",
			representation="exact",
			analysis_scope="row",
			benchmark_total_rows=benchmark_rows_total,
			training_total_rows=training_rows_total,
			benchmark_values=benchmark_sequence_values,
			training_values=training_sequence_values,
		),
		_overlap_summary_row(
			feature="ec_number",
			representation="exact",
			analysis_scope="row",
			benchmark_total_rows=benchmark_rows_total,
			training_total_rows=training_rows_total,
			benchmark_values=benchmark_ec_values,
			training_values=training_ec_values,
		),
		_overlap_summary_row(
			feature="substrate",
			representation="first_item",
			analysis_scope="row",
			benchmark_total_rows=benchmark_rows_total,
			training_total_rows=training_rows_total,
			benchmark_values=benchmark_substrate_first_values,
			training_values=training_substrate_first_values,
		),
		_overlap_summary_row(
			feature="substrate",
			representation="full_list",
			analysis_scope="row",
			benchmark_total_rows=benchmark_rows_total,
			training_total_rows=training_rows_total,
			benchmark_values=benchmark_substrate_full_values,
			training_values=training_substrate_full_values,
		),
		*([
			_overlap_summary_row(
				feature="product",
				representation="first_item",
				analysis_scope="row",
				benchmark_total_rows=benchmark_rows_total,
				training_total_rows=training_rows_total,
				benchmark_values=benchmark_product_first_values,
				training_values=training_product_first_values,
			),
			_overlap_summary_row(
				feature="product",
				representation="full_list",
				analysis_scope="row",
				benchmark_total_rows=benchmark_rows_total,
				training_total_rows=training_rows_total,
				benchmark_values=benchmark_product_full_values,
				training_values=training_product_full_values,
			),
		] if products_available else []),
		_overlap_summary_row(
			feature="substrate_sequence_pair",
			representation="first_item",
			analysis_scope="row",
			benchmark_total_rows=benchmark_rows_total,
			training_total_rows=training_rows_total,
			benchmark_values=benchmark_pair_first_values,
			training_values=training_pair_first_values,
		),
		_overlap_summary_row(
			feature="substrate_sequence_pair",
			representation="full_list",
			analysis_scope="row",
			benchmark_total_rows=benchmark_rows_total,
			training_total_rows=training_rows_total,
			benchmark_values=benchmark_pair_full_values,
			training_values=training_pair_full_values,
		),
	]
	emit_progress("Computed exact-overlap categories")

	benchmark_similarity_sequences = list(
		dict.fromkeys(
			sequence for sequence in benchmark_sequence_values if sequence is not None
		)
	)
	training_similarity_sequences = list(
		dict.fromkeys(
			sequence for sequence in training_sequence_values if sequence is not None
		)
	)
	benchmark_sequence_scores = _best_match_sequence_scores(
		benchmark_similarity_sequences,
		training_similarity_sequences,
	)
	emit_progress("Computed best-match sequence similarity scores")

	for threshold in sequence_similarity_thresholds:
		if not 0 <= threshold <= 1:
			raise ValueError(
				"sequence_similarity_thresholds must contain values between 0 and 1. "
				f"Received {threshold!r}."
			)

		report_rows.append(
			_overlap_summary_row(
				feature="sequence_similarity",
				representation="best_match",
				analysis_scope="unique_sequence",
				benchmark_total_rows=len(benchmark_similarity_sequences),
				training_total_rows=len(training_similarity_sequences),
				benchmark_values=benchmark_similarity_sequences,
				training_values=training_similarity_sequences,
				threshold=threshold,
				best_match_scores=benchmark_sequence_scores,
			)
		)
		emit_progress(f"Added sequence-similarity threshold row for {threshold:.2f}")

	result_df = pd.DataFrame(report_rows)
	result_df = result_df.sort_values(
		by=["comparison_type", "feature", "representation", "similarity_threshold"],
		kind="stable",
		ignore_index=True,
	)

	if output_filename is not None:
		filename = str(output_filename).strip()
		if not filename:
			raise ValueError("output_filename must not be empty when provided.")

		output_path = DATA_DIR / "leakage" / Path(filename).name
		if output_path.suffix == "":
			output_path = output_path.with_suffix(".csv")
		output_path.parent.mkdir(parents=True, exist_ok=True)
		result_df.to_csv(output_path, index=False)
		emit_progress(f"Wrote report to {output_path}")

	emit_progress("Report construction complete")
	return result_df
