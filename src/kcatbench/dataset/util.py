from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd


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
