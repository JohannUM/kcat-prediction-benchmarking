from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import pandas as pd

class BaseModel(ABC):

    name: str = "base_model"

    @abstractmethod
    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on the input data.

        Args:
            input_data (pd.DataFrame): Input data for prediction.

        Returns:
            pd.DataFrame: DataFrame containing predictions.
        """
        raise NotImplementedError
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        products_required: bool = False,
        multiple_smiles: bool = False,
        extra_key_builder: Optional[
            Callable[[Any, str, Union[str, list[str]], Optional[Union[str, list[str]]]], Any]
        ] = None,
    ) -> dict[str, list]:
        """
        Prepares and validates input data from a DataFrame for model prediction.

        Args:
            df (pd.DataFrame): The input DataFrame containing 'sequence' and 'substrates' 
                columns, and optionally a 'products' column.
            products_required (bool, optional): If True, rows without valid product data 
                are skipped, and a 'products' list is added to the returned dictionary. 
                Defaults to False.
            multiple_smiles (bool, optional): If True, preserves multiple SMILES strings
                as a cleaned list[str]. If False, extracts only the first SMILES string
                from the list. Defaults to False.
            extra_key_builder (Callable, optional): Optional callback to append model-specific
                values to the deduplication key. Called with (row, sequence, substrates, products).

        Returns:
            dict[str, list]: A dictionary containing deduplicated cleaned data. Guaranteed
            to contain the keys 'valid_indices', 'index_groups', 'sequence', and 'substrates'.
            Each entry in 'valid_indices' is the representative source index for one unique
            model input. The same-position entry in 'index_groups' contains all original row
            indices that map to that unique model input.
            If `multiple_smiles` is True, substrate/product entries are list[str].
            If `multiple_smiles` is False, substrate/product entries are str.
            Will also contain 'products' if `products_required` is True.
        """

        cleaned_data: dict[str, list] = {
            "valid_indices": [],
            "index_groups": [],
            "substrates": [],
            "sequence": []
        }
        
        if products_required:
            cleaned_data["products"] = []

        def process_smiles(items):
            if not isinstance(items, list) or not items:
                return None
            
            if multiple_smiles:
                valid_items = [str(i).strip() for i in items if pd.notna(i) and str(i).strip()]
                return valid_items if valid_items else None
            else:
                first = items[0]
                return str(first).strip() if pd.notna(first) and str(first).strip() else None

        def make_hashable(value: Any) -> Any:
            if isinstance(value, list):
                return tuple(make_hashable(item) for item in value)
            if isinstance(value, tuple):
                return tuple(make_hashable(item) for item in value)
            return value

        dedup_lookup: dict[tuple[Any, ...], int] = {}

        for row in df.itertuples(index=True):
            seq = getattr(row, 'sequence', None)
            
            if not isinstance(seq, str) or not seq.strip():
                continue

            sequence_value = seq.strip()

            subs = process_smiles(getattr(row, 'substrates', None))
            if not subs:
                continue

            prods = None
            if products_required:
                prods = process_smiles(getattr(row, 'products', None))
                if not prods:
                    continue

            dedup_key_parts: list[Any] = [sequence_value, make_hashable(subs)]
            if products_required:
                dedup_key_parts.append(make_hashable(prods))
            if extra_key_builder is not None:
                extra_key = extra_key_builder(row, sequence_value, subs, prods)
                dedup_key_parts.append(make_hashable(extra_key))

            dedup_key = tuple(dedup_key_parts)
            existing_position = dedup_lookup.get(dedup_key)

            if existing_position is None:
                unique_position = len(cleaned_data["valid_indices"])
                dedup_lookup[dedup_key] = unique_position

                cleaned_data["valid_indices"].append(row.Index)
                cleaned_data["index_groups"].append([row.Index])
                cleaned_data["sequence"].append(sequence_value)
                cleaned_data["substrates"].append(subs)
                if products_required:
                    cleaned_data["products"].append(prods)
            else:
                cleaned_data["index_groups"][existing_position].append(row.Index)

        return cleaned_data

    def _expand_predictions(
        self,
        clean_data: dict[str, list],
        unique_predictions: Sequence[Any],
    ) -> tuple[list[Any], list[Any]]:
        """
        Expands one prediction per prepared row back to all matching source rows.

        Args:
            clean_data (dict[str, list]): Output dictionary from _prepare_data.
            unique_predictions (Sequence[Any]): Prediction values aligned to prepared rows.

        Returns:
            tuple[list[Any], list[Any]]: Tuple of (row_indices, prediction_values) that can
            be used directly in DataFrame.loc assignment.
        """
        valid_indices = clean_data.get("valid_indices", [])
        index_groups = clean_data.get("index_groups")

        if len(unique_predictions) != len(valid_indices):
            raise ValueError(
                "Prediction count mismatch: "
                f"{len(unique_predictions)} predictions for {len(valid_indices)} prepared rows."
            )

        if index_groups is None:
            return list(valid_indices), list(unique_predictions)

        if len(index_groups) != len(valid_indices):
            raise ValueError(
                "Prepared data index mapping mismatch: "
                f"{len(index_groups)} index_groups for {len(valid_indices)} prepared rows."
            )

        expanded_indices: list[Any] = []
        expanded_predictions: list[Any] = []

        for prediction, group_indices in zip(unique_predictions, index_groups):
            if not isinstance(group_indices, list) or not group_indices:
                raise ValueError("Each prepared row must have a non-empty list in index_groups.")

            expanded_indices.extend(group_indices)
            expanded_predictions.extend([prediction] * len(group_indices))

        return expanded_indices, expanded_predictions