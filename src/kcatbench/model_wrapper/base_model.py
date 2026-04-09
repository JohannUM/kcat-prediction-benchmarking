from abc import ABC, abstractmethod
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
    
    def _prepare_data(self, df: pd.DataFrame, products_required: bool = False, multiple_smiles: bool = False) -> dict[str, list]:
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

        Returns:
            dict[str, list]: A dictionary containing lists of the cleaned data. Guaranteed
            to contain the keys 'valid_indices', 'sequence', and 'substrates'.
            If `multiple_smiles` is True, substrate/product entries are list[str].
            If `multiple_smiles` is False, substrate/product entries are str.
            Will also contain 'products' if `products_required` is True.
        """
        
        cleaned_data: dict[str, list] = {
            "valid_indices": [],
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

        for row in df.itertuples(index=True):
            seq = getattr(row, 'sequence', None)
            
            if not isinstance(seq, str) or not seq.strip():
                continue

            subs = process_smiles(getattr(row, 'substrates', None))
            if not subs:
                continue

            if products_required:
                prods = process_smiles(getattr(row, 'products', None))
                if not prods:
                    continue
                cleaned_data["products"].append(prods)

            cleaned_data["valid_indices"].append(row.Index)
            cleaned_data["sequence"].append(seq.strip())
            cleaned_data["substrates"].append(subs)

        return cleaned_data