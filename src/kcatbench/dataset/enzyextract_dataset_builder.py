import json
import ast
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from kcatbench.util import DATA_DIR, ensure_data_subfolder


_RAW_FILENAME = "enzy_extract_complete.parquet"
_PROCESSED_BASENAME = "enzy_extract_processed"


def _standardize_smiles(value: object) -> str:
    original = "" if value is None else str(value)
    text = original.strip()
    if not text:
        return original

    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return original

        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return original


def _to_singleton_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [str(value)]


def _is_empty_mutant(value: object) -> bool:
    if isinstance(value, list):
        return len(value) == 0

    if isinstance(value, tuple):
        return len(value) == 0

    if hasattr(value, "size") and hasattr(value, "shape"):
        return value.size == 0

    if value is None:
        return True

    try:
        missing = pd.isna(value)
    except Exception:
        missing = False

    if isinstance(missing, bool) and missing:
        return True

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return True

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return len(parsed) == 0
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue

        return False

    return False

def ee_download_db():
    url = "https://github.com/ChemBioHTP/EnzyExtract/raw/refs/heads/main/EnzyExtractDB/EnzyExtractDB_176463.parquet"
    target_dir = DATA_DIR / "enzyextract"
    ensure_data_subfolder(target_dir)
    local_filename = target_dir / _RAW_FILENAME

    response = requests.get(url)
    response.raise_for_status()

    with open(local_filename, "wb") as file:
        file.write(response.content)

    return local_filename


def ee_process_db(path: Path = (DATA_DIR / "enzyextract")):
    target_dir = Path(path)
    ensure_data_subfolder(target_dir)

    source_path = target_dir / _RAW_FILENAME
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing raw EnzyExtract parquet at {source_path}. Run ee_download_db() first."
        )

    df = pd.read_parquet(source_path)

    required_columns = {"kcat_value", "smiles", "sequence", "clean_mutant"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in EnzyExtract data: {missing}")

    df = df.dropna(subset=["kcat_value", "smiles", "sequence"])

    clean_mutant = df["clean_mutant"]
    mutant_mask = clean_mutant.apply(_is_empty_mutant)
    df = df[mutant_mask].copy()
    df = df.reset_index(drop=True)

    df = df.rename(columns={"smiles": "substrates", "kcat_value": "experimental_kcat"})
    df["substrates"] = df["substrates"].apply(_standardize_smiles)
    df["substrates"] = df["substrates"].apply(_to_singleton_list)

    csv_path = target_dir / f"{_PROCESSED_BASENAME}.csv"
    pkl_path = target_dir / f"{_PROCESSED_BASENAME}.pkl"

    df.to_pickle(pkl_path)

    csv_df = df.copy()
    csv_df["substrates"] = csv_df["substrates"].apply(json.dumps)
    csv_df.to_csv(csv_path, index=False)

    return csv_path, pkl_path


def ee_build_db():
    ee_download_db()
    return ee_process_db()
