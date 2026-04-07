import os
import subprocess
import json
import ast
from pathlib import Path
from contextlib import contextmanager
from typing import Union

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT_DIR / "config.json"

_config = {}
if CONFIG_FILE.is_file():
    with open(CONFIG_FILE, "r") as f:
        _config = json.load(f)

def _resolve_path(config_key: str, default_folder_name: str) -> Path:
    """
    Looks up a path in the config. 
    If missing, falls back to ROOT_DIR / default_folder_name.
    Handles both relative and absolute paths in the JSON.
    """
    path_str = _config.get(config_key)
    
    if path_str:
        p = Path(path_str)
        return p if p.is_absolute() else ROOT_DIR / p
    
    return ROOT_DIR / default_folder_name

MODELS_DIR = ROOT_DIR / "models"
DATA_DIR   = _resolve_path("data_dir", "data")
RESULT_DIR = _resolve_path("results_dir", "results")

DEVICE = _config.get("device", "cuda:0")


def _parse_list_str_cell(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]

    if pd.isna(value):
        return []

    if not isinstance(value, str):
        raise ValueError(f"Expected a string-encoded list, got {type(value).__name__}.")

    text = value.strip()
    if not text:
        return []

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("Could not parse list value.") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"Expected list value, got {type(parsed).__name__}.")

    return [str(item) for item in parsed]


def _parse_list_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            continue

        parsed_values: list[list[str]] = []
        for row_index, value in df[column].items():
            try:
                parsed_values.append(_parse_list_str_cell(value))
            except ValueError as exc:
                raise ValueError(
                    f"Failed to parse '{column}' at row {row_index}: {value!r}."
                ) from exc

        df[column] = parsed_values

    return df


def read_csv_with_schema(csv_path: Union[Path, str]) -> pd.DataFrame:
    """
    Read a CSV file and parse schema-specific list columns.

    The columns 'substrates' and 'products' are parsed as list[str] values.
    Other columns are read with pandas defaults.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return _parse_list_columns(df, ("substrates", "products"))


def ensure_data_subfolder(target_dir: Path) -> None:
    """
    Validates that the base DATA_DIR exists and creates the target 
    subfolder if it is missing.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Base data directory not found at: {DATA_DIR}. "
            "Please ensure the path is configured correctly in config.json."
        )
    target_dir.mkdir(parents=True, exist_ok=True)


def wget_download(url, output_path, retries=3, timeout=90, show_progress=True):
    cmd = ['wget', url]
    cmd.extend(['-O', str(output_path)])
    cmd.extend(['--tries', str(retries)])
    cmd.extend(['--timeout', str(timeout)])
    if not show_progress:
        cmd.append('--quiet')

    try:
        subprocess.run(cmd, check=True)
        return {"success": True, "message": "Download completed successfully."}
    except subprocess.CalledProcessError as e:
        return {"success": False, "message": f"Download failed: {e}"}


def extract_tar_gz(archive_path, extract_to_dir):
    cmd = ['tar', '-xzf', str(archive_path), '-C', str(extract_to_dir)]
    try:
        subprocess.run(cmd, check=True)
        return {"success": True, "message": "Extraction completed successfully."}
    except subprocess.CalledProcessError as e:
        return {"success": False, "message": f"Extraction failed: {e}"}
    

@contextmanager
def work_in_dir(path):
    origin = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

@contextmanager
def force_torch_load_device(target_device: str):
    """
    Temporarily hijacks torch.load to force a specific map_location.
    """
    import torch 
    
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        kwargs['map_location'] = target_device
        return original_load(*args, **kwargs)
        
    torch.load = patched_load
    
    try:
        yield
    finally:
        torch.load = original_load