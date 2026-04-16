import os
import subprocess
import json
import ast
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import contextmanager
from typing import Optional, Union

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


def _resolve_optional_device(config_key: str) -> Optional[str]:
    value = _config.get(config_key)
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    clean_value = value.strip()
    return clean_value if clean_value else None


SECOND_DEVICE = _resolve_optional_device("second_device")


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


def gdrive_download_file_from_folder(folder_url, expected_filename, output_path, quiet=True):
    """
    Download a specific file from a public Google Drive folder URL.

    Returns a dict with the same shape as other utility helpers:
    {"success": bool, "message": str}
    """
    try:
        import gdown
    except ImportError:
        return {
            "success": False,
            "message": (
                "Missing dependency 'gdown'. Install it in the active environment "
                "to enable Google Drive checkpoint download."
            ),
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="kcatbench_gdrive_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        try:
            downloaded_files = gdown.download_folder(
                url=folder_url,
                output=str(tmp_root),
                quiet=quiet,
                remaining_ok=True,
            )
        except Exception as exc:
            return {
                "success": False,
                "message": f"Google Drive folder download failed: {exc}",
            }

        if not downloaded_files:
            return {
                "success": False,
                "message": "Google Drive folder download returned no files.",
            }

        candidates = [
            Path(file_path)
            for file_path in downloaded_files
            if Path(file_path).name == expected_filename
        ]
        if not candidates:
            candidates = list(tmp_root.rglob(expected_filename))

        if not candidates:
            return {
                "success": False,
                "message": (
                    f"Expected file '{expected_filename}' was not found in downloaded "
                    "Google Drive folder contents."
                ),
            }

        source_path = candidates[0]
        if not source_path.exists() or source_path.stat().st_size == 0:
            return {
                "success": False,
                "message": (
                    f"Downloaded file '{source_path}' is missing or empty."
                ),
            }

        try:
            shutil.move(str(source_path), str(output_path))
        except OSError as exc:
            return {
                "success": False,
                "message": f"Failed moving downloaded checkpoint to destination: {exc}",
            }

    return {
        "success": True,
        "message": f"Downloaded '{expected_filename}' to '{output_path}'.",
    }
    

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