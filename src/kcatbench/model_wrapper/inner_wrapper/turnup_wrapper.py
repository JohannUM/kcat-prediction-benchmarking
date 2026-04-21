import sys
import zipfile
import logging
import io
import re
from contextlib import redirect_stderr, redirect_stdout
import pandas as pd
from kcatbench.util import MODELS_DIR, DATA_DIR, DEVICE, ensure_data_subfolder, force_torch_load_device, wget_download, work_in_dir
from kcatbench.model_wrapper.base_model import BaseModel
from kcatbench.model_wrapper.wrapper_progress import progress_completed, progress_started

TURNUP_CODE_DIR = MODELS_DIR / "TurNuP"
TURNUP_DATA_DIR = DATA_DIR / "TurNuP"

if str(TURNUP_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(TURNUP_CODE_DIR / "code"))


LOGGER = logging.getLogger(__name__)


_EMPTY_RDKit_ERROR_LINE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s+ERROR:$")
_EMPTY_RDKit_TIMESTAMP_LINE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]$")


class TurNuPWrapper(BaseModel):
    name = "TurNuP"

    def __init__(self):
        super().__init__()
        self._prepare_resources()

    def _prepare_resources(self):
        ensure_data_subfolder(TURNUP_DATA_DIR)

        success_marker = TURNUP_DATA_DIR / ".setup_complete"
        if success_marker.exists():
            return
        
        archive_path = TURNUP_DATA_DIR / "data.zip"

        result = wget_download(url="https://zenodo.org/records/8038678/files/data.zip?download=1", output_path=archive_path)
        if not result['success']:
            raise RuntimeError(result['message'])
        
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(TURNUP_DATA_DIR)

        expected_data_link = TURNUP_CODE_DIR / "data"

        if not expected_data_link.exists():
            expected_data_link.symlink_to((TURNUP_DATA_DIR / "data"), target_is_directory=True)
            LOGGER.info("Created symlink: %s -> %s", expected_data_link, (TURNUP_DATA_DIR / "data"))
        
        try:
            archive_path.unlink()
            success_marker.touch()
        except OSError as e:
            LOGGER.warning("Could not remove TurNuP archive file: %s", e)

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        progress_started(LOGGER, "turnup.predict", "TurNuP prediction started rows=%s.", len(input_data.index))
        
        output = input_data.copy()
        output['turnup_kcat'] = pd.NA

        target_cwd = TURNUP_CODE_DIR / "code"

        with work_in_dir(target_cwd), force_torch_load_device(DEVICE):
            from kcat_prediction import kcat_predicton

            progress_started(LOGGER, "turnup.validation", "TurNuP input validation started.")
            clean_data = self._prepare_data(input_data, products_required=True, multiple_smiles=True)
            if not clean_data["valid_indices"]:
                progress_completed(LOGGER, "turnup.validation", "TurNuP input validation completed valid_rows=0.")
                LOGGER.warning("TurNuP found no valid rows after preprocessing.")
                return output

            progress_completed(
                LOGGER,
                "turnup.validation",
                "TurNuP input validation completed valid_rows=%s.",
                len(clean_data["valid_indices"]),
            )

            substrates, products, enzymes, valid_indices = self._prepare_turnup_input(clean_data)
            progress_started(
                LOGGER,
                "turnup.inference",
                "TurNuP inference started valid_rows=%s.",
                len(valid_indices),
            )
            predictions = []
            failed_rows = 0

            for row_pos, row_index in enumerate(valid_indices):
                prediction = self._predict_single_reaction(
                    kcat_predicton=kcat_predicton,
                    substrate=substrates[row_pos],
                    product=products[row_pos],
                    enzyme=enzymes[row_pos],
                    row_index=row_index,
                )
                if pd.isna(prediction):
                    failed_rows += 1
                predictions.append(prediction)

            success_rows = len(predictions) - failed_rows
            progress_completed(
                LOGGER,
                "turnup.inference",
                "TurNuP inference completed attempted_rows=%s successful_rows=%s failed_rows=%s.",
                len(predictions),
                success_rows,
                failed_rows,
            )

            assign_indices, assign_values = self._expand_predictions(clean_data, predictions)
            if assign_indices:
                output.loc[assign_indices, 'turnup_kcat'] = assign_values

            progress_completed(
                LOGGER,
                "turnup.predict",
                "TurNuP predictions assigned rows=%s.",
                len(assign_indices),
            )

        return output

    def _predict_single_reaction(self, kcat_predicton, substrate: str, product: str, enzyme: str, row_index: int):
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                result = kcat_predicton(
                    substrates=[substrate],
                    products=[product],
                    enzymes=[enzyme],
                )

            prediction = self._extract_prediction(result)
            return prediction
        except Exception as exc:
            stderr_lines = self._filter_stderr_lines(stderr_buffer.getvalue())
            message = (
                "TurNuP row failed index=%s substrate=%s product=%s sequence_len=%s error=%s: %s"
            )

            if stderr_lines:
                LOGGER.error(
                    message + " stderr=%s",
                    row_index,
                    self._preview_for_log(substrate),
                    self._preview_for_log(product),
                    len(enzyme),
                    type(exc).__name__,
                    exc,
                    " | ".join(stderr_lines[:3]),
                )
            else:
                LOGGER.error(
                    message,
                    row_index,
                    self._preview_for_log(substrate),
                    self._preview_for_log(product),
                    len(enzyme),
                    type(exc).__name__,
                    exc,
                )
            return pd.NA

    def _extract_prediction(self, result: pd.DataFrame):
        prediction_column = "kcat [s^(-1)]"
        if prediction_column not in result.columns or result.empty:
            return pd.NA

        value = result.iloc[0][prediction_column]
        return value if pd.notna(value) else pd.NA

    def _filter_stderr_lines(self, stderr_text: str) -> list[str]:
        filtered_lines = []
        for line in stderr_text.splitlines():
            stripped_line = line.strip()
            if not stripped_line:
                continue
            if _EMPTY_RDKit_ERROR_LINE.match(stripped_line):
                continue
            if _EMPTY_RDKit_TIMESTAMP_LINE.match(stripped_line):
                continue
            filtered_lines.append(stripped_line)
        return filtered_lines

    def _preview_for_log(self, value: str, max_len: int = 100) -> str:
        clean_value = value.strip()
        if len(clean_value) <= max_len:
            return clean_value
        return clean_value[: max_len - 3] + "..."
        
    def _prepare_turnup_input(self, clean_data: dict[str, list]):
        substrates = [";".join(items) for items in clean_data["substrates"]]
        products = [";".join(items) for items in clean_data["products"]]
        enzymes = clean_data["sequence"]
        valid_indices = clean_data["valid_indices"]

        return substrates, products, enzymes, valid_indices
