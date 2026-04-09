import sys
import zipfile
import logging
import pandas as pd
from kcatbench.util import MODELS_DIR, DATA_DIR, DEVICE, ensure_data_subfolder, force_torch_load_device, wget_download, work_in_dir
from kcatbench.model_wrapper.base_model import BaseModel

TURNUP_CODE_DIR = MODELS_DIR / "TurNuP"
TURNUP_DATA_DIR = DATA_DIR / "TurNuP"

if str(TURNUP_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(TURNUP_CODE_DIR / "code"))


LOGGER = logging.getLogger(__name__)


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
            print(f"Created symlink: {expected_data_link} -> {(TURNUP_DATA_DIR / 'data')}")
        
        try:
            archive_path.unlink()
            success_marker.touch()
        except OSError as e:
            print(f"Warning: Could not remove archive file: {e}")

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        
        output = input_data.copy()
        output['turnup_kcat'] = pd.NA

        target_cwd = TURNUP_CODE_DIR / "code"

        with work_in_dir(target_cwd), force_torch_load_device(DEVICE):
            from kcat_prediction import kcat_predicton

            clean_data = self._prepare_data(input_data, products_required=True, multiple_smiles=True)
            if not clean_data["valid_indices"]:
                LOGGER.warning("TurNuP found no valid rows after preprocessing.")
                return output

            substrates, products, enzymes, valid_indices = self._prepare_turnup_input(clean_data)
            
            result = kcat_predicton(substrates=substrates, products=products, enzymes=enzymes)

            predictions = result["kcat [s^(-1)]"].to_list()

            assign_count = min(len(valid_indices), len(predictions))
            output.loc[valid_indices[:assign_count], 'turnup_kcat'] = predictions[:assign_count]
            if assign_count != len(valid_indices) or len(predictions) != len(valid_indices):
                LOGGER.warning(
                    "TurNuP prediction count mismatch: expected=%s predicted=%s assigned=%s",
                    len(valid_indices),
                    len(predictions),
                    assign_count,
                )

        return output
        
    def _prepare_turnup_input(self, clean_data: dict[str, list]):
        substrates = [";".join(items) for items in clean_data["substrates"]]
        products = [";".join(items) for items in clean_data["products"]]
        enzymes = clean_data["sequence"]
        valid_indices = clean_data["valid_indices"]

        return substrates, products, enzymes, valid_indices
