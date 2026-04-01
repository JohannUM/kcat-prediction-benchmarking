import sys
import zipfile
import pandas as pd
from kcatbench.util import MODELS_DIR, DATA_DIR, DEVICE, ensure_data_subfolder, force_torch_load_device, wget_download, work_in_dir
from kcatbench.model_wrapper.base_model import BaseModel

TURNUP_CODE_DIR = MODELS_DIR / "TurNuP"
TURNUP_DATA_DIR = DATA_DIR / "TurNuP"

if str(TURNUP_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(TURNUP_CODE_DIR / "code"))


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

        target_cwd = TURNUP_CODE_DIR / "code"

        with work_in_dir(target_cwd), force_torch_load_device(DEVICE):
            from kcat_prediction import kcat_predicton

            subs_list, prods_list, enz_list, valid_indices = self._prepare_turnup_input(input_data)

            print(subs_list)
            print(prods_list)
            print(enz_list)
            
            result = kcat_predicton(substrates = subs_list, products = prods_list, enzymes = enz_list)

            predictions = result["kcat [s^(-1)]"].to_list()

            output['turnup_kcat'] = pd.NA
            output.loc[valid_indices, 'turnup_kcat'] = predictions

        return output
        
    def _prepare_turnup_input(self, df: pd.DataFrame):
        
        valid_indices = []
        substrates_list = []
        products_list = []
        enzymes_list = []

        for idx, row in df.iterrows():
            subs = row.get('substrates')
            prods = row.get('products')
            seq = row.get('sequence')

            if pd.isna(seq) or not isinstance(seq, str) or not seq.strip():
                continue

            if not isinstance(subs, list) or len(subs) == 0:
                continue

            if not isinstance(prods, list) or len(prods) == 0:
                continue

            joined_subs = ";".join([str(s).strip() for s in subs if pd.notna(s) and str(s).strip()])
            joined_prods = ";".join([str(p).strip() for p in prods if pd.notna(p) and str(p).strip()])

            if not joined_subs or not joined_prods:
                continue

            valid_indices.append(idx)
            substrates_list.append(joined_subs)
            products_list.append(joined_prods)
            enzymes_list.append(seq)

        return substrates_list, products_list, enzymes_list, valid_indices
