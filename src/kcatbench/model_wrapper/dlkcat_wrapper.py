from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
DLKCAT_CODE_DIR = (ROOT_DIR / "models" / "DLKcat" / "DeeplearningApproach" / "Code" / "example")
DLKCAT_DATA_DIR = (ROOT_DIR / "data" / "DLKcat") 

if str(DLKCAT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(DLKCAT_CODE_DIR))


from .base import BaseModel
# from prediction_for_input import split_sequence
import pandas as pd
import zipfile


class DLKcatWrapper(BaseModel):
    name = "DLKcat"

    def __init__(self):
        super().__init__()
        self._prepare_resources()

    def _prepare_resources(self):
        input_zip_file = ROOT_DIR / "models" / "DLKcat" / "DeeplearningApproach" / "Data" / "input.zip"
        
        if not input_zip_file.exists():
            raise FileNotFoundError(f"DLKcat resource zip not found at: {input_zip_file}")

        with zipfile.ZipFile(input_zip_file, 'r') as zip_ref:
            zip_ref.extractall(DLKCAT_DATA_DIR)

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        # ignore for now
        return pd.DataFrame()


