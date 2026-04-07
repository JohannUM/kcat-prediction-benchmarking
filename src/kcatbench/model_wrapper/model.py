import os
import pandas as pd
import subprocess
import tempfile
from pathlib import Path

from kcatbench.model_wrapper.base_model import BaseModel
from kcatbench.util import read_csv_with_schema

ENVIRONMENT_NAMES: dict[str, str] = {
    "dlkcat": "dlkcat_env",
    "catapro": "catapro_env",
    "catpred": "catpred_env",
    "turnup": "turnup_env",
    "unikp": "unikp_env"
}

class Model(BaseModel):

    def __init__(self, model_name:str, env_name:str=None):
        if(model_name not in ENVIRONMENT_NAMES.keys()):
            raise ValueError(f"{model_name} is not valid\nValid names are: {ENVIRONMENT_NAMES.keys()}")

        if(env_name != None):
            self.env_name = env_name
        else:
            self.env_name = ENVIRONMENT_NAMES[model_name]
        self.model_name = model_name 
        super().__init__()

    def _get_isolated_env(self) -> dict[str, str]:
        """Creates a clean environment dictionary for the subprocess."""
        env = os.environ.copy()
        
        keys_to_remove = [
            "PYTHONPATH",        
            "PYTHONHOME", 
            "MPLBACKEND",
        ]
        
        for key in keys_to_remove:
            env.pop(key, None)
            
        return env

    def predict(self, input_data:pd.DataFrame) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.csv"
            output_path = tmpdir / "output.csv"

            input_data.to_csv(input_path, index=False)

            cmd = [
                "conda", "run", "-n", self.env_name, "--no-capture-output",
                "python", "-m", "kcatbench.model_wrapper.model_worker", "predict",
                "--model", self.model_name, "--input", str(input_path), "--output", str(output_path)
            ]

            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                env=self._get_isolated_env()
            )

            if(result.returncode != 0):
                raise RuntimeError(f"Model {self.model_name} failed:\nOUTPUT:\n{result.stdout}\n\nERROR:\n{result.stderr}")

            return read_csv_with_schema(output_path)