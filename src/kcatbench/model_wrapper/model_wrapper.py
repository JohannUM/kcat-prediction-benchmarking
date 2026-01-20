import pandas as pd
import subprocess
import tempfile
from pathlib import Path

from kcatbench.model_wrapper.base import BaseModel

ENVIRONMENT_NAMES: dict[str, str] = {
    "dlkcat": "dlkcat_env"
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

    def predict(self, input_data:pd.DataFrame) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.tsv"
            output_path = tmpdir / "output.tsv"

            input_data.to_csv(input_path, sep="\t", index=False)

            cmd = [
                "conda", "run", "-n", self.env_name,
                "python", "-m", "kcatbench.model_worker.py", "predict"
                "--model", self.model_name, "--input", str(input_path), "--output", str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if(result.returncode != 0):
                raise RuntimeError(f"Model {self.model_name} failed:\nOUTPUT:\n{result.stdout}\n\nERROR:\n{result.stderr}")

            return pd.read_csv(output_path, sep="\t")
        
        return super().predict(input_data)