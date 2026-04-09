import os
import logging
import subprocess
from kcatbench.util import MODELS_DIR, DATA_DIR, DEVICE, extract_tar_gz, wget_download, work_in_dir, ensure_data_subfolder

CATPRED_CODE_DIR = MODELS_DIR / "CatPred"
CATPRED_DATA_DIR = DATA_DIR / "CatPred"

from kcatbench.model_wrapper.base_model import BaseModel
import pandas as pd
import numpy as np
from rdkit import Chem


LOGGER = logging.getLogger(__name__)

class CatPredWrapper(BaseModel):
    name = "CatPred"

    def __init__(self):
        super().__init__()
        self._prepare_resources()

    def _prepare_resources(self):
        ensure_data_subfolder(CATPRED_DATA_DIR)

        success_marker = CATPRED_DATA_DIR / ".setup_complete"
        if success_marker.exists():
            return

        archive_path = CATPRED_DATA_DIR / "capsule_data_update.tar.gz"

        result = wget_download(url="https://catpred.s3.us-east-1.amazonaws.com/capsule_data_update.tar.gz", output_path=archive_path)
        if not result['success']:
            raise RuntimeError(result['message'])
        
        result = extract_tar_gz(archive_path, CATPRED_DATA_DIR)
        if not result['success']:
            archive_path.unlink(missing_ok=True)
            raise RuntimeError(result['message'])
        
        try:
            archive_path.unlink()
            success_marker.touch()
        except OSError as e:
            print(f"Warning: Could not remove archive file: {e}")

    def predict(self, input_data: pd.DataFrame):
        output = input_data.copy()
        output['catpred_kcat'] = pd.NA

        with work_in_dir(CATPRED_CODE_DIR):
            outfile, clean_data = self._create_csv_sh("kcat", input_data, str(CATPRED_DATA_DIR / "data" / "pretrained" / "production" / "kcat"))
            if outfile is None or not clean_data["valid_indices"]:
                LOGGER.warning("CatPred found no valid rows after validation. Returning NA predictions.")
                return output

            run_result = self._run_prediction_script()
            if run_result.returncode != 0:
                LOGGER.error(
                    "CatPred predict.sh failed with exit code %s. stdout=%s stderr=%s",
                    run_result.returncode,
                    run_result.stdout,
                    run_result.stderr,
                )
                return output

            try:
                output_catpred = self._get_predictions("kcat", outfile)
            except Exception:
                LOGGER.exception("CatPred failed to parse prediction output file: %s", outfile)
                return output

        prediction_col = 'Prediction_(s^(-1))'
        if prediction_col not in output_catpred.columns:
            LOGGER.error("CatPred output is missing expected column: %s", prediction_col)
            return output

        predictions = output_catpred[prediction_col].tolist()
        target_indices = clean_data["valid_indices"]
        assign_count = min(len(target_indices), len(predictions))
        if assign_count == 0:
            LOGGER.warning("CatPred produced no assignable predictions after filtering.")
            return output

        output.loc[target_indices[:assign_count], 'catpred_kcat'] = predictions[:assign_count]
        if assign_count != len(target_indices) or len(predictions) != len(target_indices):
            LOGGER.warning(
                "CatPred prediction count mismatch: expected=%s predicted=%s assigned=%s",
                len(target_indices),
                len(predictions),
                assign_count,
            )

        return output

    def _empty_clean_data(self):
        return {
            "valid_indices": [],
            "substrates": [],
            "sequence": [],
        }

    def _run_prediction_script(self):
        env = os.environ.copy()
        env["PROTEIN_EMBED_USE_CPU"] = "0"
        return subprocess.run(
            ["bash", "./predict.sh"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def _canonicalize_smiles(self, smiles: str, parameter: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            return None

        if mol is None:
            return None

        try:
            canonical = Chem.MolToSmiles(mol)
        except Exception:
            return None

        if parameter == 'kcat' and '.' in canonical:
            canonical = '.'.join(sorted(canonical.split('.')))
        return canonical

    def _is_valid_sequence(self, sequence: str, valid_aas: set):
        return isinstance(sequence, str) and bool(sequence) and set(sequence).issubset(valid_aas)
    
    def _create_csv_sh(self, parameter, input_data:pd.DataFrame, checkpoint_dir):

        clean_data = self._prepare_data(input_data)
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        filtered_clean_data = self._empty_clean_data()
        smiles_list_new = []
        sequence_list_new = []
        invalid_smiles_indices = []
        invalid_sequence_indices = []

        for row_index, smi, seq in zip(
            clean_data["valid_indices"],
            clean_data["substrates"],
            clean_data["sequence"],
        ):
            canonical_smiles = self._canonicalize_smiles(smi, parameter)
            if canonical_smiles is None:
                invalid_smiles_indices.append(row_index)
                continue

            if not self._is_valid_sequence(seq, valid_aas):
                invalid_sequence_indices.append(row_index)
                continue

            filtered_clean_data["valid_indices"].append(row_index)
            filtered_clean_data["substrates"].append(canonical_smiles)
            filtered_clean_data["sequence"].append(seq)
            smiles_list_new.append(canonical_smiles)
            sequence_list_new.append(seq)

        if invalid_smiles_indices:
            LOGGER.warning("CatPred skipped invalid SMILES rows: %s", invalid_smiles_indices)
        if invalid_sequence_indices:
            LOGGER.warning("CatPred skipped invalid sequence rows: %s", invalid_sequence_indices)
        if not filtered_clean_data["valid_indices"]:
            LOGGER.warning("CatPred has no valid rows after row-level validation.")
            return None, filtered_clean_data

        input_file_new_path = str(CATPRED_DATA_DIR / "kcat_prediction_input.csv")
        df = pd.DataFrame()
        df['SMILES'] = smiles_list_new
        df['sequence'] = sequence_list_new
        df['pdbpath'] = [f"sequence_{i}" for i in range(len(df))]
        df.to_csv(input_file_new_path, index=False)

        gpu_id = DEVICE.split(":")[1] if ":" in DEVICE else "0"

        with open('predict.sh', 'w') as f:
            f.write(f'''
            TEST_FILE_PREFIX={input_file_new_path[:-4]}
            RECORDS_FILE=${{TEST_FILE_PREFIX}}.json
            CHECKPOINT_DIR={checkpoint_dir}
            
            python ./scripts/create_pdbrecords.py --data_file ${{TEST_FILE_PREFIX}}.csv --out_file ${{RECORDS_FILE}}
            python predict.py --test_path ${{TEST_FILE_PREFIX}}.csv --preds_path ${{TEST_FILE_PREFIX}}_output.csv --checkpoint_dir $CHECKPOINT_DIR --uncertainty_method mve --smiles_column SMILES --individual_ensemble_predictions --protein_records_path $RECORDS_FILE --gpu {gpu_id}
            ''')

        return input_file_new_path[:-4]+'_output.csv', filtered_clean_data
    
    def _get_predictions(self, parameter, outfile):
        """
        Process prediction results and add additional metrics.

        Args:
            parameter (str): The kinetics parameter that was predicted.
            outfile (str): Path to the output CSV file from the prediction.

        Returns:
            pandas.DataFrame: Processed predictions with additional metrics.
        """
        df = pd.read_csv(outfile)
        pred_col, pred_logcol, pred_sd_totcol, pred_sd_aleacol, pred_sd_epicol = [], [], [], [], []

        unit = 'mM'
        if parameter == 'kcat':
            target_col = 'log10kcat_max'
            unit = 's^(-1)'
        elif parameter == 'km':
            target_col = 'log10km_mean'
        else:
            target_col = 'log10ki_mean'

        unc_col = f'{target_col}_mve_uncal_var'

        for _, row in df.iterrows():
            model_cols = [col for col in row.index if col.startswith(target_col) and 'model_' in col]

            unc = row[unc_col]
            prediction = row[target_col]
            prediction_linear = np.power(10, prediction)

            model_outs = np.array([row[col] for col in model_cols])
            epi_unc = np.var(model_outs)
            alea_unc = unc - epi_unc
            epi_unc = np.sqrt(epi_unc)
            alea_unc = np.sqrt(alea_unc)
            unc = np.sqrt(unc)

            pred_col.append(prediction_linear)
            pred_logcol.append(prediction)
            pred_sd_totcol.append(unc)
            pred_sd_aleacol.append(alea_unc)
            pred_sd_epicol.append(epi_unc)

        df[f'Prediction_({unit})'] = pred_col
        df['Prediction_log10'] = pred_logcol
        df['SD_total'] = pred_sd_totcol
        df['SD_aleatoric'] = pred_sd_aleacol
        df['SD_epistemic'] = pred_sd_epicol

        return df
    